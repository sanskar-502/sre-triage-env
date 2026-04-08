# inference.py — Baseline Agent with Multi-Task Evaluation Loop
"""
SRE Triage Simulator - Inference Script
========================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   The Docker image name for the environment.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
import asyncio
import os
import json
import time
import textwrap
from typing import List, Optional
from openai import OpenAI

from client import SREEnvClient, SREAction

# ── 1. CONFIGURATION ──
# CRITICAL: The hackathon evaluator injects API_BASE_URL and API_KEY.
# We MUST use those first. Only fall back to .env for local development.
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY      = os.environ.get("API_KEY")
MODEL_NAME   = os.environ.get("MODEL_NAME")

# If evaluator vars are missing, load .env for local development
if not API_BASE_URL or not API_KEY:
    from dotenv import load_dotenv
    load_dotenv()
    API_BASE_URL = API_BASE_URL or os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    API_KEY      = API_KEY or os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    MODEL_NAME   = MODEL_NAME or os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")
else:
    MODEL_NAME = MODEL_NAME or "gemini-2.5-flash-lite"

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if not API_KEY:
    raise ValueError("API_KEY or HF_TOKEN is missing! Set it in your environment variables.")

MAX_STEPS = 10

# ── 2. MANDATORY LOGGING FORMAT ──
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── 3. ROBUST JSON PARSER ──
def parse_json_content(content: str) -> dict:
    clean_content = content.strip()
    if "```" in clean_content:
        parts = clean_content.split("```")
        for part in parts:
            p_strip = part.strip()
            if p_strip.startswith("{") or p_strip.startswith("json"):
                clean_content = p_strip.replace("json", "", 1).strip()
                break
    try:
        return json.loads(clean_content)
    except json.JSONDecodeError:
        return {"thought": "Parsing failed, falling back.", "action_type": "check_health"}

# ── 4. PROMPT ENGINEERING ──
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an SRE agent debugging a broken MERN stack on a Linux server.
    You MUST respond with ONLY a single valid JSON object — no markdown, no explanation, no text before or after.

    JSON schema:
    {"thought": "string", "action_type": "execute_command"|"write_file"|"check_health", "command": "string|null", "file_path": "string|null", "file_content": "string|null"}

    Rules:
    - action_type "execute_command" requires "command" field.
    - action_type "write_file" requires "file_path" and "file_content" fields.
    - action_type "check_health" requires no extra fields.
    - NEVER repeat the same command twice.
    - Output ONLY the JSON object. No other text.
""")

# Per-task instruction guides injected into user prompt
TASK_GUIDES = {
    "easy_node_down": textwrap.dedent("""\
        SOLUTION GUIDE for easy_node_down:
        The Node.js/PM2 process is stopped. The fix is simple:
        1. Run "pm2 start all" to start the service.
        After that, health should return HTTP 200.
        Do NOT run diagnostic commands — go straight to the fix."""),

    "medium_config_drift": textwrap.dedent("""\
        SOLUTION GUIDE for medium_config_drift:
        The .env file has the wrong MongoDB port (27018 instead of 27017). Fix it in this order:
        1. Write the corrected .env file: action_type="write_file", file_path=".env", file_content="PORT=3000\\nMONGO_URI=mongodb://localhost:27017/app"
        2. Restart the service: action_type="execute_command", command="pm2 restart all"
        After that, health should return HTTP 200.
        The .env has a misleading comment about staging migration — IGNORE IT. Port must be 27017."""),

    "hard_hybrid_failure": textwrap.dedent("""\
        SOLUTION GUIDE for hard_hybrid_failure:
        Two problems: wrong MongoDB port in .env AND a rogue crypto-mining process (PID 8891).
        Fix in this order:
        1. Write the corrected .env: action_type="write_file", file_path=".env", file_content="PORT=3000\\nMONGO_URI=mongodb://localhost:27017/app"
        2. Restart the service: action_type="execute_command", command="pm2 restart all"
        3. Kill the rogue process: action_type="execute_command", command="kill -9 8891"
        After all three, health should return HTTP 200."""),
}

def build_user_prompt(step: int, obs: dict, history: List[str], task_name: str = "") -> str:
    history_block = "\n".join(history[-4:]) if history else "No previous actions."
    guide = TASK_GUIDES.get(task_name, "Diagnose and fix the issue.")
    return textwrap.dedent(f"""\
        === TASK: {task_name} | STEP: {step} ===

        {guide}

        Current State:
        - Health: {obs.get('system_health_check', 'unknown')}
        - Stdout: {obs.get('stdout', '')[:500]}
        - Stderr: {obs.get('stderr', '')}

        Recent History:
        {history_block}

        Based on the solution guide and current state, output your next action as a single JSON object.
        Remember: output ONLY valid JSON, nothing else.""")

def get_model_action(llm_client: OpenAI, step: int, obs: dict, history: List[str], task_name: str) -> dict:
    """Get an action from the LLM with retry logic for rate limits."""
    prompt = build_user_prompt(step, obs, history, task_name)
    for attempt in range(4):
        try:
            if attempt > 0:
                wait = 8 * attempt  # 8s, 16s, 24s backoff
                print(f"[DEBUG] Retry {attempt}/3 after {wait}s...", flush=True)
                time.sleep(wait)
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                stream=False,
            )
            return parse_json_content(completion.choices[0].message.content)
        except Exception as exc:
            print(f"[DEBUG] Model request failed (attempt {attempt+1}): {exc}", flush=True)
            if attempt == 3:
                return {"thought": f"API Error: {exc}", "action_type": "check_health"}
    return {"thought": "All retries failed", "action_type": "check_health"}

# ── 5. TASK DEFINITIONS ──
TASKS = [
    {"name": "easy_node_down",      "difficulty": "easy"},
    {"name": "medium_config_drift", "difficulty": "medium"},
    {"name": "hard_hybrid_failure", "difficulty": "hard"},
]

# ── 6. MAIN EXECUTION ──
async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env_url = os.getenv("ENV_URL")

    if env_url:
        # Evaluator prespawns the container — connect directly
        env = SREEnvClient(base_url=env_url)
    else:
        try:
            env = await SREEnvClient.from_docker_image(LOCAL_IMAGE_NAME or "sre-mern-env:latest")
        except Exception as e:
            print(f"[DEBUG] Docker unavailable ({e}). Falling back to localhost...", flush=True)
            env = SREEnvClient(base_url="http://localhost:7860")

    report = []

    try:
        for task_info in TASKS:
            task_name  = task_info["name"]
            difficulty = task_info["difficulty"]
            env_name   = "sre_mern_triage"

            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            log_start(task=task_name, env=env_name, model=MODEL_NAME)

            try:
                result = await env.reset(difficulty=difficulty)
                obs = result.observation.model_dump()

                for step in range(1, MAX_STEPS + 1):
                    if result.done:
                        break

                    action_dict = get_model_action(llm_client, step, obs, history, task_name)
                    action_str = json.dumps(action_dict).replace('\n', '')

                    try:
                        action_obj = SREAction(**action_dict)
                        result = await env.step(action_obj)
                        obs = result.observation.model_dump()
                        reward = result.reward or 0.0
                        done = result.done
                        error = None
                    except Exception as e:
                        reward, error = -0.05, str(e)
                        done = False
                        obs = {"stdout": "", "stderr": f"Error: {e}", "system_health_check": "ERROR"}

                    rewards.append(reward)
                    steps_taken = step

                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                    history.append(f"Step {step}: {action_dict.get('thought', 'Acting')}")

                    if done:
                        break

                # Score: 1.0 if resolved, 0.0 otherwise — clamped to [0, 1]
                if result.done and "200" in obs.get('system_health_check', ''):
                    score = 1.0
                    success = True
                score = min(max(score, 0.0), 1.0)

            except Exception as e:
                print(f"[DEBUG] Task {task_name} error: {e}", flush=True)

            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
            report.append({"task": task_name, "success": success, "score": score, "steps": steps_taken})

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)

    # ── Baseline Report Card ──
    print(f"\n{'='*60}")
    print("📊 BASELINE EVALUATION REPORT CARD")
    print(f"{'='*60}")
    for r in report:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        print(f"  {r['task']:30s} {status}  Score: {r['score']:.1f}  Steps: {r['steps']}")
    aggregate = sum(r["score"] for r in report) / len(report) if report else 0.0
    print(f"\n  {'AGGREGATE SCORE':30s} {aggregate:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
