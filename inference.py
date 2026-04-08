import os
import json
import asyncio
import textwrap
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv  # Add this
load_dotenv()

from client import SREEnvClient, SREAction

# --- 1. CONFIGURATION & ENVIRONMENT ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing! Set it in your environment variables.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# --- 2. LOGGING UTILITIES (STRICT FORMAT) ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: dict, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_str = json.dumps(action).replace('\n', '')
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- 3. HELPER: ROBUST JSON PARSER ---
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

# --- 4. PROMPT ENGINEERING ---
SYSTEM_PROMPT = textwrap.dedent("""
    You are an elite Site Reliability Engineer (SRE).
    You are debugging a broken MERN stack application on a Linux server.

    You interact with the environment by outputting ONLY valid JSON matching this exact schema:
    {
      "thought": "1-2 sentences explaining your logic and what you are trying to achieve.",
      "action_type": "execute_command" | "write_file" | "check_health",
      "command": "string (REQUIRED if action_type is execute_command, otherwise null)",
      "file_path": "string (REQUIRED if action_type is write_file, otherwise null)",
      "file_content": "string (REQUIRED if action_type is write_file, otherwise null)"
    }

    CRITICAL DIRECTIVES - READ CAREFULLY:
    1. NEVER REPEAT COMMANDS: If you just ran 'cat .env', DO NOT run it again. You must take a new action or fix the file.
    2. THE SRE WORKFLOW: 
       - If logs show a database connection error, DO NOT just stare at the .env file.
       - You MUST run 'netstat' or 'ss' to find out what port MongoDB is ACTUALLY listening on (usually 27017).
       - Once you find the discrepancy, use "action_type": "write_file" to fix the .env file to match the real port.
    3. APPLY CHANGES: After using write_file, you MUST restart the application using 'pm2 restart all' with "execute_command".
    4. STRICT ACTION TYPES: Use "execute_command" for bash commands. "check_health" takes no arguments and just checks if the site is back online.
""").strip()

def build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "No previous actions."
    return textwrap.dedent(f"""
        Step: {step}
        Health: {obs.get('system_health_check')}
        Stdout: {obs.get('stdout')}
        Stderr: {obs.get('stderr')}

        Recent History:
        {history_block}
    """).strip()

# --- 5. MAIN EXECUTION LOOP ---
MAX_STEPS = 10

async def main():
    task_name = os.getenv("SRE_TASK", "medium_config_drift")
    env_name = "sre_mern_triage"
    log_start(task=task_name, env=env_name, model=MODEL_NAME)
    
    history, rewards = [], []
    steps_taken, score, success = 0, 0.0, False

    local_url = "http://127.0.0.1:8000"
    async with SREEnvClient(base_url=local_url) as env:
        result = await env.reset()
        obs = result.observation.model_dump()
        
        for step in range(1, MAX_STEPS + 1):
            if result.done: break
                
            prompt = build_user_prompt(step, obs, history)
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                action_dict = parse_json_content(response.choices[0].message.content)
            except Exception as e:
                action_dict = {"thought": f"API Error: {e}", "action_type": "check_health"}

            try:
                action_obj = SREAction(**action_dict)
                result = await env.step(action_obj)
                obs = result.observation.model_dump()
                step_reward = result.reward or 0.0
                step_error = None
            except Exception as e:
                step_reward, step_error = -0.05, str(e)
                obs = {"stdout": "", "stderr": f"Error: {e}", "system_health_check": "ERROR"}

            rewards.append(step_reward)
            steps_taken = step
            log_step(step, action_dict, step_reward, result.done, step_error)
            history.append(f"Step {step}: {action_dict.get('thought', 'Acting')}")

            if result.done: break

        # Scoring Logic
        if result.done and obs.get('system_health_check') == "HTTP 200 OK":
            score = 1.0
            success = True

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())













