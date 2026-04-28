from __future__ import annotations

import asyncio
import json
import textwrap
import time
from statistics import mean
from typing import Any

import litellm

from sre_triage.client import SREEnvClient
from sre_triage.core.tasks import TASKS
from sre_triage.schemas import SREAction
from sre_triage.settings import settings


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an SRE agent debugging a broken MERN stack.
    Output only a single valid JSON object representing your next action.

    Rules:
    - Prefer evidence collection before file changes unless the issue is already obvious from observations.
    - Never repeat the same command twice in a row.
    - Make minimal, reversible changes.
    """
)

TASK_CONTEXT = {
    task_id: definition.description for task_id, definition in TASKS.items()
}

def build_user_prompt(task_id: str, step: int, obs: dict[str, Any], history: list[str]) -> str:
    return textwrap.dedent(
        f"""\
        Task: {task_id}
        Incident: {TASK_CONTEXT[task_id]}
        Step: {step}

        Current health: {obs.get("system_health_check", "unknown")}
        Stdout:
        {obs.get("stdout", "")[:800]}

        Stderr:
        {obs.get("stderr", "")[:300]}

        Recent history:
        {chr(10).join(history[-5:]) if history else "No previous actions."}

        Return the single best next action as JSON only.
        """
    )


def get_model_action(task_id: str, step: int, obs: dict[str, Any], history: list[str]) -> dict[str, Any]:
    prompt = build_user_prompt(task_id, step, obs, history)
    for attempt in range(4):
        try:
            if attempt:
                time.sleep(4 * attempt)
            
            # Using LiteLLM structured outputs (which forces the model to return valid JSON matching the Pydantic schema)
            completion = litellm.completion(
                model=settings.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=SREAction,
                temperature=0.2,
                max_tokens=250,
                api_key=settings.api_key,
                api_base=settings.api_base_url,
            )
            content = completion.choices[0].message.content
            return json.loads(content)
        except Exception as exc:
            if attempt == 3:
                return {"thought": f"API error: {exc}", "action_type": "check_health"}
    return {"thought": "Model retries exhausted.", "action_type": "check_health"}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


async def run_task(env: SREEnvClient, task_id: str) -> dict[str, Any]:
    history: list[str] = []
    rewards: list[float] = []
    result = await env.reset(task_id=task_id)
    observation = result.observation.model_dump()
    log_start(task_id, "sre_mern_triage", settings.model_name)

    last_error = None
    for step in range(1, settings.benchmark_max_steps + 1):
        action_dict = get_model_action(task_id, step, observation, history)
        action_repr = json.dumps(action_dict).replace("\n", "")
        try:
            action = SREAction(**action_dict)
            result = await env.step(action)
            observation = result.observation.model_dump()
            reward = result.reward or 0.0
            done = result.done
            last_error = None
        except Exception as exc:
            reward = -0.05
            done = False
            observation = {"stdout": "", "stderr": str(exc), "system_health_check": "ERROR"}
            last_error = str(exc)

        rewards.append(reward)
        history.append(f"Step {step}: {action_dict}")
        log_step(step, action_repr, reward, done, last_error)
        if done:
            break

    success = bool(result.done and "200" in observation.get("system_health_check", ""))
    score = 1.0 if success else 0.0
    log_end(success, len(rewards), score, rewards)
    return {
        "task": task_id,
        "success": success,
        "steps": len(rewards),
        "score": score,
        "final_health": observation.get("system_health_check"),
    }


async def main() -> None:
    if not settings.api_key:
        raise ValueError("Set HF_TOKEN or API_KEY in your environment variables.")

    env_url = settings.env_url
    env = SREEnvClient(base_url=env_url)
    report = []
    try:
        for task_id in TASKS:
            report.append(await run_task(env, task_id))
    finally:
        await env.close()

    successes = sum(1 for item in report if item["success"])
    average_steps = mean([item["steps"] for item in report]) if report else 0.0
    print("\nBenchmark summary")
    print("=" * 60)
    for item in report:
        print(
            f"{item['task']:24s} success={item['success']} steps={item['steps']} health={item['final_health']}"
        )
    print(f"pass_rate={successes}/{len(report)} average_steps={average_steps:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
