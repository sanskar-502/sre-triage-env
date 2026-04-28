from __future__ import annotations

import httpx
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from sre_triage.schemas import SREAction, SREObservation, SREState


class SREEnvClient(EnvClient[SREAction, SREObservation, SREState]):
    async def reset(
        self,
        difficulty: str = "medium",
        task_id: str | None = None,
    ) -> StepResult[SREObservation]:
        base_url = getattr(self, "_base_url", None) or getattr(self, "base_url", "http://localhost:7860")
        payload = {"difficulty": difficulty}
        if task_id:
            payload["task_id"] = task_id

        async with httpx.AsyncClient(timeout=30.0) as http:
            response = await http.post(f"{base_url}/reset", json=payload)
            response.raise_for_status()
            return self._parse_result(response.json())

    def _step_payload(self, action: SREAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[SREObservation]:
        reward = payload.get("reward", 0.0)
        done = payload.get("done", False)
        observation_payload = dict(payload.get("observation", payload))
        observation_payload["reward"] = reward
        observation_payload["done"] = done
        return StepResult(
            observation=SREObservation(**observation_payload),
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> SREState:
        return SREState(**payload)
