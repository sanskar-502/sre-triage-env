# client.py — OpenEnv Client for SRE Triage Simulator
import httpx
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import SREAction, SREObservation, SREState
from dotenv import load_dotenv
load_dotenv()


class SREEnvClient(EnvClient[SREAction, SREObservation, SREState]):

    async def reset(self, difficulty: str = "medium") -> StepResult[SREObservation]:
        """Reset environment with a specific difficulty level.

        Overrides the base reset to pass difficulty as a JSON body
        parameter to the /reset endpoint, enabling multi-task evaluation.
        """
        # Support both manual __init__(base_url=...) and from_docker_image() construction
        base_url = getattr(self, '_base_url', None) or getattr(self, 'base_url', "http://localhost:7860")
        async with httpx.AsyncClient() as http:
            response = await http.post(
                f"{base_url}/reset",
                json={"difficulty": difficulty},
                timeout=30.0
            )
            response.raise_for_status()
            return self._parse_result(response.json())

    def _step_payload(self, action: SREAction) -> dict:
        """Converts our Pydantic Action into a JSON dict for the server."""
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[SREObservation]:
        """Converts the server's JSON response back into a Pydantic Observation."""
        reward = payload.get("reward", 0.0)
        done = payload.get("done", False)
        obs_data = dict(payload.get("observation", payload))  # copy to avoid mutation

        obs_data["reward"] = reward
        obs_data["done"] = done

        return StepResult(
            observation=SREObservation(**obs_data),
            reward=reward,
            done=done
        )

    def _parse_state(self, payload: dict) -> SREState:
        """Parses the environment metadata."""
        return SREState(**payload)