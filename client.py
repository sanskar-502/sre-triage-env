# client.py
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import SREAction, SREObservation, SREState
from dotenv import load_dotenv  # Add this
load_dotenv()

class SREEnvClient(EnvClient[SREAction, SREObservation, SREState]):
    def _step_payload(self, action: SREAction) -> dict:
        """Converts our Pydantic Action into a JSON dict for the server."""
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[SREObservation]:
        """Converts the server's JSON response back into a Pydantic Observation."""
        
        # 1. Pull the variables once to avoid redundant .get() calls
        reward = payload.get("reward", 0.0)
        done = payload.get("done", False)
        obs_data = payload.get("observation", {})
        
        # 2. Inject RL variables into the observation data 
        # (This ensures SREObservation matches your models.py definition)
        obs_data["reward"] = reward
        obs_data["done"] = done

        # 3. Return the structured StepResult
        return StepResult(
            observation=SREObservation(**obs_data),
            reward=reward,
            done=done
        )

    def _parse_state(self, payload: dict) -> SREState:
        """Parses the environment metadata."""
        return SREState(**payload)