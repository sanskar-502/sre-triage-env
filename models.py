# models.py — Pydantic models for the SRE Triage Environment
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import Field
from openenv.core import Action, Observation, State


class ActionType(str, Enum):
    """Constrained action types to prevent hallucinations."""
    EXECUTE_COMMAND = "execute_command"
    WRITE_FILE = "write_file"
    CHECK_HEALTH = "check_health"


class SREAction(Action):
    """Structured action the agent sends to the environment."""
    thought: Optional[str] = Field(None, description="Your internal reasoning before taking action.")
    action_type: ActionType = Field(..., description="The type of action to perform.")
    command: Optional[str] = Field(None, description="Shell command (e.g., 'ls').")
    file_path: Optional[str] = Field(None, description="Path to file for WRITE_FILE action.")
    file_content: Optional[str] = Field(None, description="New content for WRITE_FILE action.")


class SREObservation(Observation):
    """What the agent sees after taking an action."""
    stdout: str = Field(default="", description="Standard output from the last executed command.")
    stderr: str = Field(default="", description="Standard error from the last executed command.")
    exit_code: int = Field(default=0, description="Exit code of the last command (0 usually means success).")
    current_directory: str = Field(default="/var/www/mern-app", description="The current working directory.")
    system_health_check: str = Field(default="", description="Automated ping result of the API (e.g., 'HTTP 502' or 'HTTP 200').")

    # Required OpenEnv base fields for RL Training
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SREState(State):
    """Internal episode metadata used by grader and server."""
    episode_id: str
    step_count: int
    difficulty_level: str = Field(description="'easy', 'medium', or 'hard'")
    is_resolved: bool = Field(description="True if the MERN stack is fully operational (HTTP 200).")