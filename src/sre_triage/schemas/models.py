from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from openenv.core import Action, Observation, State
from pydantic import Field, model_validator


class ActionType(str, Enum):
    EXECUTE_COMMAND = "execute_command"
    WRITE_FILE = "write_file"
    CHECK_HEALTH = "check_health"


class SREAction(Action):
    thought: Optional[str] = Field(
        default=None,
        description="Short reasoning summary before taking the action.",
    )
    action_type: ActionType = Field(..., description="The action to perform.")
    command: Optional[str] = Field(default=None, description="Shell command to execute.")
    file_path: Optional[str] = Field(default=None, description="Path to file to update.")
    file_content: Optional[str] = Field(default=None, description="Replacement file content.")

    @model_validator(mode="after")
    def validate_shape(self) -> "SREAction":
        if self.action_type == ActionType.EXECUTE_COMMAND and not self.command:
            raise ValueError("execute_command requires command")
        if self.action_type == ActionType.WRITE_FILE:
            if not self.file_path or self.file_content is None:
                raise ValueError("write_file requires file_path and file_content")
        if self.action_type == ActionType.CHECK_HEALTH:
            if self.command or self.file_path or self.file_content:
                raise ValueError("check_health does not accept command or file fields")
        return self


class SREObservation(Observation):
    stdout: str = Field(default="", description="Standard output from the last action.")
    stderr: str = Field(default="", description="Standard error from the last action.")
    exit_code: int = Field(default=0, description="Exit code of the last action.")
    current_directory: str = Field(
        default="/var/www/mern-app",
        description="Current working directory of the simulated shell.",
    )
    system_health_check: str = Field(default="", description="Current simulated health status.")
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SREState(State):
    episode_id: str
    step_count: int
    difficulty_level: str = Field(description="Task difficulty level.")
    task_id: str = Field(description="Current scenario identifier.")
    is_resolved: bool = Field(description="True when the simulated system is healthy.")


class ResetRequest(State):
    difficulty: Optional[str] = None
    task_id: Optional[str] = None


class HealthResponse(State):
    status: str
    service: str


class ReadyResponse(State):
    status: str
    service: str
    require_api_key: bool
