#models.py
from dotenv import load_dotenv  # Add this
load_dotenv()
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import Field
from openenv.core import Action, Observation, State

# This Enum prevents the AI from "hallucinating" new action types
#1

class ActionType(str, Enum):
    EXECUTE_COMMAND = "execute_command"
    WRITE_FILE = "write_file"
    CHECK_HEALTH = "check_health"

#2

class SREAction(Action):
    thought: Optional[str] = Field(None, description="Your internal reasoning before taking action.")
    action_type: ActionType = Field(..., description="The type of action to perform.")
    command: Optional[str] = Field(None, description="Shell command (e.g., 'ls').")
    file_path: Optional[str] = Field(None, description="Path to file for WRITE_FILE action.")
    file_content: Optional[str] = Field(None, description="New content for WRITE_FILE action.")


# ---------------------------------------------------------
# 3. OBSERVATIONS (What the Agent sees after acting)
# ---------------------------------------------------------
class SREObservation(Observation):
    """
    Strict representation of the terminal output and environment status.
    """
    stdout: str = Field(description="Standard output from the last executed command.")
    stderr: str = Field(description="Standard error from the last executed command.")
    exit_code: int = Field(description="Exit code of the last command (0 usually means success).")
    current_directory: str = Field(description="The current working directory.")
    system_health_check: str = Field(description="Automated ping result of the API (e.g., 'HTTP 502' or 'HTTP 200').")
    
    # Required OpenEnv base fields for RL Training
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------
# 4. STATE (Internal episode metadata)
# ---------------------------------------------------------
class SREState(State):
    """
    Used by the grader and server to track the episode's lifecycle.
    """
    episode_id: str
    step_count: int
    difficulty_level: str = Field(description="'easy', 'medium', or 'hard'")
    is_resolved: bool = Field(description="True if the MERN stack is fully operational (HTTP 200).")