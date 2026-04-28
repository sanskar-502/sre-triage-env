from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping


@dataclass(frozen=True)
class ScenarioDefinition:
    task_id: str
    difficulty: str
    description: str
    node_running: bool = True
    mongo_port: int = 27017
    rogue_pid_active: bool = False
    jwt_secret_valid: bool = True
    disk_pressure: bool = False
    max_steps: int = 15
    expected_fix_commands: tuple[str, ...] = ()
    expected_file_markers: Mapping[str, str] = field(default_factory=dict)


TASKS: Dict[str, ScenarioDefinition] = {
    "easy_node_down": ScenarioDefinition(
        task_id="easy_node_down",
        difficulty="easy",
        description="Node.js service is stopped. Start the PM2 process.",
        node_running=False,
        expected_fix_commands=("pm2 start all",),
    ),
    "medium_config_drift": ScenarioDefinition(
        task_id="medium_config_drift",
        difficulty="medium",
        description="MongoDB port mismatch in .env. Fix the config and restart the service.",
        mongo_port=27018,
        expected_fix_commands=("pm2 restart all",),
        expected_file_markers={".env": "27017"},
    ),
    "hard_hybrid_failure": ScenarioDefinition(
        task_id="hard_hybrid_failure",
        difficulty="hard",
        description="Port mismatch plus a rogue high-CPU process causing timeouts.",
        mongo_port=27018,
        rogue_pid_active=True,
        expected_fix_commands=("pm2 restart all", "kill -9 8891"),
        expected_file_markers={".env": "27017"},
    ),
    "hard_bad_secret": ScenarioDefinition(
        task_id="hard_bad_secret",
        difficulty="hard",
        description="JWT secret is invalid, causing authentication failures after deploy.",
        jwt_secret_valid=False,
        expected_fix_commands=("pm2 restart all",),
        expected_file_markers={".env": "correct-horse-battery-staple"},
    ),
    "hard_disk_pressure": ScenarioDefinition(
        task_id="hard_disk_pressure",
        difficulty="hard",
        description="Disk pressure from runaway logs blocks the application from serving traffic.",
        disk_pressure=True,
        expected_fix_commands=("logrotate -f /etc/logrotate.d/mern-app",),
    ),
}


DIFFICULTY_DEFAULTS = {
    "easy": "easy_node_down",
    "medium": "medium_config_drift",
    "hard": "hard_hybrid_failure",
}


def get_task(task_id: str) -> ScenarioDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]


def resolve_task_id(task_id: str | None, difficulty: str | None) -> str:
    if task_id:
        return get_task(task_id).task_id
    if difficulty:
        return DIFFICULTY_DEFAULTS.get(difficulty, DIFFICULTY_DEFAULTS["medium"])
    return DIFFICULTY_DEFAULTS["medium"]


def task_manifest() -> List[dict]:
    return [
        {
            "id": task.task_id,
            "difficulty": task.difficulty,
            "description": task.description,
        }
        for task in TASKS.values()
    ]
