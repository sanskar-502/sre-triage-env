from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from openenv.core.env_server import Environment

from sre_triage.core.tasks import ScenarioDefinition, get_task, resolve_task_id
from sre_triage.schemas import ActionType, SREAction, SREObservation, SREState


@dataclass
class RuntimeState:
    node_running: bool = True
    mongo_port: int = 27017
    rogue_pid_active: bool = False
    jwt_secret_valid: bool = True
    disk_pressure: bool = False
    env_fixed_in_file: bool = False
    secret_fixed_in_file: bool = False
    logrotate_ran: bool = False
    discovered_categories: set[str] = field(default_factory=set)
    last_command: str | None = None
    episode_id: str = ""
    step_count: int = 0
    is_resolved: bool = False
    current_directory: str = "/var/www/mern-app"


class SREEnvironment(Environment):
    def __init__(self, difficulty: str = "medium", task_id: str | None = None):
        self.task_id = resolve_task_id(task_id, difficulty)
        self.scenario = get_task(self.task_id)
        self.runtime = RuntimeState()
        self.reset(difficulty=difficulty, task_id=task_id)

    def reset(self, difficulty: str | None = None, task_id: str | None = None) -> SREObservation:
        self.task_id = resolve_task_id(task_id, difficulty)
        self.scenario = get_task(self.task_id)
        self.runtime = RuntimeState(
            node_running=self.scenario.node_running,
            mongo_port=self.scenario.mongo_port,
            rogue_pid_active=self.scenario.rogue_pid_active,
            jwt_secret_valid=self.scenario.jwt_secret_valid,
            disk_pressure=self.scenario.disk_pressure,
            episode_id=str(uuid.uuid4()),
        )
        return self._build_observation(
            "Terminal session started. Type commands to investigate.",
            "",
            0,
            self._run_health_check(),
            0.0,
            False,
        )

    def step(self, action: SREAction) -> SREObservation:
        self.runtime.step_count += 1
        reward = 0.0

        if action.action_type == ActionType.EXECUTE_COMMAND:
            stdout, stderr, exit_code, reward = self._handle_command(action.command or "")
        elif action.action_type == ActionType.WRITE_FILE:
            stdout, stderr, exit_code, reward = self._handle_file_write(
                action.file_path or "",
                action.file_content or "",
            )
        elif action.action_type == ActionType.CHECK_HEALTH:
            stdout, stderr, exit_code = "Checking system health...", "", 0
        else:
            stdout, stderr, exit_code, reward = "", "Invalid action", 1, -0.1

        reward -= 0.02
        current_health = self._run_health_check()
        done = False
        if current_health == "HTTP 200 OK":
            self.runtime.is_resolved = True
            done = True
            reward += 1.0
        elif self.runtime.step_count >= self.scenario.max_steps:
            done = True

        return self._build_observation(stdout, stderr, exit_code, current_health, reward, done)

    def _run_health_check(self) -> str:
        if not self.runtime.node_running:
            return "HTTP 503 Service Unavailable"
        if self.runtime.mongo_port != 27017:
            return "HTTP 500 Internal Server Error"
        if not self.runtime.jwt_secret_valid:
            return "HTTP 401 Unauthorized"
        if self.runtime.rogue_pid_active:
            return "HTTP 504 Gateway Timeout"
        if self.runtime.disk_pressure:
            return "HTTP 507 Insufficient Storage"
        return "HTTP 200 OK"

    def _handle_command(self, cmd: str) -> tuple[str, str, int, float]:
        cmd_lower = cmd.strip().lower()
        if cmd_lower == self.runtime.last_command:
            self.runtime.last_command = cmd_lower
            return "", "Warning: identical command repeated. Try a different approach.", 0, -0.1
        self.runtime.last_command = cmd_lower

        if any(bad in cmd_lower for bad in ["rm ", "drop "]):
            return "", "bash: permission denied: destructive actions are blocked.", 1, -0.2
        if "apt" in cmd_lower or "yum" in cmd_lower:
            return "", "bash: package installation is blocked in this environment.", 1, -0.1

        if cmd_lower in {"pwd"}:
            return self.runtime.current_directory, "", 0, 0.0
        if cmd_lower in {"ls", "ls -la"}:
            return self._ls_output(), "", 0, self._discovery_reward("filesystem", 0.1)
        if any(x in cmd_lower for x in ["pm2 status", "pm2 list", "pm2 ls", "ps", "top", "htop"]):
            return self._process_output(), "", 0, self._discovery_reward("process", 0.2)
        if any(x in cmd_lower for x in ["cat logs/error.log", "cat logs/access.log", "tail logs/error.log"]):
            return self._log_output(), "", 0, self._discovery_reward("logs", 0.4)
        if "cat .env" in cmd_lower:
            return self._env_output(), "", 0, self._discovery_reward("env", 0.3)
        if any(x in cmd_lower for x in ["netstat", "ss ", "lsof -i"]):
            return self._network_output(), "", 0, self._discovery_reward("network", 0.2)
        if "systemctl status mongod" in cmd_lower or "mongod.conf" in cmd_lower:
            return self._mongo_output(cmd_lower), "", 0, self._discovery_reward("mongo", 0.2)
        if "df -h" in cmd_lower:
            return self._disk_output(), "", 0, self._discovery_reward("disk", 0.3)
        if "pm2 start all" in cmd_lower:
            if not self.runtime.node_running:
                self.runtime.node_running = True
                return "[PM2] App [server] started.", "", 0, 0.5
            return "[PM2] App [server] already online.", "", 0, 0.0
        if "pm2 restart all" in cmd_lower or "systemctl restart" in cmd_lower:
            return self._restart_output(), "", 0, 0.4 if self._restart_helped() else 0.0
        if "kill -9 8891" in cmd_lower:
            if self.runtime.rogue_pid_active:
                self.runtime.rogue_pid_active = False
                return "Process 8891 (kworker) terminated.", "", 0, 0.3
            return "bash: kill: (8891) - No such process", "", 1, 0.0
        if "logrotate -f /etc/logrotate.d/mern-app" in cmd_lower:
            if self.runtime.disk_pressure:
                self.runtime.disk_pressure = False
                self.runtime.logrotate_ran = True
                return "logrotate: archived logs vacuumed, disk pressure relieved.", "", 0, 0.5
            return "logrotate: no rotation needed.", "", 0, 0.0
        if "journalctl --disk-usage" in cmd_lower:
            return "Archived and active journals take up 3.6G in the file system.", "", 0, 0.1

        return "", f"bash: {cmd}: command not found", 127, -0.05

    def _handle_file_write(self, file_path: str, file_content: str) -> tuple[str, str, int, float]:
        normalized_path = file_path.strip().lower()
        if not normalized_path.endswith(".env"):
            return "", "Access denied: only .env updates are allowed.", 1, -0.1

        reward = 0.1
        if "27017" in file_content:
            self.runtime.env_fixed_in_file = True
            reward = 0.4
        if "correct-horse-battery-staple" in file_content:
            self.runtime.secret_fixed_in_file = True
            reward = max(reward, 0.4)
        return "SUCCESS: .env updated. Restart services for changes to take effect.", "", 0, reward

    def _restart_helped(self) -> bool:
        if self.runtime.env_fixed_in_file:
            self.runtime.mongo_port = 27017
        if self.runtime.secret_fixed_in_file:
            self.runtime.jwt_secret_valid = True
        return self.runtime.env_fixed_in_file or self.runtime.secret_fixed_in_file

    def _restart_output(self) -> str:
        if self._restart_helped():
            return "[PM2] [server] restarted. Configuration reloaded."
        return "[PM2][ERROR] Application restarted but the underlying issue remains."

    def _discovery_reward(self, category: str, reward: float) -> float:
        if category in self.runtime.discovered_categories:
            return 0.0
        self.runtime.discovered_categories.add(category)
        return reward

    def _ls_output(self) -> str:
        return (
            "total 48\n"
            "-rw-r--r-- 1 sreuser sreuser   412 Jan 15 03:20 .env\n"
            "-rw-r--r-- 1 sreuser sreuser  2847 Jan 14 22:15 server.js\n"
            "drwxr-xr-x 2 sreuser sreuser  4096 Jan 15 03:20 logs/\n"
            "drwxr-xr-x 2 sreuser sreuser  4096 Jan 14 22:15 config/\n"
        )

    def _process_output(self) -> str:
        rows = ["USER       PID %CPU %MEM COMMAND"]
        if self.runtime.node_running:
            rows.append("sreuser   1242  1.2  3.8 node server.js")
        else:
            rows.append("[PM2] No processes running. Use 'pm2 start all' to launch.")
        rows.append("mongodb    452  0.8 12.1 /usr/bin/mongod --config /etc/mongod.conf")
        if self.runtime.rogue_pid_active:
            rows.append("root      8891 98.2  0.1 [kworker/0:3+crypto]")
        return "\n".join(rows)

    def _env_output(self) -> str:
        mongo_port = 27017 if self.runtime.env_fixed_in_file else self.runtime.mongo_port
        secret = "correct-horse-battery-staple" if self.runtime.secret_fixed_in_file else "rotated-staging-secret"
        return (
            "# Application Configuration\n"
            "PORT=3000\n"
            f"MONGO_URI=mongodb://localhost:{mongo_port}/app\n"
            f"JWT_SECRET={secret}\n"
            "NODE_ENV=production\n"
            "# NOTE: MONGO_URI port was updated to 27018 for staging migration.\n"
            "# Revert to 27017 only if confirmed with the DBA team.\n"
        )

    def _network_output(self) -> str:
        rows = [
            "Proto Local Address State PID/Program",
            "tcp   127.0.0.1:27017 LISTEN 452/mongod",
        ]
        if self.runtime.node_running:
            rows.append("tcp   0.0.0.0:3000 LISTEN 1242/node")
        if self.runtime.rogue_pid_active:
            rows.append("tcp   0.0.0.0:4444 LISTEN 8891/kworker")
        return "\n".join(rows)

    def _mongo_output(self, cmd_lower: str) -> str:
        if "mongod.conf" in cmd_lower:
            return (
                "storage:\n"
                "  dbPath: /var/lib/mongodb\n"
                "net:\n"
                "  port: 27017\n"
                "  bindIp: 127.0.0.1\n"
            )
        return (
            "mongod.service - MongoDB Database Server\n"
            "Active: active (running)\n"
            "Status: Waiting for connections on port 27017\n"
        )

    def _disk_output(self) -> str:
        if self.runtime.disk_pressure:
            return (
                "Filesystem Size Used Avail Use% Mounted on\n"
                "/dev/sda1 40G 39G 200M 99% /\n"
            )
        return (
            "Filesystem Size Used Avail Use% Mounted on\n"
            "/dev/sda1 40G 22G 16G 58% /\n"
        )

    def _log_output(self) -> str:
        lines = [
            "[2024-01-15T05:12:33.021Z] [WARN] DeprecationWarning: Buffer() is deprecated.",
            "[2024-01-15T05:12:33.055Z] [INFO] Express server initializing on port 3000...",
        ]
        if not self.runtime.node_running:
            lines.extend(
                [
                    "[2024-01-15T05:14:08.112Z] [ERROR] pm2: Process 'server' exited with code 1",
                    "[2024-01-15T05:14:08.116Z] [HINT] Service appears stopped. Try 'pm2 status' or 'pm2 start all'.",
                ]
            )
        if self.runtime.mongo_port != 27017:
            lines.extend(
                [
                    f"[2024-01-15T05:12:33.204Z] [ERROR] MongoNetworkError: failed to connect on localhost:{self.runtime.mongo_port}",
                    "[2024-01-15T05:12:38.220Z] [HINT] Check application config (.env) for port overrides. DB expects port 27017.",
                ]
            )
        if not self.runtime.jwt_secret_valid:
            lines.extend(
                [
                    "[2024-01-15T05:12:39.002Z] [ERROR] JsonWebTokenError: invalid signature",
                    "[2024-01-15T05:12:39.003Z] [HINT] Secrets drift detected. Compare JWT_SECRET with the expected production rotation value.",
                ]
            )
        if self.runtime.rogue_pid_active:
            lines.append(
                "[2024-01-15T05:12:38.215Z] [WARN] High CPU detected on PID 8891 - possible crypto-miner."
            )
        if self.runtime.disk_pressure:
            lines.extend(
                [
                    "[2024-01-15T05:12:40.201Z] [ERROR] ENOSPC: no space left on device, write",
                    "[2024-01-15T05:12:40.202Z] [HINT] Check disk usage and rotate oversized logs.",
                ]
            )
        return "\n".join(lines)

    def _build_observation(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        health: str,
        reward: float,
        done: bool,
    ) -> SREObservation:
        return SREObservation(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            current_directory=self.runtime.current_directory,
            system_health_check=health,
            reward=reward,
            done=done,
            metadata={
                "task_id": self.task_id,
                "difficulty": self.scenario.difficulty,
            },
        )

    @property
    def state(self) -> SREState:
        return SREState(
            episode_id=self.runtime.episode_id,
            step_count=self.runtime.step_count,
            difficulty_level=self.scenario.difficulty,
            task_id=self.task_id,
            is_resolved=self.runtime.is_resolved,
        )
