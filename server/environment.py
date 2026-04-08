#environment.py
import uuid
from openenv.core.env_server import Environment
from models import SREAction, SREObservation, SREState, ActionType
from dotenv import load_dotenv  # Add this
load_dotenv()

class SREEnvironment(Environment):
    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self._discovered_categories = set()
        self.reset()

    def reset(self) -> SREObservation:
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._is_resolved = False
        self._current_dir = "/var/www/mern-app"
        self._discovered_categories = set() 
        
        # --- State Machine Variables ---
        self.node_running = True
        self.mongo_port = 27018 if self.difficulty == "medium" else 27017
        self.rogue_pid_active = True if self.difficulty == "hard" else False
        
        # NEW: Tracks if the file is edited, but service hasn't restarted yet
        self.env_fixed_in_file = False 
        
        health = self._run_health_check()
        return self._build_observation("Terminal session started. Type commands to investigate.", "", 0, health, 0.0, False)

    def step(self, action: SREAction) -> SREObservation:
        self._step_count += 1
        reward = 0.0
        
        if action.action_type == ActionType.EXECUTE_COMMAND:
            stdout, stderr, exit_code, step_reward = self._handle_command(action.command)
            reward += step_reward
        elif action.action_type == ActionType.WRITE_FILE:
            stdout, stderr, exit_code, step_reward = self._handle_file_write(action.file_path, action.file_content)
            reward += step_reward
        elif action.action_type == ActionType.CHECK_HEALTH:
            stdout, stderr, exit_code = "Checking system health...", "", 0
        else:
            stdout, stderr, exit_code, reward = "", "Invalid Action Enum", 1, -0.1

        current_health = self._run_health_check()
        done = False
        if current_health == "HTTP 200 OK":
            self._is_resolved = True
            done = True
            reward += 1.0  

        return self._build_observation(stdout, stderr, exit_code, current_health, reward, done)

    def _run_health_check(self) -> str:
        if not self.node_running: return "HTTP 502 Bad Gateway"
        if self.mongo_port != 27017: return "HTTP 500 Internal Server Error"
        if self.rogue_pid_active: return "HTTP 504 Gateway Timeout"
        return "HTTP 200 OK"

    def _handle_command(self, cmd: str):
        cmd_lower = (cmd or "").strip().lower()

        def get_discovery_reward(category: str, value: float) -> float:
            if category not in self._discovered_categories:
                self._discovered_categories.add(category)
                return value
            return 0.0 

        # --- 1. Deflections & Constraints ---
        if any(bad in cmd_lower for bad in ["rm ", "drop"]):
            return "", "bash: permission denied: destructive actions are blocked.", 1, -0.2
            
        if "apt" in cmd_lower or "yum" in cmd_lower:
            return "", "bash: apt: permission denied. Use existing tools: netstat, ps, cat, lsof.", 1, -0.1

        if any(x in cmd_lower for x in ["systemctl status mongod", "service mongod status", "mongod.log"]):
            return "mongod.service - MongoDB Database Server\n   Active: active (running)\n   Main PID: 452 (mongod)\n   Status: 'Waiting for connections on port 27017'", "", 0, 0.1

        if "mongod.conf" in cmd_lower:
            return "net:\n  port: 27017\n  bindIp: 127.0.0.1\n# POLICY: DO NOT CHANGE PORT. STACK REQUIRES 27017.", "", 0, 0.2

        # --- 2. The Fix Application (Restarting the App) ---
        if "pm2 restart" in cmd_lower or "systemctl restart" in cmd_lower or "restart" in cmd_lower:
            if self.env_fixed_in_file:
                self.mongo_port = 27017 # THE FIX IS APPLIED TO THE SERVER MEMORY
                return "Restarting Node... Configuration applied successfully.", "", 0, 0.4
            return "Restarting Node... (Warning: Error connecting to DB on 27018)", "", 0, 0.0

        # --- 3. Navigation ---
        if "pwd" in cmd_lower: return self._current_dir, "", 0, 0.0
        if "ls" in cmd_lower:
            output = "total 12\n-rw-r--r-- .env\ndrwxr-xr-x logs/\n-rw-r--r-- server.js"
            return output, "", 0, get_discovery_reward('ls', 0.1)

        # --- 4. Logs ---
        if "log" in cmd_lower or "error.log" in cmd_lower:
            log_output = (
                f"[ERROR] MongoNetworkError: failed to connect to [localhost:{self.mongo_port}]\n"
                f"[HINT] Check application local config (.env) for port overrides."
            )
            return log_output, "", 0, get_discovery_reward('logs', 0.4)

        # --- 5. Semantic Router: Networking & Processes ---
        if any(tool in cmd_lower for tool in ["netstat", "ss", "lsof", "tcpdump"]):
            output = (
                "Active Internet connections\n"
                "Proto Recv-Q Send-Q Local Address           State\n"
                "tcp        0      0 127.0.0.1:27017         LISTEN\n"
                "tcp        0      0 0.0.0.0:3000            LISTEN"
            )
            return output, "", 0, get_discovery_reward('network', 0.2)

        if any(tool in cmd_lower for tool in ["pm2", "ps", "top", "htop"]):
            return "PID 1242: node server.js (Active)\nPID 452: mongod (Active)\nPID 8891: kworker (high cpu)", "", 0, get_discovery_reward('process', 0.2)

        # --- 6. Config Inspection ---
        if "cat" in cmd_lower and ".env" in cmd_lower:
            # Show the state based on whether they edited it or not
            current_port = 27017 if self.env_fixed_in_file else self.mongo_port
            return f"PORT=3000\nMONGO_URI=mongodb://localhost:{current_port}/app", "", 0, get_discovery_reward('env', 0.3)

        # --- 7. Fallback ---
        if "which" in cmd_lower or "/bin/" in cmd_lower or "/usr/bin/" in cmd_lower:
            return f"/usr/bin/{cmd_lower.split()[-1]}", "", 0, 0.0
            
        return "", f"bash: {cmd}: command not found", 127, -0.05

    def _handle_file_write(self, file_path: str, file_content: str):
        if (file_path or "").lower().endswith(".env"):
            if "27017" in (file_content or ""):
                self.env_fixed_in_file = True # Change the file, but don't fix the server until restart
                return "SUCCESS: .env updated. (Hint: Changes require a service restart to take effect).", "", 0, 0.4
            return "File updated, but port configuration remains invalid.", "", 0, 0.1
        return "", "Access Denied: You only have permission to edit app configs.", 1, -0.1

    def _build_observation(self, stdout, stderr, exit_code, health, reward, done):
        return SREObservation(stdout=stdout, stderr=stderr, exit_code=exit_code, 
                            current_directory=self._current_dir, system_health_check=health,
                            done=done, reward=reward)

    @property
    def state(self) -> SREState:
        return SREState(episode_id=self._episode_id, step_count=self._step_count,
                        difficulty_level=self.difficulty, is_resolved=self._is_resolved)