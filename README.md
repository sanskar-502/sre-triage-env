---
title: SRE Triage Simulator
emoji: 🔧
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - sre
  - infrastructure
  - mern-stack
  - triage
pinned: false
---

# 🔧 SRE Triage Simulator

> **A deterministic reinforcement learning environment for training AI agents to diagnose and resolve real-world infrastructure failures in a MERN stack application.**

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-v1%20Compliant-brightgreen)](https://openenv.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](Dockerfile)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow)](pyproject.toml)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Why SRE Triage?](#why-sre-triage)
- [Architecture](#architecture)
- [Task Definitions](#task-definitions)
- [Action & Observation Spaces](#action--observation-spaces)
- [Reward Design](#reward-design)
- [Grading System](#grading-system)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Baseline Agent](#baseline-agent)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Deployment](#deployment)

---

## Overview

The SRE Triage Simulator is a high-fidelity OpenEnv environment that simulates a **broken MERN (MongoDB, Express, React, Node.js) stack** on a Linux server. An AI agent takes the role of a Site Reliability Engineer (SRE), diagnosing and resolving infrastructure failures through structured terminal commands and configuration management.

Unlike toy environments or games, this system models a **real professional task** — the kind of incident response that SRE teams handle daily at companies like Google, Netflix, and Meta.

### Key Features

| Feature | Description |
|---------|-------------|
| **3 Difficulty Levels** | Easy → Medium → Hard with genuine complexity progression |
| **Deterministic State Machine** | 100% reproducible episodes for fair evaluation |
| **Rich Reward Shaping** | Partial credit for diagnostic progress, not just binary success |
| **Realistic Log Output** | Timestamps, red herrings, deprecation warnings — just like production |
| **Efficiency Incentives** | Step penalties and repeat-command penalties encourage optimal behavior |
| **Anti-Exploit Guards** | Destructive commands blocked, unauthorized installs penalized |

---

## Why SRE Triage?

```
"The most expensive bug is the one that takes 3 hours to diagnose but 3 seconds to fix."
```

SRE triage is a uniquely suitable domain for RL/agent benchmarking because:

1. **It's a real job** — Site Reliability Engineers at every major tech company perform this task daily
2. **It requires multi-step reasoning** — Read logs → Identify root cause → Fix config → Restart services → Verify
3. **There are clear success criteria** — HTTP 200 = system healthy, anything else = still broken
4. **Difficulty scales naturally** — From a single stopped service to cascading multi-component failures
5. **Red herrings exist** — Production logs are noisy. Agents must distinguish signal from noise

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SRE Triage Simulator                      │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   FastAPI     │◄───│ State Machine│    │ Programmatic │  │
│  │   Server      │    │  (episode    │───►│   Grader     │  │
│  │  (app.py)     │───►│   logic)     │    │  (health     │  │
│  │               │    │ environment  │    │   check)     │  │
│  └──────┬───────┘    │    .py       │    └──────────────┘  │
│         │            └──────────────┘                       │
│         │                    ▲                               │
│    /reset  /step  /state     │                               │
│         │                    │                               │
│         ▼                    │                               │
│  ┌──────────────┐    ┌──────┴───────┐                       │
│  │  Pydantic     │    │  Typed       │                       │
│  │  Models       │    │  Actions     │                       │
│  │ (models.py)   │    │  & Obs       │                       │
│  └──────────────┘    └──────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
          ▲                                    │
          │         HTTP JSON API              │
          │                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     Baseline Agent                           │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  OpenAI       │    │  EnvClient   │    │  Multi-Task  │  │
│  │  Client       │───►│  (client.py) │───►│  Loop        │  │
│  │  (LLM calls)  │    │              │    │  (inference  │  │
│  │               │    │              │    │     .py)     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Task Definitions

### 🟢 Easy — Node.js Service Down

| Property | Value |
|----------|-------|
| **Root Cause** | Node.js/PM2 process is stopped |
| **Initial Health** | `HTTP 503 Service Unavailable` |
| **Fix** | `pm2 start all` |
| **Minimum Steps** | 1 |
| **Agent Must** | Check process status → Start the service |

```
Agent: pm2 status → "No processes running"
Agent: pm2 start all → "Process started successfully"
Health: HTTP 200 OK ✅
```

### 🟡 Medium — Configuration Drift

| Property | Value |
|----------|-------|
| **Root Cause** | `.env` has wrong MongoDB port (27018 instead of 27017) |
| **Initial Health** | `HTTP 500 Internal Server Error` |
| **Fix** | Edit `.env` to fix port + `pm2 restart all` |
| **Minimum Steps** | 2 |
| **Agent Must** | Read logs → Identify port mismatch → Fix config → Restart |

**Challenge:** The `.env` file contains a misleading comment:
```
# NOTE: MONGO_URI port was updated to 27018 for staging migration.
# Revert to 27017 only if confirmed with the DBA team.
```
The agent must reason that this comment is misleading and fix the port anyway.

### 🔴 Hard — Hybrid Cascading Failure

| Property | Value |
|----------|-------|
| **Root Cause** | Port mismatch + rogue crypto-mining process consuming 98% CPU |
| **Initial Health** | `HTTP 500 Internal Server Error` |
| **Fix** | Edit `.env` + `pm2 restart all` + `kill -9 8891` |
| **Minimum Steps** | 3 |
| **Agent Must** | Diagnose port issue + fix config + restart + identify and kill rogue process |

**Challenge:** After fixing the config and restarting, the health check returns `HTTP 504 Gateway Timeout` instead of 200. The agent must investigate further, identify PID 8891 (`[kworker/0:3+crypto]`) consuming 98.2% CPU, and terminate it.

```
Step 1: cat logs/error.log → "MongoNetworkError... port 27018"
                              "High CPU detected on PID 8891 — possible crypto-miner"
Step 2: write_file .env → Fix MONGO_URI to port 27017
Step 3: pm2 restart all → "Configuration reloaded"
        Health: HTTP 504 Gateway Timeout (still broken!)
Step 4: kill -9 8891 → "Process 8891 (kworker) terminated"
        Health: HTTP 200 OK ✅
```

---

## Action & Observation Spaces

### Action Space (Structured JSON)

```json
{
  "thought": "My reasoning about what to do next",
  "action_type": "execute_command | write_file | check_health",
  "command": "shell command (required for execute_command)",
  "file_path": "path to file (required for write_file)",
  "file_content": "new file content (required for write_file)"
}
```

| Action Type | Description | Example |
|-------------|-------------|---------|
| `execute_command` | Run a shell command | `{"action_type": "execute_command", "command": "pm2 status"}` |
| `write_file` | Write/overwrite a file | `{"action_type": "write_file", "file_path": ".env", "file_content": "PORT=3000\nMONGO_URI=mongodb://localhost:27017/app"}` |
| `check_health` | Trigger a health check | `{"action_type": "check_health"}` |

### Supported Commands

| Category | Commands | Discovery Reward |
|----------|----------|-----------------|
| **Process Management** | `pm2 status`, `pm2 start`, `pm2 restart`, `ps aux`, `top` | +0.2 |
| **Log Inspection** | `cat logs/error.log`, `cat logs/access.log` | +0.4 |
| **Network Diagnostics** | `netstat -tlnp`, `ss -tlnp`, `lsof -i` | +0.2 |
| **Config Inspection** | `cat .env`, `cat config/database.yml` | +0.3 |
| **Service Management** | `systemctl status mongod`, `cat /etc/mongod.conf` | +0.1–0.2 |
| **System Diagnostics** | `uptime`, `free -m`, `df -h`, `pwd`, `ls` | +0.0–0.1 |
| **Process Control** | `kill -9 <PID>` | +0.3 |
| **Blocked** | `rm`, `drop`, `apt`, `yum` | −0.1 to −0.2 |

### Observation Space

```json
{
  "stdout": "Standard output from the command",
  "stderr": "Standard error (empty on success)",
  "exit_code": 0,
  "current_directory": "/var/www/mern-app",
  "system_health_check": "HTTP 503 Service Unavailable",
  "reward": 0.18,
  "done": false
}
```

---

## Reward Design

The reward function provides **meaningful signal throughout the trajectory**, not just sparse binary success/failure.

### Reward Components

| Component | Value | When |
|-----------|-------|------|
| **Discovery rewards** | +0.1 to +0.4 | First time agent inspects a new diagnostic category |
| **Fix rewards** | +0.3 to +0.5 | Agent applies a correct fix (start service, fix config, kill process) |
| **Success bonus** | +1.0 | Health check returns HTTP 200 (episode complete) |
| **Step penalty** | −0.02 | Applied every step to encourage efficiency |
| **Repeat penalty** | −0.1 | Agent runs the exact same command consecutively |
| **Destructive penalty** | −0.2 | Agent attempts `rm`, `drop`, or other destructive commands |
| **Install penalty** | −0.1 | Agent attempts `apt install`, `yum install`, etc. |
| **Unknown command** | −0.05 | Agent runs a command the environment doesn't recognize |

### Example Reward Trajectory (Medium Difficulty)

```
Step 1: cat logs/error.log  → reward = +0.40 - 0.02 = +0.38  (discovery: logs)
Step 2: cat .env            → reward = +0.30 - 0.02 = +0.28  (discovery: config)
Step 3: write_file .env     → reward = +0.40 - 0.02 = +0.38  (correct fix)
Step 4: pm2 restart all     → reward = +0.40 + 1.0 - 0.02 = +1.38  (apply + success!)
                              ────────
                              Total: +2.42
```

### Episode Boundaries

- **Success**: Health check returns `HTTP 200 OK` → `done = true`
- **Timeout**: 15 steps reached without resolution → `done = true`, no success bonus
- **No early termination**: Agent can always keep trying until timeout

---

## Grading System

The grader is a **hierarchical, deterministic health-check function** that maps system state to HTTP status codes:

```
┌─────────────────────────┐
│ Node.js running?        │──── No ──→ HTTP 503 Service Unavailable
│                         │
│         Yes             │
│         ▼               │
│ MongoDB port correct?   │──── No ──→ HTTP 500 Internal Server Error
│                         │
│         Yes             │
│         ▼               │
│ Rogue process active?   │── Yes ──→ HTTP 504 Gateway Timeout
│                         │
│         No              │
│         ▼               │
│   HTTP 200 OK ✅        │
└─────────────────────────┘
```

| Health Code | Meaning | Fix Required |
|-------------|---------|-------------|
| `503` | Node.js service is down | `pm2 start all` |
| `500` | Database connection misconfigured | Fix `.env` + `pm2 restart` |
| `504` | Rogue process causing timeouts | `kill -9 <PID>` |
| `200` | All systems operational | Episode complete ✅ |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- A Hugging Face API token (for baseline agent)

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Tests

```bash
python test_logic.py
```

Expected output:
```
✅ Easy — Node service stopped: PASSED
✅ Medium — Configuration drift: PASSED
✅ Hard — Hybrid failure: PASSED
✅ Dynamic reset(difficulty): PASSED
✅ Step penalty works
✅ Repeat penalty works
✅ Logs contain timestamps and red herrings
🎉 ALL TESTS PASSED
```

### 3. Start the Server

```bash
# Via uv (recommended)
uv run server

# Or via uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 4. Test Endpoints

```bash
# Reset (initialize a new episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Step (send an action)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "execute_command", "command": "pm2 status"}}'

# State (get episode metadata)
curl http://localhost:7860/state
```

### 5. Run the Baseline Agent

```bash
export HF_TOKEN="your-huggingface-token"
python inference.py
```

---

## API Reference

### `POST /reset`

Initializes a new episode with the specified difficulty.

**Request Body:**
```json
{"difficulty": "easy"}  // "easy", "medium", or "hard"
```

**Response:**
```json
{
  "observation": {
    "stdout": "Terminal session started. Type commands to investigate.",
    "stderr": "",
    "exit_code": 0,
    "current_directory": "/var/www/mern-app",
    "system_health_check": "HTTP 503 Service Unavailable",
    "done": false,
    "reward": 0.0
  },
  "reward": 0.0,
  "done": false
}
```

### `POST /step`

Executes an action and returns the new observation.

**Request Body:**
```json
{
  "action": {
    "thought": "Checking process status",
    "action_type": "execute_command",
    "command": "pm2 status"
  }
}
```

**Response:**
```json
{
  "observation": {
    "stdout": "USER       PID %CPU %MEM ...\nsreuser   1242  1.2  3.8 ... node server.js",
    "stderr": "",
    "exit_code": 0,
    "current_directory": "/var/www/mern-app",
    "system_health_check": "HTTP 500 Internal Server Error",
    "done": false,
    "reward": 0.18
  },
  "reward": 0.18,
  "done": false
}
```

### `GET /state`

Returns the current episode metadata.

**Response:**
```json
{
  "episode_id": "851ade5c-a169-4dbe-8d00-4db94b4d824d",
  "step_count": 1,
  "difficulty_level": "medium",
  "is_resolved": false
}
```

---

## Baseline Agent

The baseline agent (`inference.py`) uses the **Qwen/Qwen2.5-72B-Instruct** model via the Hugging Face Inference API to solve all 3 tasks.

### How It Works

1. **Multi-task loop**: Iterates through easy → medium → hard
2. **LLM-powered reasoning**: Each step, the model receives the current observation and outputs a structured JSON action
3. **Mandatory logging**: Emits `[START]`, `[STEP]`, and `[END]` tags for automated judging
4. **Error recovery**: Falls back to `check_health` on API failures
5. **Resource-efficient**: 10-step limit per task, sync OpenAI client

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `LOCAL_IMAGE_NAME` | No | `sre-mern-env:latest` | Docker image name |

### Output Format

```
[START] task=easy_node_down env=sre_mern_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"execute_command","command":"pm2 status"} reward=0.18 done=false error=null
[STEP] step=2 action={"action_type":"execute_command","command":"pm2 start all"} reward=1.48 done=true error=null
[END] success=true steps=2 score=1.000 rewards=0.18,1.48

📊 BASELINE EVALUATION REPORT CARD
============================================================
  easy_node_down                 ✅ PASS  Score: 1.0  Steps: 2
  medium_config_drift            ✅ PASS  Score: 1.0  Steps: 4
  hard_hybrid_failure            ✅ PASS  Score: 1.0  Steps: 5

  AGGREGATE SCORE                0.967
============================================================
```

---

## File Structure

```
sre-triage-env/
│
├── server/                     # Server-side environment package
│   ├── __init__.py             # Package init + sys.path configuration
│   ├── app.py                  # FastAPI server with main() entry point
│   └── environment.py          # Core state machine (220 lines)
│
├── models.py                   # Pydantic models (Action, Observation, State)
├── client.py                   # OpenEnv EnvClient with reset(difficulty) override
├── inference.py                # Baseline agent with multi-task evaluation loop
├── test_logic.py               # Comprehensive unit test suite
│
├── Dockerfile                  # Production container (python:3.11-slim, UID 1000)
├── .dockerignore               # Excludes venv, cache, tests from build context
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package metadata + [project.scripts] entry point
├── uv.lock                     # Deterministic dependency lock file
├── openenv.yaml                # OpenEnv manifest with task definitions
└── README.md                   # This file
```

---

## Configuration

### Resource Requirements

| Resource | Limit |
|----------|-------|
| vCPU | 2 |
| Memory | 8 GB |
| Max Runtime | 20 minutes |
| Max Steps per Task | 10 (inference) / 15 (environment timeout) |

### Docker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | Server listen port |
| `WORKERS` | `2` | Uvicorn worker count |

---

## Deployment

### Deploy to Hugging Face Spaces

1. **Create a new Space** on [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Docker**
   - Visibility: Public

2. **Push your code:**
   ```bash
   openenv push --repo-id your-username/sre-triage-env
   ```

3. **Add secrets** in Space Settings → Variables and Secrets:
   - `HF_TOKEN` — your Hugging Face API token

4. **Verify** once the build completes:
   ```bash
   curl -X POST https://your-space.hf.space/reset -H "Content-Type: application/json" -d '{}'
   ```

### Run the Pre-Submission Validator

```bash
./validate-submission.sh https://your-space.hf.space .
```

Expected output:
```
[PASSED] HF Space is live and responds to /reset
[PASSED] Docker build succeeded
[PASSED] openenv validate passed
All 3/3 checks passed! Your submission is ready to submit.
```

---

## OpenEnv Compliance

| Requirement | Status |
|-------------|--------|
| `openenv validate` passes | ✅ `[OK] Ready for multi-mode deployment` |
| Typed Pydantic models | ✅ `SREAction(Action)`, `SREObservation(Observation)`, `SREState(State)` |
| `step()` / `reset()` / `state()` | ✅ All implemented |
| `openenv.yaml` manifest | ✅ With 3 task definitions |
| `pyproject.toml` + `uv.lock` | ✅ For reproducible builds |
| `[project.scripts]` server entry | ✅ `server = "server.app:main"` |
| `from_docker_image()` support | ✅ Via `EnvClient` base class |
| `Dockerfile` at root | ✅ `python:3.11-slim`, UID 1000, port 7860 |

---

## License

MIT

---

*Built for the [OpenEnv Hackathon](https://openenv.dev) — Training AI agents for real-world tasks.*
