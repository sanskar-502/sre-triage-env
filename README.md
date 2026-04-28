---
title: SRE Triage Env
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---
# SRE Triage Environment: Autonomous Agent Benchmark

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)](#)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.7%2B-e92063?logo=pydantic)](#)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-Multi--Model-purple)](#)

A deterministic, production-grade benchmarking environment designed to evaluate autonomous AI agents on realistic Site Reliability Engineering (SRE) and DevOps tasks.

## 🚀 Overview

As the AI industry shifts toward autonomous, tool-using agents, evaluating their ability to execute complex, multi-step actions is critical. Most benchmarks rely on static Q&A or toy examples. 

**`sre-triage-env`** bridges this gap by providing an interactive, stateful environment where AI agents must:
1. Parse system metrics and terminal logs (`stdout`/`stderr`).
2. Hypothesize the root cause of MERN stack failures.
3. Emit strictly typed and validated JSON actions (using Pydantic Structured Outputs) to execute shell commands, alter files, or check health.
4. Restore service uptime within a constrained number of steps.

**Key Features:**
- **Multi-Model Support:** Powered by `LiteLLM`, allowing seamless benchmarking against Claude 3.5 Sonnet, GPT-4o, Llama-3, Gemini, or any local vLLM/Ollama instance.
- **Deterministic State Machine:** Simulated application states guarantee fair, repeatable agent evaluation without spinning up expensive cloud infrastructure.
- **Type-Safe API:** Agents connect via a robust FastAPI backend. All interactions (`Action`, `Observation`, `State`) are strictly governed by Pydantic models.
- **Production-Ready Packaging:** Structured as a modern, installable Python package (`src/sre_triage`), ensuring painless CI/CD and script execution.

---

## 🏗️ Architecture

The project enforces a strict separation of concerns, decoupling the simulation engine from the API server and the LLM inference loop:

```text
src/sre_triage/
├── api/            # FastAPI service exposing the interactive Agent API
├── benchmark/      # Multi-model evaluation loop (LiteLLM + Structured Outputs)
├── core/           # Deterministic environment engine & scenario state state-machine
├── schemas/        # Pydantic models enforcing typed Action/Observation contracts
├── telemetry/      # Structured logging and request metrics
├── client.py       # SDK for external agents to connect to the environment
└── settings.py     # Centralized Pydantic BaseSettings configuration
tests/              # Comprehensive Pytest suite asserting API and engine behavior
scripts/            # Deployment tooling (e.g., Hugging Face Space deployments)
```

---

## 📋 Task Catalog (The Benchmark)

Agents are evaluated against increasingly difficult, realistic failure modes:

| Task ID | Difficulty | Failure Mode | Required Agent Intervention |
| :--- | :--- | :--- | :--- |
| `easy_node_down` | 🟢 Easy | Application process abruptly stopped. | Diagnose process state, start node/PM2. |
| `medium_config_drift` | 🟡 Medium | MongoDB mapped port mismatch. | Inspect logs, fix `.env`, restart services. |
| `hard_hybrid_failure` | 🔴 Hard | Port drift + rogue CPU-hogging process. | Fix `.env`, restart, identify and `kill` rogue PID. |
| `hard_bad_secret` | 🔴 Hard | Invalid JWT cryptographic secret. | Inspect logs, rotate secret in config, restart. |
| `hard_disk_pressure` | 🔴 Hard | Disk exhaustion from unrotated logs. | Identify disk space issue, execute `logrotate` / clear logs. |

---

## 💻 Getting Started

### 1. Installation

Clone the repository and install the package with modern Python tooling:

```bash
python -m pip install --upgrade pip
pip install -e .[dev]
```

### 2. Start the Environment Server

The environment runs as a standalone FastAPI service. If agents make changes, they alter the simulated state on this server.

```bash
# Starts the server on http://localhost:7860
sre-triage-server
```

*(Alternatively, run `python -m uvicorn sre_triage.api.app:app --host 0.0.0.0 --port 7860`)*

### 3. Run the LLM Agent Benchmark

You can evaluate any LLM immediately using the built-in benchmarking tool. Simply export your API key and URL.

```bash
# Example for OpenAI's GPT-4o
export API_KEY="sk-..."
export MODEL_NAME="gpt-4o"
sre-triage-benchmark

# Example for Anthropic Claude 3.5
export API_KEY="sk-ant-..."
export MODEL_NAME="claude-3-5-sonnet-20240620"
sre-triage-benchmark

# Example for Local Ollama Model
export API_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="ollama/llama3"
sre-triage-benchmark
```
*Note: Because the benchmark relies on native Pydantic Structured Outputs via LiteLLM, model responses are guaranteed to be syntactically valid JSON.*

---

## 🤖 For Agent Developers (API Surface)

Building your own custom agent? Connect to the environment using standard HTTP calls or our provided `SREEnvClient`.

**Endpoints:**
- `POST /reset` — Resets the environment. Provide `{"task_id":"hard_bad_secret"}`.
- `POST /step` — Submit an `SREAction` (see schema below) and receive an `SREObservation`.
- `GET /state` — Inspect current environment step count and active scenario details.
- `GET /health` — Check if the MERN stack is returning `200 OK`.

**Action Schema:**
Agents must emit this exact Pydantic structure inside their JSON response:
```json
{
  "thought": "The port in the .env file is wrong. I need to update it.",
  "action_type": "write_file",  // "execute_command" | "write_file" | "check_health"
  "command": null,
  "file_path": "/var/www/mern-app/.env",
  "file_content": "MONGO_PORT=27017\n..."
}
```

---

## 🐳 Deployment

The environment is strictly containerized, making it an excellent platform for cloud-native benchmarking.

```bash
# Build the production image
docker build -t sre-triage-env:latest .

# Run the container
docker run -p 7860:7860 sre-triage-env:latest
```

To deploy to a public URL (like a Hugging Face Space), use the included utility script:
```bash
export HF_SPACE_REPO_ID="your-username/sre-triage-env"
python scripts/deploy_hf.py
```

---

## 📝 Running Tests

To ensure the deterministic core logic behaves exactly as expected, run the comprehensive `pytest` suite:

```bash
pytest tests/ -v
```

---
*Built as a state-of-the-art framework for the next generation of autonomous engineering agents.*
