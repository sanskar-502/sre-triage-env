# app.py
from fastapi import Request
from openenv.core.env_server import create_fastapi_app
from .environment import SREEnvironment
from dotenv import load_dotenv
load_dotenv()

from models import SREAction, SREObservation

# Create the base app with OpenEnv's standard endpoints
env_instance = SREEnvironment()

app = create_fastapi_app(
    env=lambda: env_instance,  # Use shared instance so reset(difficulty) persists
    action_cls=SREAction,
    observation_cls=SREObservation
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "SRE Triage Simulator Running"}

# Override /reset to accept difficulty parameter
@app.post("/reset")
async def reset_with_difficulty(request: Request):
    """Custom reset that accepts difficulty from the JSON body."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    difficulty = body.get("difficulty", None)
    obs = env_instance.reset(difficulty=difficulty)
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False
    }


def main():
    """Entry point for 'uv run server' and direct execution."""
    import uvicorn
    import os
    port = int(os.getenv("PORT", "7860"))
    workers = int(os.getenv("WORKERS", "2"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=workers)


if __name__ == "__main__":
    main()