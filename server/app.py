# app.py
from openenv.core.env_server import create_fastapi_app
from .environment import SREEnvironment
from dotenv import load_dotenv
load_dotenv()

from models import SREAction, SREObservation

app = create_fastapi_app(
    env=lambda: SREEnvironment(),  # Default "medium"; difficulty set via reset()
    action_cls=SREAction,
    observation_cls=SREObservation
)


def main():
    """Entry point for 'uv run server' and direct execution."""
    import uvicorn
    import os
    port = int(os.getenv("PORT", "7860"))
    workers = int(os.getenv("WORKERS", "2"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=workers)


if __name__ == "__main__":
    main()