#app.py
from openenv.core.env_server import create_fastapi_app
from .environment import SREEnvironment
from dotenv import load_dotenv  # Add this
load_dotenv()

# Import the strict schemas from your models.py
from models import SREAction, SREObservation

app = create_fastapi_app(
    env=lambda: SREEnvironment(difficulty="medium"),
    action_cls=SREAction,
    observation_cls=SREObservation
)