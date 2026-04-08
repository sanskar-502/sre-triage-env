# server package
import sys
import os

# Ensure the project root is in sys.path so that 'models' can be imported
# when running via 'uv run server' or 'uvicorn server.app:app'
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
