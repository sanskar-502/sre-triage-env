"""Deploy to Hugging Face Spaces using the Python API."""
import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "sanskar502/sre-triage-env"

# Create the Space (Docker SDK)
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )
    print(f" Space created/exists: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f" create_repo: {e}")

# Upload all project files
api = HfApi()

# Files to upload (everything except venv, git, pycache)
EXCLUDE = {"venv", ".venv", ".git", "__pycache__", "outputs", "sre_triage_env.egg-info", ".env"}

files_uploaded = 0
for root, dirs, files in os.walk("."):
    # Skip excluded dirs
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    for fname in files:
        fpath = os.path.join(root, fname)
        # Skip hidden/binary
        if fname.startswith(".") and fname not in [".dockerignore", ".gitignore"]:
            continue
        # Path in repo (strip leading ./)
        repo_path = fpath[2:].replace("\\", "/")
        try:
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="space",
            )
            print(f"  {repo_path}")
            files_uploaded += 1
        except Exception as e:
            print(f"  {repo_path}: {e}")

print(f"\n Uploaded {files_uploaded} files to https://huggingface.co/spaces/{REPO_ID}")
print(" Space will build automatically. Check the URL above in ~2 minutes.")
