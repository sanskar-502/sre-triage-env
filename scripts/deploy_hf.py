"""Optional utility for uploading the repo to a Hugging Face Space."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


REPO_ID = os.getenv("HF_SPACE_REPO_ID", "")
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", "outputs", ".pytest_cache"}
ALLOWED_DOTFILES = {".dockerignore", ".gitignore"}


def iter_upload_files(root: Path):
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in EXCLUDE_DIRS]
        for filename in files:
            path = Path(current_root, filename)
            if filename.startswith(".") and filename not in ALLOWED_DOTFILES:
                continue
            if path.name == ".env":
                continue
            yield path


def main() -> None:
    if not REPO_ID:
        raise ValueError("Set HF_SPACE_REPO_ID before running this script.")

    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )
    print(f"Space ready: https://huggingface.co/spaces/{REPO_ID}")

    api = HfApi()
    uploaded = 0
    for path in iter_upload_files(Path(".")):
        repo_path = path.as_posix().removeprefix("./")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="space",
        )
        uploaded += 1
        print(f"uploaded {repo_path}")

    print(f"Uploaded {uploaded} files to https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
