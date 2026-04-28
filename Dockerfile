FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    WORKERS=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 sreuser
ENV HOME=/home/sreuser \
    PATH=/home/sreuser/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=sreuser pyproject.toml requirements.txt README.md ./
COPY --chown=sreuser src ./src
COPY --chown=sreuser server ./server
COPY --chown=sreuser openenv.yaml ./
RUN pip install --upgrade pip && pip install .

USER sreuser

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "sre_triage.api.app:app", "--host", "0.0.0.0", "--port", "7860"]