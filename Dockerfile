# 1. Use a slim, stable Python base image
FROM python:3.11-slim

# 2. Set environment variables for Python optimization
# Hugging Face requires listening on port 7860
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    WORKERS=2 \
    MAX_CONCURRENT_ENVS=64

# 3. Create a non-root user specifically with UID 1000 for Hugging Face
RUN useradd -m -u 1000 sreuser
# Set up the home directory paths properly for pip installs
ENV HOME=/home/sreuser \
    PATH=/home/sreuser/.local/bin:$PATH

WORKDIR $HOME/app

# 4. Install dependencies FIRST to leverage Docker layer caching
COPY --chown=sreuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application
COPY --chown=sreuser . .

# 6. Switch to the Hugging Face compliant user
USER sreuser

# 7. Expose the Hugging Face port
EXPOSE 7860

# 8. Start Uvicorn dynamically using the scaling variables
CMD uvicorn server.app:app --host 0.0.0.0 --port $PORT --workers $WORKERS