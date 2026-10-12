# OpenPronounce web app (FastAPI + UI). CPU only.
#
# Build:
#   podman build --build-arg HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2) \
#                -t openpronounce .
#
# Save image to /mnt/data/docker-images/:
#   mkdir -p /mnt/data/docker-images
#   podman save openpronounce | gzip > /mnt/data/docker-images/openpronounce.tar.gz
#
# Run:
#   podman run -p 8000:8000 openpronounce
#
# Models are downloaded at build time (with HF_TOKEN for faster speeds).
FROM python:3.12-slim

ARG HF_TOKEN

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    PORT=8000 \
    HF_TOKEN=${HF_TOKEN}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY openpronounce ./openpronounce
RUN pip install ".[app]"

RUN python -c "from openpronounce.speech import _load_models; _load_models()" \
    && python -c "from openpronounce.phones import _load_model; _load_model()"

COPY server.py ./
COPY templates ./templates
COPY static ./static

EXPOSE 8000
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
