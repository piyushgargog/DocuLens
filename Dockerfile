# Minimal image for running the FastAPI backend + static frontend.
# 3.12-slim is used deliberately over a newer Python: it's a well-established
# base with broad availability, safely inside this project's 3.10+ support
# range (see README) -- not chasing the newest interpreter for its own sake.
FROM python:3.12-slim

WORKDIR /app

# tesseract is the OCR engine used for scanned / image-only PDFs (pytesseract
# is just a wrapper). --no-install-recommends + the English data only keeps
# this to ~a few tens of MB; the apt lists are removed in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# The base image ships setuptools 78.1.0, which has published advisories
# (PYSEC-2025-49, PYSEC-2026-3447) -- found by the pip-audit CI step. Nothing
# here uses it at runtime, but it shouldn't sit in the image unpatched.
RUN pip install --no-cache-dir --upgrade pip "setuptools>=83"

# Install the CPU-only PyTorch build first, from PyTorch's own CPU wheel
# index. This app never uses a GPU, but PyPI's default "torch" wheel on
# Linux bundles several hundred MB of NVIDIA CUDA packages regardless --
# installing the CPU build here first means sentence-transformers (pulled
# in by requirements.txt below) finds torch already satisfied and never
# reaches for the CUDA-bundled variant.
RUN pip install --no-cache-dir "torch>=2.13.0" --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user, not root: defence in depth, so a hypothetical
# code-execution bug in a dependency isn't already root inside the container.
# The MiniLM model is fetched on first use, so its cache dir must be writable
# by that user (HF_HOME); /tmp, used by Starlette to spool large uploads, is
# world-writable already.
ENV HF_HOME=/app/.cache/huggingface XDG_CACHE_HOME=/app/.cache
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER 10001

EXPOSE 8000

# Docker marks the container unhealthy if the app stops answering (no LLM call).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
