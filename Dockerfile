FROM python:3.11-slim

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces run the container as user 1000, so the app runs as that user
RUN useradd -m -u 1000 user

WORKDIR /app

# System dependencies required by some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into a fixed cache; the app only loads from here and
# fails at startup if it is missing, never downloads (ruling 4). fastembed 0.9
# cannot pin a revision by name, so the build fails if the model file changes.
ENV EMBED_CACHE_DIR=/opt/models
ARG MODEL_SHA256=51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='$EMBED_CACHE_DIR')" \
    && echo "$MODEL_SHA256  $(ls $EMBED_CACHE_DIR/models--Qdrant--bge-small-en-v1.5-onnx-Q/snapshots/*/model_optimized.onnx)" | sha256sum -c - \
    && chmod -R a+rX $EMBED_CACHE_DIR

# Copy application code, owned by user 1000 so it can write /app/tmp
COPY --chown=user . .
RUN chown user /app

USER user

# Hugging Face Spaces port; Render and others set $PORT
EXPOSE 7860

# Start FastAPI (shell form so ${PORT} expands; exec keeps signals working)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
