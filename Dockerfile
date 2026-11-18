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

# Copy application code, owned by user 1000 so it can write /app/tmp
COPY --chown=user . .
RUN chown user /app

USER user

# Hugging Face Spaces port; Render and others set $PORT
EXPOSE 7860

# Start FastAPI (shell form so ${PORT} expands; exec keeps signals working)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
