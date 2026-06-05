FROM python:3.11-slim

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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

# Copy application code
COPY . .

# Hugging Face Spaces port
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]