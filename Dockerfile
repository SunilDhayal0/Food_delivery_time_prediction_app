FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (build essentials, git, git-lfs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application files
COPY . /app

# Expose port 7860 (Hugging Face default) or 8000
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
