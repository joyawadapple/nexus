FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the sentence-transformers model weights so the container starts
# without a ~3s cold-start delay on first request.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Run as non-root user
RUN adduser --disabled-password --gecos "" nexus
USER nexus

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
