# ============================================================
# Dockerfile — Spam-Ham Email Detector
# ============================================================
# Multi-stage build: keeps the final image lean by separating
# the build (pip install) stage from the runtime stage.

# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps for lxml / BeautifulSoup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy project files
COPY . .

# Ensure data and model directories exist
RUN mkdir -p data/raw data/processed models/distilbert_spam logs

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit app
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
