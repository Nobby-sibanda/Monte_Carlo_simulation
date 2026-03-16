# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System libs required by matplotlib (freetype, libpng)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfreetype6-dev \
        libpng-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Minimal runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfreetype6 \
        libpng16-16 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY monte_carlo_options.py .
COPY app.py .

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
