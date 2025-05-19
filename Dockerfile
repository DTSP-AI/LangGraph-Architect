# ─── Use a lightweight Python runtime ───────────────────────────────────────────
FROM python:3.11-slim

# ─── Install system dependencies for Postgres and building Python packages ──────
RUN apt-get update && apt-get install -y \
        libpq-dev \
        postgresql-client \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ─── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ─── Copy Alembic config + migrations ──────────────────────────────────────────
COPY alembic.ini ./
COPY alembic/ ./alembic/

# ─── Expose Streamlit port ─────────────────────────────────────────────────────
EXPOSE 8501

# ─── Entrypoint: wait for Postgres, run migrations, then launch Streamlit ─────
CMD ["sh", "-c", "\
  until pg_isready -h postgres -p 5432; do \
    echo 'Waiting for Postgres...'; \
    sleep 1; \
  done && \
  alembic upgrade head && \
  streamlit run streamlit_ui.py --server.port=8501 --server.address=0.0.0.0 \
"]
