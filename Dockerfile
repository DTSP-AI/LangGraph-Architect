# ─── Use a lightweight Python runtime ───────────────────────────────────────────
FROM python:3.11-slim

# ─── Install system dependencies for Postgres and building Python packages ──────
RUN apt-get update && apt-get install -y \
        libpq-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ─── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ─── Copy Alembic migrations directory ─────────────────────────────────────────
# (If your repository already includes alembic/ under the project root, this step may be redundant)
COPY alembic/ ./alembic/

# ─── Expose Streamlit port ─────────────────────────────────────────────────────
EXPOSE 8501

# ─── Environment variables (populated by docker-compose or .env) ────────────────
ENV OPENAI_API_KEY=""
ENV PGVECTOR_CONNECTION_STRING=""
ENV PGVECTOR_COLLECTION_NAME="vector_memory"
ENV VECTOR_DECAY_RATE="0.01"
ENV VECTOR_RETRIEVAL_K="6"
ENV VECTOR_TTL_DAYS="30"
ENV CHAT_HISTORY_DIR="./data/history"
ENV FEEDBACK_LOG_DIR="./data"
ENV MAX_HISTORY_LENGTH="100"

# ─── Entrypoint: run migrations, then launch Streamlit ─────────────────────────
CMD ["sh", "-c", "alembic upgrade head && streamlit run streamlit_ui.py --server.port=8501 --server.address=0.0.0.0"]
