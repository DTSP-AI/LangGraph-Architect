"""
create pgvector extension

Revision ID: a1b2c3d4e5f6
Revises: 
Create Date: 2025-05-19 12:00:00
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Ensure the pgvector extension is available in Postgres
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

def downgrade():
    # Remove the pgvector extension if downgrading
    op.execute("DROP EXTENSION IF EXISTS vector;")
