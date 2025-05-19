"""fix metadata naming in vector_memory

Revision ID: 3b404a4d3c61
Revises: None
Create Date: 2025-05-18 18:11:11.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '3b404a4d3c61'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'vector_memory',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('embedding', Vector(1536), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.TIMESTAMP(timezone=True),
            server_default=text('NOW()')
        ),
    )


def downgrade():
    op.drop_table('vector_memory')
