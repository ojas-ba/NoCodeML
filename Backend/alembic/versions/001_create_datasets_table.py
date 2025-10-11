"""Create datasets table

Revision ID: 001
Revises: 
Create Date: 2025-10-03 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the datasets table."""
    op.create_table(
        'datasets',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('storage_path', sa.String(1024), nullable=False, unique=True),
        sa.Column('file_name', sa.String(255), nullable=False),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=False),
        sa.Column('column_count', sa.Integer(), nullable=False),
        sa.Column('column_info', JSONB, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), onupdate=sa.func.now(), nullable=True)
    )
    
    # Create index on id for faster lookups
    op.create_index('ix_datasets_id', 'datasets', ['id'])


def downgrade() -> None:
    """Drop the datasets table."""
    op.drop_index('ix_datasets_id', 'datasets')
    op.drop_table('datasets')
