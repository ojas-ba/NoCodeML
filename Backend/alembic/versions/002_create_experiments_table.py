"""Create experiments table

Revision ID: 002
Revises: 001
Create Date: 2025-10-03 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the experiments table."""
    op.create_table(
        'experiments',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('dataset_id', UUID(as_uuid=True), sa.ForeignKey('datasets.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('status', sa.Enum('in_progress', 'completed', name='experimentstatus'), nullable=False, server_default='in_progress'),
        sa.Column('config', JSONB, nullable=False, server_default='{}'),
        sa.Column('results', JSONB, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), onupdate=sa.func.now(), nullable=True)
    )
    
    # Create index on id for faster lookups
    op.create_index('ix_experiments_id', 'experiments', ['id'])
    
    # Create index on user_id for fast user queries
    op.create_index('ix_experiments_user_id', 'experiments', ['user_id'])
    
    # Create index on dataset_id for dependency checks
    op.create_index('ix_experiments_dataset_id', 'experiments', ['dataset_id'])
    
    # Create unique constraint on (user_id, name) to enforce unique names per user
    op.create_unique_constraint('uq_user_experiment_name', 'experiments', ['user_id', 'name'])


def downgrade() -> None:
    """Drop the experiments table."""
    op.drop_constraint('uq_user_experiment_name', 'experiments', type_='unique')
    op.drop_index('ix_experiments_dataset_id', 'experiments')
    op.drop_index('ix_experiments_user_id', 'experiments')
    op.drop_index('ix_experiments_id', 'experiments')
    op.drop_table('experiments')
    
    # Drop the enum type
    op.execute('DROP TYPE experimentstatus')
