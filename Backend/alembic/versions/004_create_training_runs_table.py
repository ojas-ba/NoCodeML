"""Create training runs table

Revision ID: 004
Revises: 003
Create Date: 2025-10-08 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create training_runs table for run-based training architecture."""
    
    op.create_table(
        'training_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('experiment_id', UUID(as_uuid=True), sa.ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('run_number', sa.Integer, nullable=False),
        sa.Column('job_id', sa.String(255), unique=True, nullable=True, index=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending', index=True),
        sa.Column('config_snapshot', JSONB, nullable=False, server_default='{}'),
        sa.Column('results', JSONB, nullable=True),
        sa.Column('artifacts', JSONB, nullable=True),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), onupdate=sa.func.now()),
    )
    
    # Create unique constraint on experiment_id + run_number
    op.create_unique_constraint(
        'uq_experiment_run_number',
        'training_runs',
        ['experiment_id', 'run_number']
    )
    
    # Create composite index for querying
    op.create_index(
        'idx_training_runs_experiment_created',
        'training_runs',
        ['experiment_id', 'created_at']
    )


def downgrade() -> None:
    """Drop training_runs table."""
    op.drop_index('idx_training_runs_experiment_created', table_name='training_runs')
    op.drop_constraint('uq_experiment_run_number', 'training_runs', type_='unique')
    op.drop_table('training_runs')
