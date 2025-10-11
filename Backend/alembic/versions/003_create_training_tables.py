"""Create training tables

Revision ID: 003
Revises: 002
Create Date: 2025-10-06 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create training-related tables and update experiments table."""
    
    # First, add training_status to experiments table
    op.add_column(
        'experiments',
        sa.Column(
            'training_status', 
            sa.Enum('not_started', 'training', 'completed', 'failed', name='trainingstatus'), 
            nullable=False, 
            server_default='not_started'
        )
    )
    op.create_index('ix_experiments_training_status', 'experiments', ['training_status'])
    
    # Create training_jobs table
    op.create_table(
        'training_jobs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('experiment_id', UUID(as_uuid=True), sa.ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('model_type', sa.String(100), nullable=False),
        sa.Column('status', sa.Enum('queued', 'running', 'completed', 'failed', 'cancelled', name='trainingjobstatus'), nullable=False, server_default='queued'),
        sa.Column('config_json', JSONB, nullable=False, server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('celery_task_id', sa.String(255), nullable=True),
    )
    
    # Create indexes for training_jobs
    op.create_index('ix_training_jobs_id', 'training_jobs', ['id'])
    op.create_index('ix_training_jobs_experiment_id', 'training_jobs', ['experiment_id'])
    op.create_index('ix_training_jobs_model_type', 'training_jobs', ['model_type'])
    op.create_index('ix_training_jobs_status', 'training_jobs', ['status'])
    op.create_index('ix_training_jobs_celery_task_id', 'training_jobs', ['celery_task_id'])
    
    # Create training_results table
    op.create_table(
        'training_results',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('job_id', UUID(as_uuid=True), sa.ForeignKey('training_jobs.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('model_path', sa.String(1024), nullable=False),
        sa.Column('metrics_json', JSONB, nullable=False, server_default='{}'),
        sa.Column('feature_importance_json', JSONB, nullable=True),
        sa.Column('confusion_matrix_json', JSONB, nullable=True),
        sa.Column('training_time_seconds', sa.Float(), nullable=False),
        sa.Column('cross_val_scores', JSONB, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    
    # Create indexes for training_results
    op.create_index('ix_training_results_id', 'training_results', ['id'])
    op.create_index('ix_training_results_job_id', 'training_results', ['job_id'])
    
    # Create training_logs table
    op.create_table(
        'training_logs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('job_id', UUID(as_uuid=True), sa.ForeignKey('training_jobs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('epoch', sa.Integer(), nullable=True),
        sa.Column('progress_percent', sa.Float(), nullable=True),
        sa.Column('metrics_json', JSONB, nullable=False, server_default='{}'),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    
    # Create indexes for training_logs
    op.create_index('ix_training_logs_id', 'training_logs', ['id'])
    op.create_index('ix_training_logs_job_id', 'training_logs', ['job_id'])
    op.create_index('ix_training_logs_timestamp', 'training_logs', ['timestamp'])


def downgrade() -> None:
    """Drop training tables and remove training_status from experiments."""
    
    # Drop tables in reverse order of creation (due to foreign keys)
    op.drop_table('training_logs')
    op.drop_table('training_results')
    op.drop_table('training_jobs')
    
    # Remove training_status column from experiments
    op.drop_index('ix_experiments_training_status', 'experiments')
    op.drop_column('experiments', 'training_status')
    
    # Drop the enums
    op.execute('DROP TYPE IF EXISTS trainingstatus')
    op.execute('DROP TYPE IF EXISTS trainingjobstatus')