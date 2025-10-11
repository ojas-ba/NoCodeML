"""create prediction_batches table

Revision ID: 005
Revises: 004
Create Date: 2025-10-11

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create prediction_batches table
    op.create_table(
        'prediction_batches',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('experiment_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('file_path', sa.String(1024), nullable=False),
        sa.Column('total_predictions', sa.Integer, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('prediction_batches')
