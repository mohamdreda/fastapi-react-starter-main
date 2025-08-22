"""add_outlier_detection_run_table

Revision ID: f02715a782ef
Revises: 14cb2deda639
Create Date: 2025-05-15 15:45:46.153959

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f02715a782ef'
down_revision: Union[str, None] = '14cb2deda639'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
