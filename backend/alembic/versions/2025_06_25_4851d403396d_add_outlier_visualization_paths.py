"""add_outlier_visualization_paths

Revision ID: 4851d403396d
Revises: 682dc1602c8b
Create Date: 2025-06-25 17:18:16.102824

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4851d403396d'
down_revision: Union[str, None] = '682dc1602c8b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
