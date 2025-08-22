"""add_outlier_distribution_path_column

Revision ID: 9855d9bab0fb
Revises: 4851d403396d
Create Date: 2025-07-10 02:58:14.785784

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9855d9bab0fb'
down_revision: Union[str, None] = '4851d403396d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
