"""add_updated_at_to_datasets

Revision ID: 91f5f4a00852
Revises: 4adf57724984
Create Date: 2025-05-02 11:21:08.847251

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '91f5f4a00852'
down_revision: Union[str, None] = '4adf57724984'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
