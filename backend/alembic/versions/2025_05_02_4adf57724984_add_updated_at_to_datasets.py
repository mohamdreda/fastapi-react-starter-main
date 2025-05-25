"""add_updated_at_to_datasets

Revision ID: 4adf57724984
Revises: 4eac9ed83593
Create Date: 2025-05-02 11:15:07.955673

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4adf57724984'
down_revision: Union[str, None] = '4eac9ed83593'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
