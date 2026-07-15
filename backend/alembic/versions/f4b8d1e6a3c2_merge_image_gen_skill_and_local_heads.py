"""merge image-generation-skill and local heads

Revision ID: f4b8d1e6a3c2
Revises: d3e5a7c9b1f4, bd38e2a494ff
Create Date: 2026-07-15 00:00:00.000000

Pure merge migration. The latest upstream sync added bd38e2a494ff
(update_image_generation_skill), which became a second alembic head parallel to
our prior merge d3e5a7c9b1f4. This reunites them so `alembic upgrade head`
resolves to a single head. No schema changes.
"""

from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401

# revision identifiers, used by Alembic.
revision = "f4b8d1e6a3c2"
down_revision = ("d3e5a7c9b1f4", "bd38e2a494ff")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
