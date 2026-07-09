"""merge emit_documents and telemetry secrets heads

Revision ID: b2d4f6a8c0e1
Revises: f7a3c1e9b2d8, f6b0949ea33d
Create Date: 2026-07-09 00:00:00.000000

Pure merge migration: the emit_documents migration (f7a3c1e9b2d8) and an
upstream telemetry-secrets migration (f6b0949ea33d) diverged into two parallel
alembic heads. This reunites them so `alembic upgrade head` resolves to a
single head. No schema changes.
"""

from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401

# revision identifiers, used by Alembic.
revision = "b2d4f6a8c0e1"
down_revision = ("f7a3c1e9b2d8", "f6b0949ea33d")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
