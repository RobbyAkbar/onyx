"""merge doc_created_at and emit_documents heads

Revision ID: d3e5a7c9b1f4
Revises: b2d4f6a8c0e1, c7d1f0a4b8e2
Create Date: 2026-07-11 00:00:00.000000

Pure merge migration. An upstream migration (c7d1f0a4b8e2, add doc_created_at)
branched off f6b0949ea33d — the same revision our earlier merge (b2d4f6a8c0e1)
already unified — creating a second parallel alembic head. This reunites them so
`alembic upgrade head` resolves to a single head. No schema changes.
"""

from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401

# revision identifiers, used by Alembic.
revision = "d3e5a7c9b1f4"
down_revision = ("b2d4f6a8c0e1", "c7d1f0a4b8e2")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
