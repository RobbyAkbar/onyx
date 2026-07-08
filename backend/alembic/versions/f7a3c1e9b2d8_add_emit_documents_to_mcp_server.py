"""add emit_documents to mcp server

Revision ID: f7a3c1e9b2d8
Revises: 20f09b642ed0
Create Date: 2026-07-08 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "f7a3c1e9b2d8"
down_revision = "20f09b642ed0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "mcp_server",
        sa.Column(
            "emit_documents",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("mcp_server", "emit_documents")
