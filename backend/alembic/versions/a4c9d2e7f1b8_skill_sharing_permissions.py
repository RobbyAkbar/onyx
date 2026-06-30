"""skill sharing permissions

Revision ID: a4c9d2e7f1b8
Revises: 99c855a8f2a1
Create Date: 2026-06-30

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "a4c9d2e7f1b8"
down_revision = "99c855a8f2a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "skill",
        sa.Column(
            "public_permission",
            sa.String(),
            nullable=False,
            server_default="VIEWER",
        ),
    )
    op.add_column(
        "skill__user_group",
        sa.Column("permission", sa.String(), nullable=False, server_default="VIEWER"),
    )
    op.create_table(
        "skill__user",
        sa.Column("skill_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("permission", sa.String(), nullable=False, server_default="VIEWER"),
        sa.ForeignKeyConstraint(
            ["skill_id"],
            ["skill.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("skill_id", "user_id"),
    )


def downgrade() -> None:
    op.drop_table("skill__user")
    op.drop_column("skill__user_group", "permission")
    op.drop_column("skill", "public_permission")
