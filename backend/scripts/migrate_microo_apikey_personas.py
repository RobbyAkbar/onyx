"""One-off migration: re-own Microo API-key personas to real, email-keyed users.

Background
----------
Before the SSO bridge, every Microo user acted as a synthetic Onyx API-key user
named ``API_KEY__microo_<logto_sub>`` (email ``api_key__microo_<sub>@<uuid>onyxapikey.ai``).
Personas/chats they created are owned by that API-key user instead of the real,
email-keyed Onyx user that SSO now provisions. This script transfers ownership
from each API-key user to the matching real user.

Identity mapping
----------------
The real email cannot be derived from Onyx alone — it lives in Microo (Logto).
Supply a JSON file mapping ``{ "<logto_sub>": "<email>" }``. Export it from
Microo's user directory. The script extracts ``<sub>`` from each API-key user's
email and looks up the target email in the mapping.

Usage
-----
    # dry run (default) — prints what WOULD change, commits nothing
    python -m scripts.migrate_microo_apikey_personas --mapping mapping.json

    # apply, also move chat history, and deactivate the old API-key users
    python -m scripts.migrate_microo_apikey_personas \
        --mapping mapping.json --include-chats --deactivate --apply

The target real user must already exist in Onyx (i.e. has logged in via SSO at
least once, or was provisioned by /admin/sso/exchange). Missing targets are
reported and skipped.
"""

import argparse
import json
import sys

from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import update

from onyx.db.api_key import is_api_key_email_address
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import ChatSession
from onyx.db.models import Persona
from onyx.db.models import Persona__User
from onyx.db.models import User

# API-key user local-part looks like ``api_key__microo_<sub>``.
SUB_MARKER = "microo_"


def _extract_sub(email: str) -> str | None:
    """Return the Logto sub embedded in an API-key user's email, or None."""
    local = email.split("@")[0].lower()
    idx = local.find(SUB_MARKER)
    if idx == -1:
        return None
    sub = local[idx + len(SUB_MARKER) :].strip()
    return sub or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        required=True,
        help="Path to JSON file mapping { logto_sub: email }.",
    )
    parser.add_argument(
        "--include-chats",
        action="store_true",
        help="Also re-own the API-key user's chat sessions.",
    )
    parser.add_argument(
        "--deactivate",
        action="store_true",
        help="Set is_active=False on the API-key users after a successful transfer.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit changes. Without it the script only reports (dry run).",
    )
    args = parser.parse_args()

    with open(args.mapping, encoding="utf-8") as f:
        raw_mapping = json.load(f)
    # Normalise: lowercased sub -> lowercased email.
    sub_to_email = {str(k).lower(): str(v).lower() for k, v in raw_mapping.items()}

    transferred = 0
    skipped = 0

    with get_session_with_current_tenant() as db_session:
        api_key_users = [
            u
            for u in db_session.scalars(select(User)).all()
            if is_api_key_email_address(u.email) and SUB_MARKER in u.email.lower()
        ]
        print(f"Found {len(api_key_users)} Microo API-key user(s).")

        for src in api_key_users:
            sub = _extract_sub(src.email)
            target_email = sub_to_email.get(sub) if sub else None
            if not target_email:
                print(f"  SKIP {src.email}: no mapping for sub={sub!r}")
                skipped += 1
                continue

            target = db_session.scalar(
                select(User).where(func.lower(User.email) == target_email)
            )
            if target is None:
                print(
                    f"  SKIP {src.email}: target user {target_email!r} not found "
                    "(must log in via SSO first)"
                )
                skipped += 1
                continue
            if target.id == src.id:
                continue

            persona_count = db_session.scalar(
                select(func.count())
                .select_from(Persona)
                .where(Persona.user_id == src.id)
            )
            chat_count = db_session.scalar(
                select(func.count())
                .select_from(ChatSession)
                .where(ChatSession.user_id == src.id)
            )
            print(
                f"  {src.email} -> {target_email}: "
                f"{persona_count} persona(s)"
                + (f", {chat_count} chat(s)" if args.include_chats else "")
                + (" [DRY RUN]" if not args.apply else "")
            )

            if not args.apply:
                transferred += 1
                continue

            # Re-own personas.
            db_session.execute(
                update(Persona)
                .where(Persona.user_id == src.id)
                .values(user_id=target.id)
            )
            # Drop the API-key user's share rows (ownership moved above; merging
            # into existing target shares isn't needed — the owner sees all).
            # ponytail: delete-not-merge; revisit only if explicit per-share
            # permissions on these personas must be preserved for the old user.
            db_session.query(Persona__User).filter(
                Persona__User.user_id == src.id
            ).delete(synchronize_session=False)

            if args.include_chats:
                db_session.execute(
                    update(ChatSession)
                    .where(ChatSession.user_id == src.id)
                    .values(user_id=target.id)
                )

            if args.deactivate:
                src.is_active = False

            transferred += 1

        if args.apply:
            db_session.commit()
            print("Committed.")
        else:
            print("Dry run — nothing committed. Re-run with --apply to commit.")

    print(f"Done. transferred={transferred} skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
