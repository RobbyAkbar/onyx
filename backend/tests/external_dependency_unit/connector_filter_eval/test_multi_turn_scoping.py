"""Multi-turn source-scoping evals.

Covers flows that single-message scenarios miss: how a source directive in one
turn affects (or doesn't affect) later turns.

- A single named source PERSISTS across a same-topic follow-up (re-querying the
  same source rather than broadening or switching) — the "repeat a single
  source" case.
- A NEW explicit directive in a later turn OVERRIDES the earlier one.
- A STANDING persona pin persists across turns (the desired flip-side of a
  one-off user directive leaking — see test_directive_scoping).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.db.models import User
from onyx.db.persona import delete_persona_by_name
from tests.external_dependency_unit.connector_filter_eval.harness import (
    create_eval_persona,
)
from tests.external_dependency_unit.connector_filter_eval.harness import EvalCase
from tests.external_dependency_unit.connector_filter_eval.harness import EvalResult
from tests.external_dependency_unit.connector_filter_eval.harness import MockDoc
from tests.external_dependency_unit.connector_filter_eval.harness import run_eval_chain

CONNECTED = [
    DocumentSource.GOOGLE_DRIVE,
    DocumentSource.CONFLUENCE,
    DocumentSource.ZENDESK,
]


def _scopes(result: EvalResult) -> list[list[DocumentSource] | None]:
    return result.applied_filters


def test_single_source_persists_across_followup(
    eval_user: User,
    db_session: Session,
) -> None:
    """ "Search Google Drive for X" then a same-topic follow-up: the follow-up
    keeps searching Google Drive (re-query the same source), never broadening to
    everything or switching to another source."""
    case = EvalCase(
        name="single-source-persists",
        connected_sources=list(CONNECTED),
        mock_docs=[
            MockDoc(
                source=DocumentSource.GOOGLE_DRIVE,
                title="SLA Policy",
                content="Our SLA guarantees 99.9% uptime; support responds within 4 hours.",
            ),
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Eng Handbook",
                content="General engineering onboarding.",
            ),
        ],
        query="",
    )

    turn0, turn1 = run_eval_chain(
        case,
        [
            "Search Google Drive for our SLA doc.",
            "Can you find more detail on the response time commitment?",
        ],
        db_session,
        eval_user,
    )
    print("\n" + turn0.report() + "\n" + turn1.report())

    assert turn0.internal_searches and all(
        sf == [DocumentSource.GOOGLE_DRIVE] for sf in _scopes(turn0)
    ), (
        f"turn 0 should scope to Google Drive; got {[s.scope for s in turn0.internal_searches]}"
    )

    assert turn1.internal_searches, "follow-up ran no internal search"
    assert all(sf == [DocumentSource.GOOGLE_DRIVE] for sf in _scopes(turn1)), (
        "the follow-up should stay scoped to Google Drive (repeat the same "
        f"source), got {[s.scope for s in turn1.internal_searches]}"
    )


def test_latest_directive_overrides_earlier(
    eval_user: User,
    db_session: Session,
) -> None:
    """A new explicit source directive in a later turn overrides the earlier one
    — the second turn scopes to the newly named source, not the first."""
    case = EvalCase(
        name="latest-directive-overrides",
        connected_sources=list(CONNECTED),
        mock_docs=[
            MockDoc(
                source=DocumentSource.ZENDESK,
                title="Login bug ticket",
                content="Customers cannot log in after a password reset.",
            ),
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Deploy Runbook",
                content="To recover a bad deploy, run the rollback playbook.",
            ),
        ],
        query="",
    )

    turn0, turn1 = run_eval_chain(
        case,
        [
            "Search Zendesk for the login bug.",
            "Now look in Confluence for the deploy runbook instead.",
        ],
        db_session,
        eval_user,
    )
    print("\n" + turn0.report() + "\n" + turn1.report())

    assert turn0.internal_searches and all(
        sf == [DocumentSource.ZENDESK] for sf in _scopes(turn0)
    ), (
        f"turn 0 should scope to Zendesk; got {[s.scope for s in turn0.internal_searches]}"
    )

    assert turn1.internal_searches, "turn 1 ran no internal search"
    assert all(sf == [DocumentSource.CONFLUENCE] for sf in _scopes(turn1)), (
        "turn 1 names Confluence, so it should override Zendesk and scope to "
        f"Confluence; got {[s.scope for s in turn1.internal_searches]}"
    )


_PERSONA_NAME = "connector-filter-eval-multi-turn-pin"
_PIN_PROMPT = (
    "You are the company wiki assistant. Always look for answers ONLY in "
    "Confluence, our internal wiki. Never use any other source."
)


@pytest.fixture
def pinned_persona_id(
    eval_user: User, db_session: Session
) -> Generator[int, None, None]:
    persona_id = create_eval_persona(
        db_session,
        system_prompt=_PIN_PROMPT,
        name=_PERSONA_NAME,
        user=eval_user,
    )
    try:
        yield persona_id
    finally:
        delete_persona_by_name(_PERSONA_NAME, db_session, is_default=False)


def test_persona_pin_persists_across_turns(
    pinned_persona_id: int,
    eval_user: User,
    db_session: Session,
) -> None:
    """A persona that pins lookups to one source keeps every turn scoped to it,
    across multiple unrelated questions — a STANDING instruction persists (unlike
    a one-off user directive, which must not leak)."""
    case = EvalCase(
        name="persona-pin-persists",
        persona_id=pinned_persona_id,
        connected_sources=[
            DocumentSource.CONFLUENCE,
            DocumentSource.GITHUB,
            DocumentSource.SLACK,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="HR Policies",
                content="Employees accrue 20 days PTO; parental leave is 16 weeks.",
            ),
            MockDoc(
                source=DocumentSource.GITHUB,
                title="README",
                content="Build instructions.",
            ),
        ],
        query="",
    )

    turn0, turn1 = run_eval_chain(
        case,
        [
            "How much PTO do employees get?",
            "And what's the parental leave policy?",
        ],
        db_session,
        eval_user,
    )
    print("\n" + turn0.report() + "\n" + turn1.report())

    for turn, label in ((turn0, "turn 0"), (turn1, "turn 1")):
        assert turn.internal_searches, f"{label} ran no internal search"
        assert all(sf == [DocumentSource.CONFLUENCE] for sf in _scopes(turn)), (
            f"{label} should stay pinned to Confluence (standing persona "
            f"instruction); got {[s.scope for s in turn.internal_searches]}"
        )
