"""Eval: a one-off source directive in an earlier turn must not leak into a
later, unrelated turn.

Turn 0 explicitly names a source ("Check google drive ...") -> that search is
scoped to Google Drive. Turn 1 asks an unrelated question that names no source
("Where is the HR guide") -> the search must NOT inherit Google Drive; the HR
guide lives in Confluence and the question routes to no source, so the correct
scope is unscoped (search everything).

This is the failure-mode probe for directive scoping across turns: the planner
sees the whole conversation, and a source named in a prior request can wrongly
carry forward. See the package README and the prompt in filter_extration.py.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.db.models import User
from tests.external_dependency_unit.connector_filter_eval.harness import EvalCase
from tests.external_dependency_unit.connector_filter_eval.harness import MockDoc
from tests.external_dependency_unit.connector_filter_eval.harness import run_eval_chain

CONNECTED_SOURCES = [
    DocumentSource.GOOGLE_DRIVE,
    DocumentSource.CONFLUENCE,
    DocumentSource.SLACK,
]


def test_inline_directive_does_not_leak_to_next_turn(
    eval_user: User,
    db_session: Session,
) -> None:
    case = EvalCase(
        name="directive-scoping-drive-then-hr",
        connected_sources=list(CONNECTED_SOURCES),
        mock_docs=[
            MockDoc(
                source=DocumentSource.GOOGLE_DRIVE,
                title="Cloud Runbook",
                content="Cloud deploy runbook: to recover, run the rollback playbook.",
            ),
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="HR Guide",
                content="HR guide: employees get 16 weeks of parental leave.",
            ),
            MockDoc(
                source=DocumentSource.SLACK,
                title="#random",
                content="Lunch plans for Friday.",
            ),
        ],
        # Ignored by run_eval_chain; the messages list drives the turns.
        query="",
    )

    turn0, turn1 = run_eval_chain(
        case,
        [
            "Check google drive to find the cloud runbook.",
            "Where is the HR guide?",
        ],
        db_session,
        eval_user,
    )
    print("\n" + turn0.report() + "\n" + turn1.report())

    # Turn 0: the explicit directive scopes the search to Google Drive.
    assert turn0.internal_searches, "turn 0 ran no internal search"
    turn0_scopes = {src for sf in turn0.applied_filters if sf for src in sf}
    assert DocumentSource.GOOGLE_DRIVE in turn0_scopes, (
        f"turn 0 explicitly names Google Drive, so it should be scoped there; "
        f"got {[s.scope for s in turn0.internal_searches]}"
    )

    # Turn 1: the prior turn's one-off directive must NOT carry forward. The
    # question names no source, so the first search should be unscoped.
    assert turn1.internal_searches, "turn 1 ran no internal search"
    first_turn1 = turn1.internal_searches[0]
    assert first_turn1.source_filter != [DocumentSource.GOOGLE_DRIVE], (
        "the Google Drive directive from turn 0 leaked into turn 1: its first "
        f"search was pinned to Google Drive (scope={first_turn1.scope})"
    )
    assert first_turn1.source_filter is None, (
        "turn 1 names no source, so its first search should be unscoped "
        f"(search everything); got scope={first_turn1.scope}"
    )

    # End-to-end: the HR guide (Confluence-only) must be reachable on turn 1.
    turn1_reached = {src for s in turn1.internal_searches for src in s.returned_sources}
    assert DocumentSource.CONFLUENCE in turn1_reached, (
        f"turn 1 should reach Confluence where the HR guide lives; "
        f"reached {sorted(s.value for s in turn1_reached)}"
    )


def test_explicit_no_filter_request_overrides_prior_directive(
    eval_user: User,
    db_session: Session,
) -> None:
    """A request that explicitly says NOT to filter must search everywhere, even
    when an earlier turn named a source."""
    case = EvalCase(
        name="directive-scoping-explicit-no-filter",
        connected_sources=list(CONNECTED_SOURCES),
        mock_docs=[
            MockDoc(
                source=DocumentSource.GOOGLE_DRIVE,
                title="Cloud Runbook",
                content="Cloud deploy runbook: to recover, run the rollback playbook.",
            ),
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="HR Guide",
                content="HR guide: employees get 16 weeks of parental leave.",
            ),
            MockDoc(
                source=DocumentSource.SLACK,
                title="#random",
                content="Lunch plans for Friday.",
            ),
        ],
        query="",
    )

    _turn0, turn1 = run_eval_chain(
        case,
        [
            "Check google drive to find the cloud runbook.",
            "Try again to find the HR guide, but don't add any filters.",
        ],
        db_session,
        eval_user,
    )
    print("\n" + turn1.report())

    assert turn1.internal_searches, "turn 1 ran no internal search"
    first_turn1 = turn1.internal_searches[0]
    assert first_turn1.source_filter is None, (
        "the request explicitly asked not to filter, so the search must be "
        f"unscoped; got scope={first_turn1.scope}"
    )
