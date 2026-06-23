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
from tests.external_dependency_unit.connector_filter_eval.harness import MockDoc
from tests.external_dependency_unit.connector_filter_eval.harness import run_eval_case

_PERSONA_NAME = "connector-filter-eval-pinned-source"

# Routing lives entirely in the persona — the user's question names no source.
# The planner reads the persona instruction (injected as a user message) and
# should scope every search to Confluence regardless of the question's topic.
PINNED_SOURCE_PROMPT = (
    "You are the company wiki assistant. Always look for answers ONLY in "
    "Confluence, our internal wiki. Never use any other source."
)

CONNECTED_SOURCES = [
    DocumentSource.CONFLUENCE,
    DocumentSource.GITHUB,
    DocumentSource.SLACK,
]


@pytest.fixture
def pinned_persona_id(
    eval_user: User, db_session: Session
) -> Generator[int, None, None]:
    persona_id = create_eval_persona(
        db_session,
        system_prompt=PINNED_SOURCE_PROMPT,
        name=_PERSONA_NAME,
        user=eval_user,
    )
    try:
        yield persona_id
    finally:
        delete_persona_by_name(_PERSONA_NAME, db_session, is_default=False)


def test_persona_pins_scope_to_single_source(
    pinned_persona_id: int,
    eval_user: User,
    db_session: Session,
) -> None:
    """A persona that pins lookups to one source scopes every search to it, even
    though the user's question names no source at all."""
    case = EvalCase(
        name="persona-pins-confluence",
        persona_id=pinned_persona_id,
        connected_sources=list(CONNECTED_SOURCES),
        mock_docs=[
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="PTO Policy",
                content="Employees accrue 20 days of PTO per year.",
            ),
            MockDoc(
                source=DocumentSource.GITHUB,
                title="README",
                content="Build instructions for the platform.",
            ),
            MockDoc(
                source=DocumentSource.SLACK,
                title="#random",
                content="Lunch plans for Friday.",
            ),
        ],
        query="How much PTO do employees get?",
    )

    result = run_eval_case(case, db_session, eval_user)
    print("\n" + result.report())

    assert result.internal_searches, "no internal search was performed"
    scopes = [s.source_filter for s in result.internal_searches]
    assert all(sf is not None for sf in scopes), (
        f"the persona pins a source, so no search should be unscoped; got {scopes}"
    )
    searched = {source for sf in scopes if sf for source in sf}
    assert searched == {DocumentSource.CONFLUENCE}, (
        f"only the persona's pinned source (Confluence) should be searched, "
        f"got {searched}"
    )
    # "20" is unique to the Confluence doc, proving the answer is grounded there.
    assert "20" in result.final_answer, (
        f"answer should be grounded in Confluence's content, got: "
        f"{result.final_answer!r}"
    )
