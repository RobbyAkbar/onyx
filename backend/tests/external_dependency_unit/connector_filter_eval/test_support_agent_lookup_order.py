from __future__ import annotations

from collections.abc import Generator

import pytest
from pydantic import BaseModel
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
from tests.external_dependency_unit.connector_filter_eval.harness import run_eval_case

_PERSONA_NAME = "connector-filter-eval-support-agent"

SUPPORT_AGENT_PROMPT = (
    "You help resolve customer support issues. You will be given an issue. "
    "To resolve it you must look for the answer in our connected sources, "
    "preferring them in this order: Zendesk first, then Confluence, and finally "
    "Slack. Use the best answer you find and report it."
)

# Connected sources, in the prescribed lookup order.
LOOKUP_ORDER = [DocumentSource.ZENDESK, DocumentSource.CONFLUENCE, DocumentSource.SLACK]


class _Issue(BaseModel):
    name: str
    query: str
    answer: MockDoc  # the answer lives only in this source
    proof: str  # a distinctive token the grounded answer should echo


SUPPORT_ISSUES = [
    _Issue(
        name="answer-in-zendesk",
        query="A customer says they can't reset their password. How do I help?",
        answer=MockDoc(
            source=DocumentSource.ZENDESK,
            title="Password reset ticket",
            content="Open the customer's ticket and click 'Send reset link'.",
        ),
        proof="reset link",
    ),
    _Issue(
        name="answer-in-confluence",
        query="A customer is asking for a refund. What is our policy and process?",
        answer=MockDoc(
            source=DocumentSource.CONFLUENCE,
            title="Refund policy",
            content="Refunds are issued within 30 days from the billing panel.",
        ),
        proof="30 days",
    ),
    _Issue(
        name="answer-in-slack",
        query="EU customers report a blank screen on the mobile app. Any known fix?",
        answer=MockDoc(
            source=DocumentSource.SLACK,
            title="#support-eng thread",
            content="Known issue on v4.2 for EU users; clear app cache, fix in v4.3.",
        ),
        proof="4.3",
    ),
]


@pytest.fixture
def support_persona_id(
    eval_user: User, db_session: Session
) -> Generator[int, None, None]:
    persona_id = create_eval_persona(
        db_session,
        system_prompt=SUPPORT_AGENT_PROMPT,
        name=_PERSONA_NAME,
        user=eval_user,
    )
    try:
        yield persona_id
    finally:
        delete_persona_by_name(_PERSONA_NAME, db_session, is_default=False)


def _assert_answered_from_source(result: EvalResult, issue: _Issue) -> None:
    """Wherever the answer lives in the lookup order, its source is reached, the
    answer doc is surfaced, and the final answer is grounded in it.

    Mechanism-agnostic: the agent may reach the source via one union search or by
    walking the order one source at a time (scoping each via the `sources` param).
    """
    print("\n" + result.report())
    assert result.internal_searches, "agent performed no internal search"

    answer_source = issue.answer.source
    reached = any(
        s.source_filter is None or answer_source in s.source_filter
        for s in result.internal_searches
    )
    assert reached, (
        f"{answer_source.value} was never in scope; "
        f"scopes={[s.scope for s in result.internal_searches]}"
    )

    returned = {
        source
        for search in result.internal_searches
        for source in search.returned_sources
    }
    assert answer_source in returned, (
        f"{answer_source.value} doc was never surfaced; returned {returned}"
    )
    assert issue.proof.lower() in result.final_answer.lower(), (
        f"answer should be grounded in the {answer_source.value} doc "
        f"(expected {issue.proof!r}), got: {result.final_answer!r}"
    )


@pytest.mark.parametrize("issue", SUPPORT_ISSUES, ids=lambda i: i.name)
def test_support_agent_answers_from_preferred_source(
    issue: _Issue,
    support_persona_id: int,
    eval_user: User,
    db_session: Session,
) -> None:
    case = EvalCase(
        name=issue.name,
        persona_id=support_persona_id,
        connected_sources=list(LOOKUP_ORDER),
        mock_docs=[issue.answer],
        query=issue.query,
        # Let the agent decide when/where to search so we observe real behavior.
        force_first_search=False,
    )
    result = run_eval_case(case, db_session, eval_user)
    _assert_answered_from_source(result, issue)
