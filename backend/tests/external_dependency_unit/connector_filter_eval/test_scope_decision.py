"""Direct, real-LLM regression tests for `decide_search_scope`.

These call the decision flow in isolation (no agent loop), so they pin the exact
behavior the prompt is responsible for — independent of how the agent decides to
retry. They guard the ways this flow can regress:

1. SEQUENTIAL/BACKOFF must ADVANCE through sources when the queries stay on the
   same line of inquiry, and RE-SEARCH the priority source when the queries shift
   to a new topic.
2. COMBINED must HOLD the full set on every cycle, regardless of query changes.
3. No source named -> unscoped.

LLM behavior varies, so each assertion targets the resolved scope only.
"""

from __future__ import annotations

from onyx.configs.constants import DocumentSource
from onyx.configs.constants import MessageType
from onyx.llm.interfaces import LLM
from onyx.secondary_llm_flows.source_filter import decide_search_scope
from onyx.secondary_llm_flows.source_filter import SearchCycle
from onyx.tools.models import ChatMinimalTextMessage

Z = DocumentSource.ZENDESK
A = DocumentSource.ASANA
G = DocumentSource.GOOGLE_DRIVE


def _user(text: str) -> list[ChatMinimalTextMessage]:
    return [ChatMinimalTextMessage(message=text, message_type=MessageType.USER)]


def _cycle(n: int, queries: list[str], filters: list[str]) -> SearchCycle:
    return SearchCycle(cycle_number=n, queries=queries, searched_sources=filters)


# --- BACKOFF: advance when on-topic, re-search when the topic shifts ---------


def test_backoff_first_cycle_picks_the_first_source(eval_llm: LLM) -> None:
    history = _user(
        "Check Zendesk first. If you don't find anything, check Asana. "
        "Help me resolve this support ticket about a billing error."
    )
    scope = decide_search_scope(history, eval_llm, [Z, A], [], ["billing error ticket"])
    assert scope == [Z], f"first cycle should scope to Zendesk alone, got {scope}"


def test_backoff_advances_when_queries_stay_on_topic(eval_llm: LLM) -> None:
    """Zendesk was searched for this question and came up short; a reworded query
    on the SAME topic should advance to Asana."""
    history = _user(
        "Check Zendesk first. If you don't find anything, check Asana. "
        "Help me resolve this support ticket about a billing error."
    )
    prev = [_cycle(1, ["billing error support ticket"], ["zendesk"])]
    scope = decide_search_scope(
        history, eval_llm, [Z, A], prev, ["customer billing charge dispute"]
    )
    assert scope == [A], f"on-topic repeat should advance to Asana, got {scope}"


def test_backoff_re_searches_same_source_on_very_different_queries(
    eval_llm: LLM,
) -> None:
    """When this cycle's queries are substantially different from the previous
    cycle's (a genuinely new exploration, not a rewording), the previous source
    has not been explored with these queries — re-search it rather than advancing."""
    history = _user(
        "For anything I ask, check Zendesk first, then Asana if nothing turns up."
    )
    prev = [_cycle(1, ["VPN client setup instructions"], ["zendesk"])]
    scope = decide_search_scope(
        history, eval_llm, [Z, A], prev, ["expense reimbursement policy limits"]
    )
    assert scope == [Z], (
        f"very different queries should re-search the previous source, got {scope}"
    )


# --- COMBINED: hold the full set, ignore query changes ----------------------


def test_combined_returns_both_on_first_cycle(eval_llm: LLM) -> None:
    history = _user("Search both Zendesk and Asana for the deploy runbook.")
    scope = decide_search_scope(history, eval_llm, [Z, A], [], ["deploy runbook"])
    assert scope is not None and set(scope) == {Z, A}, (
        f"combined routing should return both sources, got {scope}"
    )


def test_combined_holds_set_even_when_queries_change(eval_llm: LLM) -> None:
    """Combined routing ignores cycle history / query similarity — it re-runs the
    full named set every cycle."""
    history = _user("Search both Zendesk and Asana for whatever I ask.")
    prev = [_cycle(1, ["deploy runbook"], ["zendesk", "asana"])]
    scope = decide_search_scope(
        history, eval_llm, [Z, A], prev, ["unrelated payroll question"]
    )
    assert scope is not None and set(scope) == {Z, A}, (
        f"combined routing should still return both, got {scope}"
    )


# --- NO SCOPE: topic and generic queries must not invent a filter ----------


def test_no_source_named_searches_everything(eval_llm: LLM) -> None:
    history = _user("What's our standard process for requesting time off?")
    scope = decide_search_scope(
        history,
        eval_llm,
        [DocumentSource.CONFLUENCE, DocumentSource.SLACK, DocumentSource.JIRA],
        [],
        ["time off request process"],
    )
    assert scope is None, f"no source named should be unscoped, got {scope}"


def test_topic_is_not_treated_as_a_source(eval_llm: LLM) -> None:
    history = _user("How do we handle on-call rotations?")
    scope = decide_search_scope(
        history,
        eval_llm,
        [DocumentSource.CONFLUENCE, DocumentSource.SLACK, DocumentSource.JIRA],
        [],
        ["on-call rotation handling"],
    )
    assert scope is None, f"topic must not be hallucinated into a filter, got {scope}"
