"""Data-driven eval of source-filter scope *shapes* across varied prompts and
connectors.

Each scenario fixes the connected sources, a mock corpus, and a query, then
asserts the shape of the searches the agent runs. Routing has two shapes:

- COMBINED ("search X and Y", "look in X, Y and Z") -> all named sources are
  searched TOGETHER in one pass.
- SEQUENTIAL / FALLBACK ("X first, then Y", "X; if nothing, Y") -> sources are
  walked ONE AT A TIME in order. The decision flow returns the first
  un-searched routed source on each call (`already_searched` is threaded back
  in), so each scoped search covers a single source and the agent stops once it
  has the answer.
- single explicitly named source -> scoped to that source.
- no source named -> an unscoped search over everything.

Assertions target the resolved scope per search (robust to LLM variance), not
exact call counts or answer phrasing — see the package README.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic import Field
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.db.models import User
from tests.external_dependency_unit.connector_filter_eval.harness import EvalCase
from tests.external_dependency_unit.connector_filter_eval.harness import EvalResult
from tests.external_dependency_unit.connector_filter_eval.harness import MockDoc
from tests.external_dependency_unit.connector_filter_eval.harness import RecordedSearch
from tests.external_dependency_unit.connector_filter_eval.harness import run_eval_case


class Expect(BaseModel):
    """What the searches for a scenario must look like."""

    # Sources that must end up searched (a None/ALL scope counts as reaching all).
    reached: list[DocumentSource]
    # Sources that must NEVER be searched.
    excluded: list[DocumentSource] = []
    # A distinctive token the grounded answer should echo (case-insensitive).
    proof: str
    # Every search must carry a source filter (no unscoped pass).
    require_scoped: bool = True
    # An unscoped (search-everything) pass is expected.
    allow_unscoped: bool = False
    # The named sources should be searched together in a single pass.
    combined_in_one_pass: bool = False
    # Sequential walk: every scoped search must cover exactly one source (proves
    # the flow narrows per call rather than scoping to the whole routed union).
    single_source_per_pass: bool = False
    # The first search's scope must equal this exactly (the deterministic part of
    # a walk — the first routed source, searched alone).
    first_scope: list[DocumentSource] | None = None


class Scenario(BaseModel):
    name: str
    connected_sources: list[DocumentSource]
    mock_docs: list[MockDoc]
    query: str
    expect: Expect = Field(...)


SCENARIOS: list[Scenario] = [
    # --- shape "1": one explicitly named source ----------------------------
    Scenario(
        name="single-explicit-jira",
        connected_sources=[
            DocumentSource.JIRA,
            DocumentSource.GITHUB,
            DocumentSource.GOOGLE_DRIVE,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.JIRA,
                title="CHECKOUT-512",
                content="Checkout crashes on empty cart; root cause is a null total.",
            ),
            MockDoc(
                source=DocumentSource.GITHUB,
                title="checkout.ts",
                content="Checkout component source.",
            ),
            MockDoc(
                source=DocumentSource.GOOGLE_DRIVE,
                title="Roadmap",
                content="Quarterly roadmap deck.",
            ),
        ],
        query="Find the Jira ticket about the checkout crash.",
        expect=Expect(
            reached=[DocumentSource.JIRA],
            excluded=[DocumentSource.GITHUB, DocumentSource.GOOGLE_DRIVE],
            proof="null total",
        ),
    ),
    # --- shape "2": two named sources, no priority -> combined -------------
    Scenario(
        name="combined-pair-slack-linear",
        connected_sources=[
            DocumentSource.SLACK,
            DocumentSource.LINEAR,
            DocumentSource.CONFLUENCE,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.SLACK,
                title="#eng-incidents",
                content="API rate-limit bug traced to a missing retry-after header.",
            ),
            MockDoc(
                source=DocumentSource.LINEAR,
                title="ENG-77",
                content="Tracking issue for the API rate-limit bug.",
            ),
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Eng Handbook",
                content="General engineering onboarding.",
            ),
        ],
        query=(
            "Check both Slack and Linear for the discussion about the API "
            "rate-limit bug."
        ),
        expect=Expect(
            reached=[DocumentSource.SLACK, DocumentSource.LINEAR],
            excluded=[DocumentSource.CONFLUENCE],
            combined_in_one_pass=True,
            proof="retry-after",
        ),
    ),
    # --- shape "3": three named sources, no priority -> combined -----------
    Scenario(
        name="combined-trio-docs",
        connected_sources=[
            DocumentSource.CONFLUENCE,
            DocumentSource.NOTION,
            DocumentSource.GOOGLE_DRIVE,
            DocumentSource.SLACK,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.GOOGLE_DRIVE,
                title="Q3 Design Doc",
                content="The Q3 design proposes a sharded ledger for payments.",
            ),
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Design index",
                content="Links to design docs.",
            ),
            MockDoc(
                source=DocumentSource.NOTION,
                title="Scratchpad",
                content="Misc design notes.",
            ),
            MockDoc(
                source=DocumentSource.SLACK,
                title="#random",
                content="Friday lunch plans.",
            ),
        ],
        query="Search Confluence, Notion, and Google Drive for the Q3 design doc.",
        expect=Expect(
            reached=[
                DocumentSource.CONFLUENCE,
                DocumentSource.NOTION,
                DocumentSource.GOOGLE_DRIVE,
            ],
            excluded=[DocumentSource.SLACK],
            combined_in_one_pass=True,
            proof="sharded ledger",
        ),
    ),
    # --- shape "1,1,1": ordered walk, answer in the LAST source ------------
    Scenario(
        name="ordered-walk-answer-last",
        connected_sources=[
            DocumentSource.ZENDESK,
            DocumentSource.JIRA,
            DocumentSource.SLACK,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.SLACK,
                title="#support-eng",
                content="Export button no-op is a known bug on v3.0; fixed in v3.1.",
            ),
        ],
        query=(
            "Look in Zendesk first, then Jira, and finally Slack. Stop as soon as "
            "you find the answer. Customers report the export button does nothing — "
            "is there a known fix?"
        ),
        expect=Expect(
            reached=[
                DocumentSource.ZENDESK,
                DocumentSource.JIRA,
                DocumentSource.SLACK,
            ],
            single_source_per_pass=True,
            first_scope=[DocumentSource.ZENDESK],
            proof="3.1",
        ),
    ),
    # --- shape "1" (walk): same ordered prompt, answer in the FIRST source -
    Scenario(
        name="ordered-walk-answer-first",
        connected_sources=[
            DocumentSource.ZENDESK,
            DocumentSource.JIRA,
            DocumentSource.SLACK,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.ZENDESK,
                title="Export help article",
                content="If export does nothing, clear the saved filter and retry.",
            ),
        ],
        query=(
            "Look in Zendesk first, then Jira, and finally Slack. Stop as soon as "
            "you find the answer. Customers report the export button does nothing — "
            "is there a known fix?"
        ),
        # Answer is in the FIRST source, so the walk should stop after Zendesk —
        # Jira and Slack should never be searched. This is the precision win of
        # the walk over a union scope.
        expect=Expect(
            reached=[DocumentSource.ZENDESK],
            excluded=[DocumentSource.JIRA, DocumentSource.SLACK],
            single_source_per_pass=True,
            first_scope=[DocumentSource.ZENDESK],
            proof="saved filter",
        ),
    ),
    # --- shape "1,1": fallback, answer in the SECOND source ----------------
    Scenario(
        name="fallback-walk-notion-then-gitlab",
        connected_sources=[DocumentSource.NOTION, DocumentSource.GITLAB],
        mock_docs=[
            MockDoc(
                source=DocumentSource.GITLAB,
                title="billing-service README",
                content="Failed charges retry with exponential backoff, up to 5 times.",
            ),
        ],
        query=(
            "Check Notion for how the billing service retries failed charges. If "
            "you don't find it there, look in GitLab."
        ),
        # Notion is empty, so the walk falls back to GitLab; each pass is scoped
        # to a single source (never the Notion+GitLab union).
        expect=Expect(
            reached=[DocumentSource.NOTION, DocumentSource.GITLAB],
            single_source_per_pass=True,
            first_scope=[DocumentSource.NOTION],
            proof="exponential backoff",
        ),
    ),
    # --- shape "ALL": no source named -> unscoped --------------------------
    Scenario(
        name="generic-no-source-named",
        connected_sources=[
            DocumentSource.SLACK,
            DocumentSource.JIRA,
            DocumentSource.CONFLUENCE,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="PTO Policy",
                content="Request time off two weeks in advance via the HR portal.",
            ),
        ],
        query="What's the standard process for requesting time off?",
        expect=Expect(
            reached=[DocumentSource.CONFLUENCE],
            require_scoped=False,
            allow_unscoped=True,
            proof="two weeks",
        ),
    ),
    # --- shape "ALL": explicit "search everywhere" -------------------------
    Scenario(
        name="explicit-search-everywhere",
        connected_sources=[
            DocumentSource.SLACK,
            DocumentSource.JIRA,
            DocumentSource.CONFLUENCE,
            DocumentSource.GOOGLE_DRIVE,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.GOOGLE_DRIVE,
                title="Travel Policy",
                content="Company travel policy: book economy class for flights "
                "under six hours.",
            ),
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="HR index",
                content="Links to HR policies.",
            ),
        ],
        # Explicitly asking to look everywhere must NOT scope to any subset.
        query="Search across all of our connected tools for the company travel policy.",
        expect=Expect(
            reached=[DocumentSource.GOOGLE_DRIVE],
            require_scoped=False,
            allow_unscoped=True,
            proof="economy class",
        ),
    ),
    # --- shape "ALL": ops question, no source named ------------------------
    Scenario(
        name="generic-ops-expense-approval",
        connected_sources=[
            DocumentSource.GMAIL,
            DocumentSource.CONFLUENCE,
            DocumentSource.JIRA,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Expense Policy",
                content="Expense reports over $1,000 require finance director "
                "approval.",
            ),
        ],
        query="Who approves expense reports over $1,000?",
        expect=Expect(
            reached=[DocumentSource.CONFLUENCE],
            require_scoped=False,
            allow_unscoped=True,
            proof="finance director",
        ),
    ),
    # --- shape "ALL": an eng-flavored topic still names no source ----------
    Scenario(
        name="topic-not-treated-as-source",
        connected_sources=[
            DocumentSource.CONFLUENCE,
            DocumentSource.SLACK,
            DocumentSource.JIRA,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="On-call Runbook",
                content="On-call rotations run weekly with a Monday handoff.",
            ),
        ],
        # Subject matter (on-call/eng ops) must not be guessed into a source.
        query="How do we handle on-call rotations?",
        expect=Expect(
            reached=[DocumentSource.CONFLUENCE],
            require_scoped=False,
            allow_unscoped=True,
            proof="monday",
        ),
    ),
    # --- shape "ALL": only unconnected sources named -> fall open ----------
    Scenario(
        name="all-named-sources-unconnected",
        connected_sources=[DocumentSource.SLACK, DocumentSource.JIRA],
        mock_docs=[
            MockDoc(
                source=DocumentSource.SLACK,
                title="#launch",
                content="The go/no-go checklist for the launch is pinned here.",
            ),
        ],
        # Notion and Google Drive are not connected, so both are dropped and the
        # search falls open to everything rather than scoping to nothing.
        query="Look in Notion and Google Drive for the launch checklist.",
        expect=Expect(
            reached=[DocumentSource.SLACK],
            require_scoped=False,
            allow_unscoped=True,
            proof="go/no-go",
        ),
    ),
]


def _render_scopes(searches: list[RecordedSearch]) -> list[str]:
    return [s.scope for s in searches]


# Models often render hyphens/dashes as their typographic variants (non-breaking
# hyphen, en/em dash) — fold them to ASCII so a `proof` token still matches.
_DASHES = {"‐", "‑", "‒", "–", "—", "−"}


def _normalize(text: str) -> str:
    out = "".join("-" if ch in _DASHES else ch for ch in text)
    return out.lower()


def _assert_scenario(result: EvalResult, sc: Scenario) -> None:
    print("\n" + result.report())
    assert result.internal_searches, f"{sc.name}: no internal search was performed"

    scopes = [s.source_filter for s in result.internal_searches]
    exp = sc.expect

    if exp.require_scoped:
        assert all(sf is not None for sf in scopes), (
            f"{sc.name}: every search should be scoped, got {_render_scopes(result.internal_searches)}"
        )
    if exp.allow_unscoped:
        assert any(sf is None for sf in scopes), (
            f"{sc.name}: expected an unscoped (search-everything) pass, got "
            f"{_render_scopes(result.internal_searches)}"
        )

    def reached(src: DocumentSource) -> bool:
        return any(sf is None or src in sf for sf in scopes)

    for src in exp.reached:
        assert reached(src), (
            f"{sc.name}: {src.value} was never searched; "
            f"scopes={_render_scopes(result.internal_searches)}"
        )
    for src in exp.excluded:
        assert not reached(src), (
            f"{sc.name}: {src.value} must not be searched; "
            f"scopes={_render_scopes(result.internal_searches)}"
        )

    if exp.combined_in_one_pass:
        want = set(exp.reached)
        assert any(sf is not None and want <= set(sf) for sf in scopes), (
            f"{sc.name}: {[s.value for s in exp.reached]} should be searched "
            f"together in one pass; scopes={_render_scopes(result.internal_searches)}"
        )

    if exp.single_source_per_pass:
        assert all(sf is None or len(sf) == 1 for sf in scopes), (
            f"{sc.name}: a sequential walk must scope each search to a single "
            f"source, not the routed union; scopes="
            f"{_render_scopes(result.internal_searches)}"
        )

    if exp.first_scope is not None:
        assert scopes[0] == exp.first_scope, (
            f"{sc.name}: first search should be scoped to "
            f"{[s.value for s in exp.first_scope]}; got "
            f"{_render_scopes(result.internal_searches)}"
        )

    assert _normalize(exp.proof) in _normalize(result.final_answer), (
        f"{sc.name}: answer should be grounded in the corpus (expected "
        f"{exp.proof!r}), got: {result.final_answer!r}"
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scope_shape(scenario: Scenario, eval_user: User, db_session: Session) -> None:
    case = EvalCase(
        name=scenario.name,
        connected_sources=list(scenario.connected_sources),
        mock_docs=list(scenario.mock_docs),
        query=scenario.query,
    )
    result = run_eval_case(case, db_session, eval_user)
    _assert_scenario(result, scenario)
