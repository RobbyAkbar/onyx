from __future__ import annotations

from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.db.models import User
from tests.external_dependency_unit.connector_filter_eval.harness import EvalCase
from tests.external_dependency_unit.connector_filter_eval.harness import MockDoc
from tests.external_dependency_unit.connector_filter_eval.harness import run_eval_case


def test_explicit_single_source_is_scoped(eval_user: User, db_session: Session) -> None:
    """Naming a connected source scopes the search to just that source."""
    case = EvalCase(
        name="explicit-confluence",
        connected_sources=[
            DocumentSource.CONFLUENCE,
            DocumentSource.GITHUB,
            DocumentSource.SLACK,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Deployment Runbook",
                content="Step-by-step deploy runbook for the platform.",
            ),
            MockDoc(
                source=DocumentSource.GITHUB,
                title="ci.yml",
                content="GitHub Actions CI workflow.",
            ),
        ],
        query="Find the deployment runbook in Confluence.",
    )

    result = run_eval_case(case, db_session, eval_user)
    print("\n" + result.report())

    assert result.internal_searches, "no internal search was performed"
    assert result.internal_searches[0].source_filter == [DocumentSource.CONFLUENCE]


def test_generic_query_searches_everything(
    eval_user: User, db_session: Session
) -> None:
    """A query that names no source is not scoped (searches all connectors)."""
    case = EvalCase(
        name="no-scope",
        connected_sources=[DocumentSource.CONFLUENCE, DocumentSource.GITHUB],
        mock_docs=[
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Onboarding",
                content="How to onboard a new engineer.",
            ),
        ],
        query="How do we onboard a new engineer?",
    )

    result = run_eval_case(case, db_session, eval_user)
    print("\n" + result.report())

    assert result.internal_searches, "no internal search was performed"
    assert result.internal_searches[0].source_filter is None


def test_filter_includes_two_of_three_sources(
    eval_user: User, db_session: Session
) -> None:
    """With 3 connectors hooked up, naming 2 of them scopes to exactly those 2."""
    case = EvalCase(
        name="two-of-three",
        connected_sources=[
            DocumentSource.CONFLUENCE,
            DocumentSource.GITHUB,
            DocumentSource.SLACK,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Deployment Runbook",
                content="Deployment steps for the auth-service.",
            ),
            MockDoc(
                source=DocumentSource.GITHUB,
                title="auth-service README",
                content="auth-service build and deploy instructions.",
            ),
            MockDoc(
                source=DocumentSource.SLACK,
                title="#random",
                content="Lunch plans for Friday.",
            ),
        ],
        query="Check Confluence and GitHub for the auth-service deployment steps.",
    )

    result = run_eval_case(case, db_session, eval_user)
    print("\n" + result.report())

    assert result.internal_searches, "no internal search was performed"
    # The agent may search the two together (one union scope) or one at a time;
    # either way both named sources are reached and the unnamed third (Slack) is
    # never searched. Assert that intent, tolerant of benign per-call variance.
    scopes = [s.source_filter for s in result.internal_searches]
    assert all(sf is not None for sf in scopes), (
        f"a search was left unscoped (would include Slack): {scopes}"
    )
    searched = {source for sf in scopes if sf for source in sf}
    assert {DocumentSource.CONFLUENCE, DocumentSource.GITHUB} <= searched, (
        f"both Confluence and GitHub should be searched, got {searched}"
    )
    assert DocumentSource.SLACK not in searched, (
        f"Slack was not named and should not be searched, got {searched}"
    )


def test_fallback_answers_from_the_source_with_content(
    eval_user: User, db_session: Session
) -> None:
    """A 'look in A, else B' instruction reaches B (the only source with the
    answer) and grounds the answer there.

    The agent may do this as a sequential A→B walk (scoping each search via the
    `sources` param) or as a single union search — this asserts the outcome, not
    the mechanism.
    """
    case = EvalCase(
        name="confluence-then-github",
        connected_sources=[DocumentSource.CONFLUENCE, DocumentSource.GITHUB],
        mock_docs=[
            # Only GitHub has the answer. Confluence contributes nothing, so a
            # union search still surfaces the GitHub doc for the agent to use.
            MockDoc(
                source=DocumentSource.GITHUB,
                title="auth-service README",
                content="The auth-service rotates JWT signing keys every 24h.",
            ),
        ],
        query=(
            "Look in Confluence for how the auth-service rotates its signing "
            "keys. If you find nothing there, check GitHub."
        ),
    )

    result = run_eval_case(case, db_session, eval_user)
    print("\n" + result.report())

    assert result.internal_searches, "no internal search was performed"
    searched = {
        source
        for search in result.internal_searches
        for source in (search.source_filter or [])
    }
    assert DocumentSource.GITHUB in searched, (
        f"GitHub (the source with the answer) was never searched, got {searched}"
    )

    returned = {
        source
        for search in result.internal_searches
        for source in search.returned_sources
    }
    assert DocumentSource.GITHUB in returned, "GitHub's answer doc was never surfaced"
    # "24" is unique to the GitHub doc (not the query), so it proves grounding
    # whether the model writes "24h" or "24 hours".
    assert "24" in result.final_answer, (
        f"answer should be grounded in GitHub's content, got: {result.final_answer!r}"
    )


def test_unconnected_named_source_is_ignored(
    eval_user: User, db_session: Session
) -> None:
    """Naming a source that isn't connected fails open to an unscoped search.

    Notion isn't in `connected_sources`, so the plan validation drops it and the
    search covers everything rather than scoping to (or erroring on) a source
    that can't be searched.
    """
    case = EvalCase(
        name="unconnected-notion",
        connected_sources=[DocumentSource.CONFLUENCE, DocumentSource.GITHUB],
        mock_docs=[
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Incident Postmortem",
                content="Postmortem for the 2026 outage.",
            ),
        ],
        query="Find the incident postmortem in Notion.",
    )

    result = run_eval_case(case, db_session, eval_user)
    print("\n" + result.report())

    assert result.internal_searches, "no internal search was performed"
    scopes = [s.source_filter for s in result.internal_searches]
    assert all(sf is None for sf in scopes), (
        f"an unconnected source must not scope the search; got {scopes}"
    )


def test_mixed_named_sources_scope_to_connected_only(
    eval_user: User, db_session: Session
) -> None:
    """When the query names a connected and an unconnected source, only the
    connected one scopes the search; the unconnected name is dropped."""
    case = EvalCase(
        name="notion-and-github",
        connected_sources=[
            DocumentSource.CONFLUENCE,
            DocumentSource.GITHUB,
            DocumentSource.SLACK,
        ],
        mock_docs=[
            MockDoc(
                source=DocumentSource.GITHUB,
                title="deploy.md",
                content="GitHub deploy guide for the platform.",
            ),
            MockDoc(
                source=DocumentSource.CONFLUENCE,
                title="Onboarding",
                content="How to onboard a new engineer.",
            ),
            MockDoc(
                source=DocumentSource.SLACK,
                title="#random",
                content="Lunch plans for Friday.",
            ),
        ],
        # Notion is not connected; GitHub is the only valid named source.
        query="Check Notion and GitHub for the deploy guide.",
    )

    result = run_eval_case(case, db_session, eval_user)
    print("\n" + result.report())

    assert result.internal_searches, "no internal search was performed"
    scopes = [s.source_filter for s in result.internal_searches]
    assert all(sf is not None for sf in scopes), (
        f"a search was left unscoped despite a valid named source; got {scopes}"
    )
    searched = {source for sf in scopes if sf for source in sf}
    assert searched == {DocumentSource.GITHUB}, (
        f"only the connected named source (GitHub) should be searched, got {searched}"
    )
