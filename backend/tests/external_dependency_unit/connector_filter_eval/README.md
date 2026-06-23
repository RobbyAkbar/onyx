# Connector-filter eval harness

Evaluate the LLM-driven connector (source) filtering on internal search: define
which connectors are "set up", define mock documents, ask a query, and inspect
**every internal search that ran and the source filter applied to each**.

The agent loop runs with a **real LLM** (so `plan_source_filters` and the agent's
follow-up searches are real decisions); only the data layer is mocked.

## What gets mocked

| Real thing | Mock | Why |
|---|---|---|
| `fetch_unique_document_sources` | returns `case.connected_sources` | defines which connectors exist; also the per-invocation boundary |
| `search_pipeline` | serves `case.mock_docs` filtered by the active `source_type` | controlled corpus + records each search |
| federated / Slack retrieval | disabled | all mock docs flow through the one `search_pipeline` seam, federated or not |

Everything else is real: query expansion, source matching, relevance, and the
agent's tool-calling loop.

## Define a case

```python
from onyx.configs.constants import DocumentSource
from tests.external_dependency_unit.connector_filter_eval.harness import (
    EvalCase, MockDoc, run_eval_case,
)

case = EvalCase(
    name="explicit-confluence",
    connected_sources=[DocumentSource.CONFLUENCE, DocumentSource.GITHUB, DocumentSource.SLACK],
    mock_docs=[
        MockDoc(source=DocumentSource.CONFLUENCE, title="Deploy Runbook", content="..."),
        MockDoc(source=DocumentSource.GITHUB, title="ci.yml", content="..."),
    ],
    query="Find the deployment runbook in Confluence.",
)

result = run_eval_case(case, db_session, eval_user)
print(result.report())

for search in result.internal_searches:      # one per internal_search tool call, in order
    print(search.invocation_index, search.source_filter, search.returned_doc_ids)
```

`result.applied_filters` is the ordered list of source filters (each a
`list[DocumentSource]`, or `None` for "searched everything") — usually all you
need to assert on:

```python
assert result.applied_filters[0] == [DocumentSource.CONFLUENCE]
```

The "search A, fall back to B" scenario is just a query that says so plus a
corpus where only B has the answer — see
`test_fallback_scopes_to_union_and_answers_from_the_source_with_content`. The
harness records whatever the agent actually did across its searches.

### Choosing the model

`EvalCase.llm_provider` / `llm_model` default to the `EVAL_LLM_PROVIDER` /
`EVAL_LLM_MODEL` env vars; when unset, the tenant's configured default provider
is used. The harness never creates or changes providers. `llm_provider` is the
provider's *configured name* in Onyx (e.g. `"DevEnvPresetOpenAI"` or `"Yup"`),
**not** the slug `"openai"`. Set per case to override:

```python
case = EvalCase(..., llm_provider="Yup", llm_model="claude-haiku-4-5")
```

## Run

This suite is **disabled by default** (it makes real LLM calls). It only runs
when `RUN_CONNECTOR_FILTER_EVAL=1` is set — otherwise every case is skipped.

```bash
RUN_CONNECTOR_FILTER_EVAL=1 \
    EVAL_LLM_PROVIDER="Yup" EVAL_LLM_MODEL="claude-haiku-4-5" \
    python -m dotenv -f .vscode/.env run -- \
    pytest -xvs backend/tests/external_dependency_unit/connector_filter_eval
```

Requires Postgres/Redis up and a working LLM provider configured (the cases also
skip if none exists). Use the cheap tiers for real calls: OpenAI `gpt-5-mini` or
Anthropic `claude-haiku-4-5`.

## Inspecting the flow (trace file)

Every `run_eval_case` appends a human-readable trace to
`backend/log/connector_filter_eval_trace.log`. For each case it records, per
internal search: the **scope selected**, the **queries** run, the **docs
returned**, and the **exact `llm_facing_response` sent back to the main agent**
(including the scope breadcrumb), followed by the final answer. This is the
fastest way to see which filters were chosen, what was searched, and what the
agent received between turns. The file is appended across runs — delete it to
start clean.

## Notes / gotchas

- The harness never creates or mutates LLM providers — it uses the existing
  default (or your `llm_provider`/`llm_model` override). `llm_provider` matches
  the provider's display **name**, not its slug.
- `force_first_search=True` (default) forces only the *first* tool call to be
  internal_search via `forced_tool_id`; follow-up searches stay agent-driven.
  Requires the persona to have the internal_search tool (default persona `0`
  does) and a real connector/file to exist — the harness patches
  `SearchTool.is_available` so the tool is present even with an empty dev DB.
- Assertions should be tolerant — LLM behavior varies between runs. Prefer
  asserting on search *scope* and on a distinctive grounded fact, not exact
  answer phrasing (the model may write "24h" or "24 hours").

## How scoping works (secondary flow decides, agent decides retries)

The main agent passes **no** filter parameter. Scope is owned by a secondary
flow (`decide_search_scope` in `secondary_llm_flows/source_filter.py`), and the
agent drives fallback by choosing to search again:

1. **`decide_search_scope` reads the conversation** (persona/assistant routing +
   the user's ask) plus the source(s) **already searched this turn**, and
   returns the source(s) to scope THIS search to (or `None` for everything). It
   runs on every internal_search call.
2. **The shape of the routing decides the walk:**
   - SEQUENTIAL / FALLBACK ("A first, then B") → the first routed source not yet
     searched. As the agent repeats, the flow advances A → B → … one at a time.
   - COMBINED ("search A and B") → the full named set every call, even on
     repeats (a repeat re-runs the same set with new query terms).
   - none named → `None` (search everything).
   A user/persona UI restriction is the outer bound and is never exceeded.
3. **The tool supplies the walk state.** `SearchTool` is built once per turn and
   accumulates `_searched_scopes` across the turn's calls, passing a snapshot
   into `decide_search_scope` as `already_searched`. The flow itself stays
   stateless; the caller threads the history in.
4. **The response tells the agent what happened.** Every scoped search appends a
   note naming the source(s) it covered and inviting a repeat with new terms, so
   the agent can decide whether a follow-up search is worthwhile.

So fallback works because the flow advances per call (driven by
`already_searched`) and the agent re-searches when a result is weak.
`test_scope_shape_scenarios.py` exercises the agent loop end-to-end (answer-in-
first → one scoped search and an early stop; answer-in-last → a full A→B→C walk,
each pass single-source). `test_scope_decision.py` pins the flow directly: the
sequential-advance and combined-hold behaviors that the agent loop relies on.
