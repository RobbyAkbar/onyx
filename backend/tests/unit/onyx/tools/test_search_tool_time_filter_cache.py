from __future__ import annotations

from datetime import datetime
from datetime import timezone
from unittest.mock import MagicMock
from unittest.mock import patch

from onyx.secondary_llm_flows.time_filter import decide_time_filter
from onyx.secondary_llm_flows.time_filter import TimeFilter
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def _make_tool() -> SearchTool:
    """A SearchTool with only the state _expand_queries_and_decide_scope reads,
    avoiding the heavy __init__ (DB, emitter, etc.)."""
    tool = SearchTool.__new__(SearchTool)
    tool.llm = MagicMock()
    tool._scope_decision_settled = True  # skip the source-scope job
    tool._cached_expansion = None
    tool._time_filter = None
    tool._time_filter_computed = False
    return tool


def test_time_filter_runs_once_and_caches_for_the_turn() -> None:
    tool = _make_tool()
    decided = TimeFilter(start=datetime(2025, 1, 1, tzinfo=timezone.utc), end=None)

    scheduled: list[list] = []

    def fake_parallel(jobs: list) -> list:
        funcs = [job[0] for job in jobs]
        scheduled.append(funcs)
        return [decided if func is decide_time_filter else None for func in funcs]

    with patch(
        "onyx.tools.tool_implementations.search.search_tool."
        "run_functions_tuples_in_parallel",
        side_effect=fake_parallel,
    ):
        first = tool._expand_queries_and_decide_scope(
            skip_query_expansion=True,
            message_history=[],
            user_info=None,
            memories=[],
            decide_args=(),
        )
        second = tool._expand_queries_and_decide_scope(
            skip_query_expansion=True,
            message_history=[],
            user_info=None,
            memories=[],
            decide_args=(),
        )

    # The first cycle schedules the time-filter job; the second schedules nothing
    # (expansion skipped, scope settled, time cached), so the parallel runner is
    # only ever invoked once and never re-runs the time decision.
    assert len(scheduled) == 1
    assert decide_time_filter in scheduled[0]

    # Both cycles surface the same cached decision.
    assert first.time_filter is decided
    assert second.time_filter is decided
    assert tool._time_filter_computed is True
