from __future__ import annotations

import json
from contextlib import nullcontext
from unittest.mock import MagicMock
from unittest.mock import patch

from onyx.configs.constants import MessageType
from onyx.llm.models import UserMessage
from onyx.secondary_llm_flows.time_filter import _parse_time_decision
from onyx.secondary_llm_flows.time_filter import decide_time_filter
from onyx.secondary_llm_flows.time_filter import TimeFilter
from onyx.tools.models import ChatMinimalTextMessage


def _run_decision(
    history: list[ChatMinimalTextMessage],
    llm_returns: str,
) -> tuple[TimeFilter | None, list]:
    """Run decide_time_filter with the LLM stubbed to return `llm_returns`.
    Returns (time_filter, prompt_messages)."""
    captured: dict = {}

    def fake_invoke(prompt: list, **_kwargs: object) -> MagicMock:
        captured["prompt"] = prompt
        resp = MagicMock()
        resp.choice.message.content = llm_returns
        return resp

    llm = MagicMock()
    llm.invoke.side_effect = fake_invoke
    with (
        patch(
            "onyx.secondary_llm_flows.time_filter.llm_generation_span",
            return_value=nullcontext(MagicMock()),
        ),
        patch("onyx.secondary_llm_flows.time_filter.record_llm_response"),
    ):
        tf = decide_time_filter(history, llm)
    return tf, captured.get("prompt", [])


# ---- _parse_time_decision (pure parsing, no LLM) ----


def test_none_filter_returns_none() -> None:
    assert _parse_time_decision(json.dumps({"filter_type": "none"})) is None


def test_hard_cutoff_sets_only_lower_bound() -> None:
    tf = _parse_time_decision(
        json.dumps(
            {"filter_type": "hard_cutoff", "start_date": "2025-03-01", "end_date": None}
        )
    )
    assert tf is not None
    assert tf.start is not None and tf.start.isoformat() == "2025-03-01T00:00:00+00:00"
    assert tf.end is None


def test_single_day_is_a_full_day_range() -> None:
    tf = _parse_time_decision(
        json.dumps(
            {
                "filter_type": "range",
                "start_date": "2024-03-25",
                "end_date": "2024-03-25",
            }
        )
    )
    assert tf is not None
    assert tf.start is not None and tf.start.isoformat() == "2024-03-25T00:00:00+00:00"
    # End is pushed to the end of the day so a <= comparison includes the whole day.
    assert (
        tf.end is not None and tf.end.isoformat() == "2024-03-25T23:59:59.999999+00:00"
    )


def test_named_month_becomes_full_span_range() -> None:
    tf = _parse_time_decision(
        json.dumps(
            {
                "filter_type": "range",
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            }
        )
    )
    assert tf is not None
    assert tf.start is not None and tf.start.isoformat() == "2025-01-01T00:00:00+00:00"
    assert tf.end is not None and tf.end.date().isoformat() == "2025-01-31"


def test_favor_recent_has_no_bounds() -> None:
    tf = _parse_time_decision(json.dumps({"filter_type": "favor_recent"}))
    assert tf is not None
    assert tf.start is None and tf.end is None and tf.favor_recent is True


def test_malformed_json_returns_none() -> None:
    assert _parse_time_decision("not json at all") is None


def test_empty_content_returns_none() -> None:
    assert _parse_time_decision("") is None
    assert _parse_time_decision(None) is None


def test_filter_type_set_but_no_dates_returns_none() -> None:
    """A hard_cutoff/range with no parseable dates is not a usable filter."""
    tf = _parse_time_decision(
        json.dumps({"filter_type": "range", "start_date": None, "end_date": None})
    )
    assert tf is None


# ---- decide_time_filter (prompt construction + LLM stub) ----


def test_prompt_is_single_user_message_and_excludes_assistant_turns() -> None:
    history = [
        ChatMinimalTextMessage(
            message="What changed last January?", message_type=MessageType.USER
        ),
        ChatMinimalTextMessage(
            message="Let me look into that.", message_type=MessageType.ASSISTANT
        ),
    ]
    tf, prompt = _run_decision(
        history,
        json.dumps(
            {
                "filter_type": "range",
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
            }
        ),
    )
    assert all(isinstance(m, UserMessage) for m in prompt)
    text = prompt[-1].content
    assert "What changed last January?" in text
    assert "Let me look into that." not in text
    assert tf is not None and tf.start is not None and tf.end is not None


def test_no_user_turns_skips_the_llm() -> None:
    llm = MagicMock()
    history = [
        ChatMinimalTextMessage(
            message="assistant only", message_type=MessageType.ASSISTANT
        )
    ]
    assert decide_time_filter(history, llm) is None
    llm.invoke.assert_not_called()


def test_only_the_last_five_user_turns_reach_the_prompt() -> None:
    history = [
        ChatMinimalTextMessage(message=f"msg {i}", message_type=MessageType.USER)
        for i in range(8)
    ]
    _tf, prompt = _run_decision(history, json.dumps({"filter_type": "none"}))
    text = prompt[-1].content
    assert "msg 7" in text and "msg 3" in text
    assert "msg 2" not in text and "msg 0" not in text
