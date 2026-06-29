import json
from datetime import datetime
from datetime import time
from datetime import timezone

from dateutil.parser import parse
from pydantic import BaseModel

from onyx.configs.constants import MessageType
from onyx.llm.interfaces import LLM
from onyx.llm.models import ChatCompletionMessage
from onyx.llm.models import ReasoningEffort
from onyx.llm.models import UserMessage
from onyx.prompts.filter_extration import TIME_SCOPE_DECISION_PROMPT
from onyx.tools.models import ChatMinimalTextMessage
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span
from onyx.tracing.llm_utils import record_llm_response
from onyx.utils.logger import setup_logger

logger = setup_logger()

# Only the most recent user turns carry time intent; older turns add tokens and
# stale directives. Mirrors MAX_SOURCE_FILTER_USER_TURNS in source_filter.py.
MAX_TIME_FILTER_USER_TURNS = 5


class TimeFilter(BaseModel):
    """A time window detected from the conversation: an inclusive [start, end]
    bound on a document's last-updated date (either side may be None).
    favor_recent is a soft preference for fresh results (reserved — not yet
    wired into ranking)."""

    start: datetime | None = None
    end: datetime | None = None
    favor_recent: bool = False

    def has_bounds(self) -> bool:
        return self.start is not None or self.end is not None


def best_match_time(time_str: str) -> datetime | None:
    preferred_formats = ["%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d"]

    for fmt in preferred_formats:
        try:
            # As we don't know if the user is interacting with the API server from
            # the same timezone as the API server, just assume the queries are UTC time
            # the few hours offset (if any) shouldn't make any significant difference
            dt = datetime.strptime(time_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    # If the above formats don't match, try using dateutil's parser
    try:
        dt = parse(time_str)
        return (
            dt.astimezone(timezone.utc)
            if dt.tzinfo
            else dt.replace(tzinfo=timezone.utc)
        )
    except (ValueError, OverflowError):
        return None


def _parse_time_decision(content: str | None) -> TimeFilter | None:
    """Parse the model's JSON ({"filter_type", "start_date", "end_date"}) into a
    TimeFilter. Returns None on anything unparseable or a "none" decision so the
    caller searches across all time."""
    if not content:
        return None
    try:
        model_json = json.loads(content, strict=False)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Time filter output was not valid JSON: %s", content)
        return None
    if not isinstance(model_json, dict):
        return None

    filter_type = str(model_json.get("filter_type", "")).strip().lower()
    if filter_type in ("none", ""):
        return None
    if filter_type == "favor_recent":
        return TimeFilter(favor_recent=True)

    start_raw = model_json.get("start_date")
    end_raw = model_json.get("end_date")

    start = best_match_time(start_raw) if isinstance(start_raw, str) else None
    # The upper bound is inclusive of the whole named day, so push it to the end
    # of that day before comparing against second-granularity document times.
    end_day = best_match_time(end_raw) if isinstance(end_raw, str) else None
    end = (
        datetime.combine(end_day.date(), time.max, tzinfo=timezone.utc)
        if end_day
        else None
    )

    if start is None and end is None:
        return None
    return TimeFilter(start=start, end=end)


def decide_time_filter(
    history: list[ChatMinimalTextMessage],
    llm: LLM,
) -> TimeFilter | None:
    """Detect, in one LLM call, the time window this turn's internal search should
    be restricted to, from the conversation.

    Returns a TimeFilter, or None to search across all time. Fails open to None on
    any error. The decision is conversation-derived and stable across the repeated
    search cycles within a turn, so the caller computes it once and caches it.
    """
    user_turns = [
        msg.message.strip()
        for msg in history
        if msg.message_type == MessageType.USER and msg.message.strip()
    ]
    if not user_turns:
        return None
    user_turns = user_turns[-MAX_TIME_FILTER_USER_TURNS:]

    last_user_query = user_turns[-1]
    prior_turns = user_turns[:-1]
    conversation_history = (
        "\n".join(prior_turns)
        if prior_turns
        else "N/A, this is the first message in the conversation."
    )
    current_day_time_str = datetime.now(timezone.utc).strftime("%A %B %d, %Y")

    prompt = TIME_SCOPE_DECISION_PROMPT.format(
        current_day_time_str=current_day_time_str,
        conversation_history=conversation_history,
        last_user_query=last_user_query,
    )
    messages: list[ChatCompletionMessage] = [UserMessage(content=prompt)]

    try:
        with llm_generation_span(
            llm=llm,
            flow=LLMFlow.TIME_FILTER_EXTRACTION,
            input_messages=messages,
        ) as span_generation:
            response = llm.invoke(prompt=messages, reasoning_effort=ReasoningEffort.OFF)
            record_llm_response(span_generation, response)
            content = response.choice.message.content
    except Exception:
        logger.exception("Time filter decision failed; searching across all time")
        return None

    return _parse_time_decision(content)
