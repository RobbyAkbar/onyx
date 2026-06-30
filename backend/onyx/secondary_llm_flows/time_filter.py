import re
from datetime import datetime
from datetime import time
from datetime import timezone

from dateutil.parser import parse

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


# An inclusive (start, end) bound on a document's last-updated date, detected
# from the conversation. Either side may be None, meaning that bound is not
# applied; (None, None) means search across all time.
TimeFilter = tuple[datetime | None, datetime | None]

# Matches the model's "(start, end)" output. Each side is captured as a token
# (a date or "None"); neither may contain a comma or parenthesis.
_TIME_FILTER_PAIR_RE = re.compile(r"\(\s*([^(),]+?)\s*,\s*([^(),]+?)\s*\)")


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


def _parse_bound(token: str) -> datetime | None:
    """Parse one side of the model's pair: a "YYYY-MM-DD" date, or None."""
    token = token.strip().strip("'\"")
    if token.lower() in ("none", "null"):
        return None
    return best_match_time(token)


def _parse_time_decision(content: str | None) -> TimeFilter:
    """Parse the model's "(start, end)" output into an inclusive (start, end)
    window. Each side is a "YYYY-MM-DD" date or None. Returns (None, None) on
    anything unparseable so the caller searches across all time."""
    if not content:
        return (None, None)
    # Tolerates code fences / stray text some models wrap the pair in.
    match = _TIME_FILTER_PAIR_RE.search(content)
    if match is None:
        logger.warning("Time filter output was not a (start, end) pair: %s", content)
        return (None, None)

    start = _parse_bound(match.group(1))
    # The upper bound is inclusive of the whole named day, so push it to the end
    # of that day before comparing against second-granularity document times.
    end_day = _parse_bound(match.group(2))
    end = (
        datetime.combine(end_day.date(), time.max, tzinfo=timezone.utc)
        if end_day
        else None
    )

    return (start, end)


def decide_time_filter(
    history: list[ChatMinimalTextMessage],
    llm: LLM,
) -> TimeFilter:
    """Detect, in one LLM call, the time window this turn's internal search should
    be restricted to, from the conversation.

    Returns an inclusive (start, end) window; either side is None to leave that
    bound unset, and (None, None) means search across all time. Fails open to
    (None, None) on any error. The decision is conversation-derived and stable
    across the repeated search cycles within a turn, so the caller computes it
    once and caches it.
    """
    user_turns = [
        msg.message.strip()
        for msg in history
        if msg.message_type == MessageType.USER and msg.message.strip()
    ]
    if not user_turns:
        return (None, None)
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
        return (None, None)

    return _parse_time_decision(content)
