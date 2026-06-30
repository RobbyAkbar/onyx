# The following prompts are used for extracting filters to apply along with the query in the
# document index. For example, a filter for dates or a filter by source type such as GitHub
# or Slack
SOURCES_KEY = "sources"

# Used in time_filter.py: detect the time an internal search should be restricted
# to and turn it into an explicit (start, end) pair of ISO dates. The model is
# given today's date and does the relative-date math itself, so ranges and named
# times fall out naturally.
# Filled with: {current_day_time_str}, {conversation_history}, {last_user_query}.
TIME_SCOPE_DECISION_PROMPT = """
You detect the time filter an internal search should be restricted to, from the user's \
conversation. When the conversation EXPLICITLY refers to a time, set a time filter on each \
document's last-updated date; when it refers to no time, set no time filter — (None, None), \
searching across all time. You filter only by time — other scoping is handled by other \
systems.

A time filter is a pair (start, end): start is an inclusive lower bound and end an inclusive \
upper bound on the last-updated date. Either bound is a date or None (no bound on that side). \
Today is {current_day_time_str}; resolve every time the user refers to against today, into \
concrete dates yourself.

## Guidance

Set a time filter only when the user EXPLICITLY refers to a time — "last week", "since \
March", "in January", "between Q1 and Q2", "on the 25th of March", "documents from 2022". \
NEVER infer a time from the topic alone. When the user refers to no time, set (None, None).

When the user DOES refer to a time, the phrasing decides the bounds:

- LOWER BOUND ONLY — an open-ended time toward the present ("since March", "in the last 3 \
months", "after 2023"). Set start; leave end None.

- UPPER BOUND ONLY — an open-ended time toward the past ("before 2023", "older than \
January"). Set end; leave start None.

- BOTH BOUNDS — a named or bounded time. Set start and end to its first and last day. \
"last January" → start its January 1st, end its January 31st. "Q1 2025" → start 2025-01-01, \
end 2025-03-31. "last quarter", "in 2022", "between March and June" all behave this way. A \
single day ("on the 25th of March", "March 25 2024") sets start and end to the same day.

- NO BOUND — a vague preference for fresh results with no time ("the latest", "most \
recent"). Set both start and end to None.

## Conversation history

{conversation_history}

## Guidance reminder

Set a time filter only when the user EXPLICITLY refers to a time; NEVER infer a time from \
the topic alone, and set (None, None) when the user refers to no time. An open-ended time \
sets one bound and leaves the other None. A named or bounded time sets both bounds. A vague \
preference for fresh results sets (None, None).

## Output format

Output ONLY the time filter as a pair: (start, end)
- start is the lower bound and end is the upper bound, each a date as "YYYY-MM-DD" or None.
- Both bounds are inclusive; None means no bound on that side.

Examples:
- "since March 2025" → (2025-03-01, None)
- "before 2023" → (None, 2022-12-31)
- "in January 2025" → (2025-01-01, 2025-01-31)
- "on March 25, 2024" → (2024-03-25, 2024-03-25)
- "the latest billing docs" → (None, None)
- "how do I configure SSO?" → (None, None)

Do not include any formatting, explanations, or other text aside from the pair.

## Query reminder

The user's latest message is:
{last_user_query}

CRITICAL: output only the (start, end) pair.
""".strip()


# Used in source_filter.py: decide which connected source(s) an internal search
# cycle should cover, given the conversation, the prior cycles this turn, and the
# queries being run this cycle. Filled with: {conversation_history},
# {current_cycle_queries}, {previous_cycles}, {valid_sources}, {last_user_query}.
# Output is a bracketed comma-separated list of sources.
SOURCE_SCOPE_DECISION_PROMPT = """
You scope an internal search to its relevant sources. When the conversation EXPLICITLY \
names source(s) to search, scope to them; when it names none, return [] (search every \
source). You scope only by source — other scoping is handled by other systems. The system \
runs multiple cycles, and the queries and sources of previous cycles are provided as \
context.

## Guidance

Scope to a source when it is EXPLICITLY named — in this cycle's queries, or in an earlier \
turn that this cycle continues. NEVER infer a source from the query's topic (e.g. an HR or \
billing query is not a source). If no source is named, return [].

A source named in an earlier turn still applies to a same-topic follow-up that names no new \
source — keep scoping to it.

When source(s) ARE named, the phrasing decides the mode:

- COMBINED — one or more named sources with NO fallback order ("in Google Drive"; "search \
A and B"; "check both A and B"): scope to all of them every cycle, regardless of previous \
cycles. A single named source is COMBINED — scope to it.

- BACKOFF ("check A first, then B", "try A; if nothing, then B" — an order): scope to ONE \
source per cycle. By DEFAULT ADVANCE — scope to the first named source NOT in any previous \
cycle's searched_sources; a reworded retry of the same search keeps advancing. BUT if this \
cycle's queries are about a clearly DIFFERENT topic than the previous cycle's, re-search the \
source the previous cycle used — it has not been searched for this new topic. Once all named \
sources have been tried, scope to all of them.

Only scope to sources listed in the Valid sources section below. If a named source is not \
listed there, ignore it and scope to the named sources that ARE listed; return [] only when \
none of the named sources are listed.

## Conversation history

{conversation_history}

## Current cycle queries

{current_cycle_queries}

## Previous cycles of this user query

{previous_cycles}

## Valid sources

{valid_sources}

## Guidance reminder

COMBINED ("A and B"): scope to all named sources, every cycle.
BACKOFF ("A first, then B"): by DEFAULT ADVANCE to the first named source not in previous \
cycles' searched_sources (a reworded retry keeps advancing). If this cycle's queries are \
about a clearly DIFFERENT topic than the previous cycle's, re-search the source the previous \
cycle used.
If no source is named anywhere in the conversation, return [].

## Output format

Output a comma separated list of sources within brackets:
[source_1, source_2]

Do not include any formatting, explanations, or other text aside from the list. Provide an \
empty list [] if no source should be scoped this cycle.

## Query reminder

The user's query is:
{last_user_query}

CRITICAL: output only the comma separated list of sources.
""".strip()

# Use the following for easy viewing of prompts
if __name__ == "__main__":
    print(TIME_SCOPE_DECISION_PROMPT)
    print("------------------")
    print(SOURCE_SCOPE_DECISION_PROMPT)
