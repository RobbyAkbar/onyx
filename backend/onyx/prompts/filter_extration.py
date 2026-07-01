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
You scope an internal search to a time filter, from the user's conversation. When the \
conversation EXPLICITLY refers to a time the documents should fall within, set a filter on \
each document's last-updated date; when it refers to none, return (None, None) (search \
across all time). You scope only by time.

## Guidance

Set a time filter when a time is EXPLICITLY referenced — in the latest message, or in an \
earlier turn it continues. NEVER infer a time from the topic alone. A date that names the \
subject or title of the document sought ("the 2020 GDPR docs", "the FY21 plan") is NOT a \
filter — it says WHAT the document is, not WHEN it was updated; let content search match it. \
If no time is referenced, return (None, None).

When a time IS referenced, the phrasing decides the bounds:

- LOWER BOUND ONLY — an open-ended time toward now, including a rolling window ("since \
March", "in the last 2 weeks", "recently"). Set start; leave end None — a rolling window has \
no upper bound, so do NOT set end to today.

- UPPER BOUND ONLY — an open-ended time toward the past ("before 2023", "older than \
January"). Set end; leave start None.

- BOTH BOUNDS — a completed, named calendar period ("last January", "Q1 2025", "in 2022", \
"between March and June", or a single day like "March 25 2024"). Set start and end to its \
first and last day.

- NO BOUND — a vague preference for fresh results with no actual time ("the latest", "most \
recent"). Return (None, None).

## Conversation history

{conversation_history}

## Current date

Today is {current_day_time_str}. Resolve every time the user refers to against today, into \
concrete dates yourself.

## Guidance reminder

LOWER / UPPER BOUND: an open-ended time sets one bound and leaves the other None — a rolling \
window up to now ("the last 2 weeks") leaves end None, not today.
BOTH BOUNDS: a completed, named calendar period ("last quarter", "in 2022") sets both bounds.
NEVER filter on a date that names the document's subject/title, and return (None, None) when \
no time is referenced.

## Output format

Output ONLY the time filter as a pair: (start, end)
- start (lower bound) and end (upper bound) are each a date "YYYY-MM-DD" or None; both are \
inclusive, and None means no bound on that side.

Examples:
- "in the last 2 weeks" (today is 2026-06-30) → (2026-06-16, None)
- "before 2023" → (None, 2022-12-31)
- "in January 2025" → (2025-01-01, 2025-01-31)
- "the 2020 GDPR docs" → (None, None)
- "the latest updates" → (None, None)

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
