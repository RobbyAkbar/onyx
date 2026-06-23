from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic import BaseModel
from pydantic import Field
from sqlalchemy.orm import Session

from onyx.chat.chat_utils import create_chat_session_from_request
from onyx.chat.process_message import handle_stream_message_objects
from onyx.configs.constants import DEFAULT_PERSONA_ID
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.db.models import User
from onyx.db.persona import upsert_persona
from onyx.db.tools import get_builtin_tool
from onyx.llm.override_models import LLMOverride
from onyx.server.query_and_chat.models import ChatSessionCreationRequest
from onyx.server.query_and_chat.models import SendMessageRequest
from onyx.server.query_and_chat.streaming_models import AgentResponseDelta
from onyx.tools.tool_implementations.search.search_tool import SearchTool

# The search tool binds its data-layer helpers in its own namespace; patch there.
_SEARCH_TOOL_MODULE = "onyx.tools.tool_implementations.search.search_tool"

# Human-readable trace of every case: scope per search, what was retrieved, and
# what was sent back to the main agent. Appended to on each run_eval_case.
_TRACE_PATH = Path(__file__).parents[3] / "log" / "connector_filter_eval_trace.log"


# --- Spec ------------------------------------------------------------------


class MockDoc(BaseModel):
    """A document internal search can return, served when a search is scoped to
    its `source` (or scoped to nothing)."""

    source: DocumentSource
    title: str
    content: str
    document_id: str | None = None
    link: str | None = None
    score: float = 0.9
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def resolved_id(self) -> str:
        slug = self.title.lower().replace(" ", "_")
        return self.document_id or f"{self.source.value}__{slug}"


class EvalCase(BaseModel):
    name: str
    # Connectors the agent is told exist; drives source-filter inference.
    connected_sources: list[DocumentSource]
    # Corpus, served filtered by the source filter active on each search.
    mock_docs: list[MockDoc]
    # User message; include any routing instructions here.
    query: str
    # 0 == Onyx default assistant (has internal_search).
    persona_id: int = DEFAULT_PERSONA_ID
    # Force the first tool call to be internal_search; later calls stay agent-driven.
    force_first_search: bool = True
    # LLM override by the provider's *configured name* (not slug). Defaults to
    # the EVAL_LLM_* env vars so cases never silently use the wrong model; when
    # unset the tenant's default provider is used.
    llm_provider: str | None = Field(
        default_factory=lambda: os.environ.get("EVAL_LLM_PROVIDER")
    )
    llm_model: str | None = Field(
        default_factory=lambda: os.environ.get("EVAL_LLM_MODEL")
    )


# --- Results ---------------------------------------------------------------


class RecordedSearch(BaseModel):
    """One internal_search tool call. Its keyword/semantic sub-queries share a
    source filter and are collapsed into this record."""

    invocation_index: int
    # Source filter applied to retrieval; None == searched everything.
    source_filter: list[DocumentSource] | None
    queries: list[str]
    returned_doc_ids: list[str]
    returned_sources: list[DocumentSource]
    # The tool's llm_facing_response — exactly what the main agent receives back
    # (includes the scope breadcrumb).
    llm_facing_response: str = ""

    @property
    def scope(self) -> str:
        if self.source_filter is None:
            return "ALL"
        return ",".join(s.value for s in self.source_filter)


class EvalResult(BaseModel):
    case_name: str
    internal_searches: list[RecordedSearch]
    final_answer: str

    @property
    def applied_filters(self) -> list[list[DocumentSource] | None]:
        """Source filter applied to each search, in order."""
        return [s.source_filter for s in self.internal_searches]

    def report(self, *, full: bool = False) -> str:
        """Readable trace of the flow. `full` keeps the entire response text sent
        back to the agent; otherwise it is truncated for console output."""
        limit = 100_000 if full else 600
        lines = [f"=== {self.case_name} ==="]
        if not self.internal_searches:
            lines.append("  <no internal searches>")
        for s in self.internal_searches:
            returned = (
                ", ".join(
                    f"{doc_id}({src.value})"
                    for doc_id, src in zip(s.returned_doc_ids, s.returned_sources)
                )
                or "<none>"
            )
            sent = s.llm_facing_response.strip() or "<empty>"
            lines += [
                f"  search #{s.invocation_index}:",
                f"    scope selected: {s.scope}",
                f"    queries:        {s.queries}",
                f"    returned docs:  {returned}",
                f"    sent to agent:  {sent[:limit]}",
            ]
        lines.append(f"final answer: {self.final_answer.strip()[:limit]}")
        return "\n".join(lines)


# --- Recording -------------------------------------------------------------


class _Invocation:
    """Mutable accumulator for one internal_search call."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.source_filter: list[DocumentSource] | None = None
        self._filter_set = False
        self.queries: list[str] = []
        self.doc_ids: list[str] = []
        self.sources: list[DocumentSource] = []
        self.llm_facing_response = ""

    def add(
        self,
        query: str,
        source_filter: list[DocumentSource] | None,
        docs: list[MockDoc],
    ) -> None:
        if not self._filter_set:  # all sub-queries share one filter; first wins
            self.source_filter = source_filter
            self._filter_set = True
        if query and query not in self.queries:
            self.queries.append(query)
        for doc in docs:
            if doc.resolved_id not in self.doc_ids:
                self.doc_ids.append(doc.resolved_id)
                self.sources.append(doc.source)

    def freeze(self) -> RecordedSearch:
        return RecordedSearch(
            invocation_index=self.index,
            source_filter=self.source_filter,
            queries=self.queries,
            returned_doc_ids=self.doc_ids,
            returned_sources=self.sources,
            llm_facing_response=self.llm_facing_response,
        )


class _SearchRecorder:
    """Collects internal-search invocations as the agent runs. Thread-safe
    because `search_pipeline` runs in worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._invocations: list[_Invocation] = []

    def open_invocation(self) -> None:
        # Called once per tool call (via fetch_unique_document_sources), before
        # its parallel search_pipeline calls fan out — a reliable boundary.
        with self._lock:
            self._invocations.append(_Invocation(len(self._invocations)))

    def record(
        self,
        query: str,
        source_filter: list[DocumentSource] | None,
        docs: list[MockDoc],
    ) -> None:
        with self._lock:
            if not self._invocations:
                self._invocations.append(_Invocation(0))
            self._invocations[-1].add(query, source_filter, docs)

    def attach_response(self, llm_facing_response: str) -> None:
        # Set on the most recent invocation still missing a response. Reliable
        # for serial (sequential-fallback) runs; parallel same-message calls may
        # mis-pair, which is acceptable for a debug trace.
        with self._lock:
            for inv in reversed(self._invocations):
                if not inv.llm_facing_response:
                    inv.llm_facing_response = llm_facing_response
                    return

    def freeze(self) -> list[RecordedSearch]:
        with self._lock:
            return [inv.freeze() for inv in self._invocations]


# --- Mocked data layer -----------------------------------------------------


def _to_chunk(doc: MockDoc) -> InferenceChunk:
    return InferenceChunk(
        chunk_id=0,
        blurb=doc.content[:200],
        content=doc.content,
        source_links={0: doc.link} if doc.link else None,
        image_file_id=None,
        section_continuation=False,
        document_id=doc.resolved_id,
        source_type=doc.source,
        semantic_identifier=doc.title,
        title=doc.title,
        boost=0,
        score=doc.score,
        hidden=False,
        metadata=dict(doc.metadata),
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=None,
        primary_owners=None,
        secondary_owners=None,
        large_chunk_reference_ids=[],
        is_federated=False,
        file_id=None,
    )


@contextmanager
def _mocked_retrieval(case: EvalCase, recorder: _SearchRecorder) -> Iterator[None]:
    def fake_connected_sources(_db_session: Session) -> list[DocumentSource]:
        recorder.open_invocation()
        return list(case.connected_sources)

    def fake_search_pipeline(*args: object, **kwargs: object) -> list[InferenceChunk]:
        req = kwargs.get("chunk_search_request") or (args[0] if args else None)
        filters = getattr(req, "user_selected_filters", None)
        source_filter = (
            list(filters.source_type) if filters and filters.source_type else None
        )
        matched = [
            doc
            for doc in case.mock_docs
            if source_filter is None or doc.source in source_filter
        ]
        recorder.record(str(getattr(req, "query", "")), source_filter, matched)
        return [_to_chunk(doc) for doc in matched]

    # Wrap run() to capture what the tool sends back to the main agent.
    original_run = SearchTool.run

    def traced_run(self: SearchTool, *args: Any, **kwargs: Any) -> Any:
        response = original_run(self, *args, **kwargs)
        recorder.attach_response(response.llm_facing_response or "")
        return response

    with (
        # The data layer is mocked, so the connector-existence gate is moot.
        patch.object(SearchTool, "is_available", return_value=True),
        patch.object(SearchTool, "run", traced_run),
        patch(
            f"{_SEARCH_TOOL_MODULE}.fetch_unique_document_sources",
            side_effect=fake_connected_sources,
        ),
        patch(
            f"{_SEARCH_TOOL_MODULE}.search_pipeline", side_effect=fake_search_pipeline
        ),
        # Keep all retrieval flowing through the single search_pipeline seam.
        patch(
            f"{_SEARCH_TOOL_MODULE}.get_federated_retrieval_functions", return_value=[]
        ),
    ):
        yield


# --- Runner ----------------------------------------------------------------


def create_eval_persona(
    db_session: Session, *, system_prompt: str, name: str, user: User | None = None
) -> int:
    """Create a non-default persona carrying `system_prompt`, with only the
    internal_search tool. Returns its id (pass as `EvalCase.persona_id`).

    A persona (not a project) injects the prompt without setting a
    project_id_filter, which would disable the source scoping under test.
    """
    with patch.object(SearchTool, "is_available", return_value=True):
        persona = upsert_persona(
            user=user,
            name=name,
            description="connector-filter eval agent",
            starter_messages=None,
            system_prompt=system_prompt,
            task_prompt=None,
            datetime_aware=False,
            is_public=True,
            db_session=db_session,
            tool_ids=[get_builtin_tool(db_session, SearchTool).id],
            replace_base_system_prompt=False,
        )
    return persona.id


def _send_and_record(
    *,
    chat_session_id: Any,
    query: str,
    name: str,
    case: EvalCase,
    db_session: Session,
    user: User,
) -> EvalResult:
    """Send one user message to an existing session and record its searches."""
    llm_override = (
        LLMOverride(model_provider=case.llm_provider, model_version=case.llm_model)
        if case.llm_provider and case.llm_model
        else None
    )
    request = SendMessageRequest(
        message=query,
        chat_session_id=chat_session_id,
        stream=True,
        forced_tool_id=(
            get_builtin_tool(db_session, SearchTool).id
            if case.force_first_search
            else None
        ),
        # internal_search_filters left unset; setting it skips the inference.
        llm_override=llm_override,
    )

    recorder = _SearchRecorder()
    answer_parts: list[str] = []
    with _mocked_retrieval(case, recorder):
        for part in handle_stream_message_objects(new_msg_req=request, user=user):
            obj = getattr(part, "obj", None)
            if isinstance(obj, AgentResponseDelta):
                answer_parts.append(obj.content)

    result = EvalResult(
        case_name=name,
        internal_searches=recorder.freeze(),
        final_answer="".join(answer_parts),
    )
    _append_trace(result)
    return result


def run_eval_case(case: EvalCase, db_session: Session, user: User) -> EvalResult:
    """Drive the chat flow with a real LLM and return the recorded searches."""
    chat_session = create_chat_session_from_request(
        chat_session_request=ChatSessionCreationRequest(persona_id=case.persona_id),
        user=user,
        db_session=db_session,
    )
    return _send_and_record(
        chat_session_id=chat_session.id,
        query=case.query,
        name=case.name,
        case=case,
        db_session=db_session,
        user=user,
    )


def run_eval_chain(
    case: EvalCase, messages: list[str], db_session: Session, user: User
) -> list[EvalResult]:
    """Send several user messages to ONE chat session (a real multi-turn chain)
    and return one EvalResult per message, each recording only that message's
    searches. `case.query` is ignored; `messages` drives the turns.

    Use this to test how a directive in an earlier turn affects (or should NOT
    affect) the source scope chosen for a later turn.
    """
    chat_session = create_chat_session_from_request(
        chat_session_request=ChatSessionCreationRequest(persona_id=case.persona_id),
        user=user,
        db_session=db_session,
    )
    results: list[EvalResult] = []
    for turn, message in enumerate(messages):
        results.append(
            _send_and_record(
                chat_session_id=chat_session.id,
                query=message,
                name=f"{case.name}[turn {turn}]",
                case=case,
                db_session=db_session,
                user=user,
            )
        )
    return results


def _append_trace(result: EvalResult) -> None:
    """Append the full flow trace to `_TRACE_PATH` (best-effort)."""
    try:
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TRACE_PATH.open("a") as f:
            f.write("\n" + result.report(full=True) + "\n")
        print(f"[connector-filter-eval] trace appended to {_TRACE_PATH}")
    except OSError as e:
        print(f"[connector-filter-eval] could not write trace: {e}")
