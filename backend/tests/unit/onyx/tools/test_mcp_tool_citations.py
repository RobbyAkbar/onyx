"""Tests for MCPTool citable-document support.

When an MCP server opts in via ``emit_documents``, a tool result shaped as
``{"documents": [...]}`` is surfaced as citable SearchDocs (chip citations +
source panel) instead of raw custom-tool JSON. When it does not opt in, or the
result is not document-shaped, the original custom-tool JSON path is preserved.
"""

import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from onyx.context.search.models import SearchDocsResponse
from onyx.db.enums import MCPAuthenticationType
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import CustomToolCallSummary
from onyx.tools.models import MCPToolOverrideKwargs
from onyx.tools.tool_implementations.mcp.mcp_tool import MCP_DOC_ID_PREFIX
from onyx.tools.tool_implementations.mcp.mcp_tool import MCPTool


def _make_tool(emit_documents: bool = True) -> MCPTool:
    mcp_server = MagicMock()
    mcp_server.name = "microo"
    mcp_server.server_url = "http://mcp.example"
    mcp_server.auth_type = MCPAuthenticationType.NONE
    mcp_server.transport = None
    mcp_server.emit_documents = emit_documents
    return MCPTool(
        tool_id=1,
        emitter=MagicMock(),
        mcp_server=mcp_server,
        tool_name="search_docs",
        tool_description="Search Microo docs",
        tool_definition={"type": "object", "properties": {}},
        connection_config=None,
    )


_ENVELOPE = json.dumps(
    {
        "documents": [
            {
                "id": "abc-123",
                "title": "CV ROBBY 2026.pdf",
                "url": "https://microo.apps.paramalab.dev/files/details/abc-123",
                "content": "Robby Akbar\nBekasi, Indonesia",
                "updated_at": "2026-07-08T09:05:27Z",
                "metadata": {"collectionId": "col-1"},
            },
            {
                "id": "def-456",
                "title": "Second doc",
                "content": "second body",
            },
        ]
    }
)


class TestBuildCitableDocumentsResponse:
    def test_valid_envelope_builds_search_docs_response(self) -> None:
        tool = _make_tool()
        resp = tool._build_citable_documents_response(
            tool_result=_ENVELOPE,
            starting_citation_num=5,
            placement=Placement(turn_index=0),
        )
        assert resp is not None
        rich = resp.rich_response
        assert isinstance(rich, SearchDocsResponse)
        assert [d.document_id for d in rich.search_docs] == [
            f"{MCP_DOC_ID_PREFIX}abc-123",
            f"{MCP_DOC_ID_PREFIX}def-456",
        ]
        # Citation numbers start at the provided base and increment
        assert rich.citation_mapping == {
            5: f"{MCP_DOC_ID_PREFIX}abc-123",
            6: f"{MCP_DOC_ID_PREFIX}def-456",
        }
        # LLM-facing text carries [N] labels matching the mapping so the model
        # knows which number to cite.
        assert "[5] CV ROBBY 2026.pdf" in resp.llm_facing_response
        assert "[6] Second doc" in resp.llm_facing_response
        # url -> link, metadata coerced to str, updated_at parsed
        first = rich.search_docs[0]
        assert first.link == "https://microo.apps.paramalab.dev/files/details/abc-123"
        assert first.metadata == {"collectionId": "col-1"}
        assert first.updated_at is not None
        # doc without url -> link None
        assert rich.search_docs[1].link is None

    def test_non_json_returns_none(self) -> None:
        tool = _make_tool()
        assert (
            tool._build_citable_documents_response(
                tool_result="not json at all",
                starting_citation_num=1,
                placement=Placement(turn_index=0),
            )
            is None
        )

    def test_missing_documents_key_returns_none(self) -> None:
        tool = _make_tool()
        assert (
            tool._build_citable_documents_response(
                tool_result=json.dumps({"result": "ok"}),
                starting_citation_num=1,
                placement=Placement(turn_index=0),
            )
            is None
        )

    def test_empty_documents_returns_none(self) -> None:
        tool = _make_tool()
        assert (
            tool._build_citable_documents_response(
                tool_result=json.dumps({"documents": []}),
                starting_citation_num=1,
                placement=Placement(turn_index=0),
            )
            is None
        )

    def test_entries_missing_required_fields_are_skipped(self) -> None:
        tool = _make_tool()
        envelope = json.dumps(
            {"documents": [{"id": "x"}, {"title": "no id", "content": "c"}]}
        )
        # First entry missing title/content, second missing id -> all skipped.
        assert (
            tool._build_citable_documents_response(
                tool_result=envelope,
                starting_citation_num=1,
                placement=Placement(turn_index=0),
            )
            is None
        )


class TestRunCitableBranch:
    def test_run_emits_documents_when_opted_in(self) -> None:
        tool = _make_tool(emit_documents=True)
        with patch(
            "onyx.tools.tool_implementations.mcp.mcp_tool.call_mcp_tool",
            return_value=_ENVELOPE,
        ):
            resp = tool.run(
                Placement(turn_index=0),
                override_kwargs=MCPToolOverrideKwargs(starting_citation_num=1),
            )
        assert isinstance(resp.rich_response, SearchDocsResponse)

    def test_run_falls_back_to_json_when_not_opted_in(self) -> None:
        tool = _make_tool(emit_documents=False)
        with patch(
            "onyx.tools.tool_implementations.mcp.mcp_tool.call_mcp_tool",
            return_value=_ENVELOPE,
        ):
            resp = tool.run(
                Placement(turn_index=0),
                override_kwargs=MCPToolOverrideKwargs(starting_citation_num=1),
            )
        assert isinstance(resp.rich_response, CustomToolCallSummary)

    def test_run_falls_back_when_result_not_document_shaped(self) -> None:
        tool = _make_tool(emit_documents=True)
        with patch(
            "onyx.tools.tool_implementations.mcp.mcp_tool.call_mcp_tool",
            return_value=json.dumps({"result": "plain"}),
        ):
            resp = tool.run(
                Placement(turn_index=0),
                override_kwargs=MCPToolOverrideKwargs(starting_citation_num=1),
            )
        assert isinstance(resp.rich_response, CustomToolCallSummary)


if __name__ == "__main__":
    pytest.main([__file__, "-xv"])
