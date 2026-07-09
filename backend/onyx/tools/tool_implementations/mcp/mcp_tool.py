import json
from datetime import datetime
from typing import Any

from mcp.client.auth import OAuthClientProvider

from onyx.chat.emitter import Emitter
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.context.search.models import SearchDocsResponse
from onyx.db.enums import MCPAuthenticationType
from onyx.db.enums import MCPTransport
from onyx.db.models import MCPConnectionConfig
from onyx.db.models import MCPServer
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import CustomToolDelta
from onyx.server.query_and_chat.streaming_models import CustomToolStart
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.tools.interface import Tool
from onyx.tools.models import CustomToolCallSummary
from onyx.tools.models import MCPToolOverrideKwargs
from onyx.tools.models import ToolResponse
from onyx.tools.tool_implementations.mcp.mcp_client import call_mcp_tool
from onyx.tools.tool_name import sanitize_tool_name
from onyx.utils.logger import setup_logger

logger = setup_logger()

# Headers that cannot be overridden by user requests to prevent security issues
# Host header is particularly critical - it can be used for Host Header Injection attacks
# to route requests to unintended internal servers
DENYLISTED_MCP_HEADERS = {
    "host",  # Prevents Host Header Injection attacks
}

# Prefix used for document ids of citable documents surfaced from MCP tool results.
MCP_DOC_ID_PREFIX = "MICROO_MCP_DOC_"
# Max characters kept for the source-card preview (blurb) of an MCP document.
MCP_DOC_BLURB_MAX_CHARS = 400

# TODO: for now we're fitting MCP tool responses into the CustomToolCallSummary class
# In the future we may want custom handling for MCP tool responses
# class MCPToolCallSummary(BaseModel):
#     tool_name: str
#     server_url: str
#     tool_result: Any
#     server_name: str


def _normalize_parameters_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    # Azure OpenAI rejects object schemas that omit `properties` with
    # "object schema missing properties". MCP servers (e.g. AWS Knowledge MCP's
    # aws___list_regions) may legally return `{"type": "object"}` with no
    # properties for zero-arg tools, so seed `properties: {}` ourselves.
    if not schema:
        return {"type": "object", "properties": {}}
    if schema.get("type", "object") == "object" and "properties" not in schema:
        return {**schema, "type": "object", "properties": {}}
    return schema


def _coerce_metadata(value: Any) -> dict[str, str | list[str]]:
    """Coerce arbitrary MCP metadata into the str/list[str] shape SearchDoc expects."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, str | list[str]] = {}
    for k, v in value.items():
        if isinstance(v, list):
            result[str(k)] = [str(item) for item in v]
        else:
            result[str(k)] = str(v)
    return result


def _parse_updated_at(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp; return None if absent or malformed."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class MCPTool(Tool[None]):
    """Tool implementation for MCP (Model Context Protocol) servers"""

    def __init__(
        self,
        tool_id: int,
        emitter: Emitter,
        mcp_server: MCPServer,  # TODO: these should be basemodels instead of db objects
        tool_name: str,
        tool_description: str,
        tool_definition: dict[str, Any],
        connection_config: MCPConnectionConfig | None = None,
        user_email: str = "",
        user_id: str = "",
        user_oauth_token: str | None = None,
        additional_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(emitter=emitter)

        self._id = tool_id
        self.mcp_server = mcp_server
        self.connection_config = connection_config
        self.user_email = user_email
        self._user_id = user_id
        self._user_oauth_token = user_oauth_token
        self._additional_headers = additional_headers or {}

        self._mcp_tool_name = tool_name
        self._name = tool_name  # NOTE: this may change in _disambiguate_mcp_tool_names
        self._tool_definition = tool_definition
        self._description = tool_description
        self._display_name = tool_definition.get("displayName", tool_name)
        self._llm_name = sanitize_tool_name(f"mcp_{mcp_server.name}_{tool_name}")

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def display_name(self) -> str:
        return self._display_name

    def use_disambiguated_name(self) -> None:
        self._name = self._llm_name

    def tool_definition(self) -> dict:
        """Return the tool definition from the MCP server"""
        # Convert MCP tool definition to OpenAI function calling format
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": _normalize_parameters_schema(self._tool_definition),
            },
        }

    def emit_start(self, placement: Placement) -> None:
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=CustomToolStart(tool_name=self._name),
            )
        )

    def _build_citable_documents_response(
        self,
        tool_result: Any,
        starting_citation_num: int,
        placement: Placement,
    ) -> ToolResponse | None:
        """Surface a document-shaped MCP result as citable SearchDocs.

        Expects the tool result to be (or contain) JSON of the form
        ``{"documents": [{"id", "title", "content", "url"?, "updated_at"?,
        "metadata"?}, ...]}``. Returns None when the result is not
        document-shaped so the caller falls back to the raw JSON path.
        """
        if isinstance(tool_result, str):
            try:
                parsed = json.loads(tool_result)
            except (json.JSONDecodeError, ValueError):
                return None
        elif isinstance(tool_result, dict):
            parsed = tool_result
        else:
            return None

        if not isinstance(parsed, dict):
            return None
        raw_documents = parsed.get("documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            return None

        search_docs: list[SearchDoc] = []
        citation_mapping: dict[int, str] = {}
        llm_sections: list[str] = []
        citation_num = starting_citation_num

        for raw in raw_documents:
            if not isinstance(raw, dict):
                continue
            doc_id = raw.get("id")
            title = raw.get("title")
            content = raw.get("content")
            # Required fields; skip malformed entries rather than failing the tool.
            if not doc_id or not title or content is None:
                continue

            document_id = f"{MCP_DOC_ID_PREFIX}{doc_id}"
            url = raw.get("url") or None
            content_str = str(content)

            search_docs.append(
                SearchDoc(
                    document_id=document_id,
                    chunk_ind=0,
                    semantic_identifier=str(title),
                    link=url,
                    blurb=content_str[:MCP_DOC_BLURB_MAX_CHARS],
                    source_type=DocumentSource.NOT_APPLICABLE,
                    boost=0,
                    hidden=False,
                    metadata=_coerce_metadata(raw.get("metadata")),
                    score=None,
                    match_highlights=[],
                    updated_at=_parse_updated_at(raw.get("updated_at")),
                    is_internet=False,
                )
            )
            citation_mapping[citation_num] = document_id
            # Surface the raw `id` so the model can reference this document in
            # follow-up tool calls (e.g. fetching its summary / full content).
            llm_sections.append(f"[{citation_num}] {title} (id: {doc_id})\n{content_str}")
            citation_num += 1

        if not search_docs:
            return None

        llm_facing_response = "\n\n".join(llm_sections)

        # Light delta so the tool-run card isn't empty; chips + source panel
        # come from the SearchDocsResponse citation flow, not this packet.
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=CustomToolDelta(
                    tool_name=self._name,
                    response_type="json",
                    data={
                        "documents_count": len(search_docs),
                        "documents": [
                            {
                                "title": doc.semantic_identifier,
                                "url": doc.link or "",
                            }
                            for doc in search_docs
                        ],
                    },
                ),
            )
        )

        return ToolResponse(
            rich_response=SearchDocsResponse(
                search_docs=search_docs,
                citation_mapping=citation_mapping,
                displayed_docs=search_docs,
            ),
            llm_facing_response=llm_facing_response,
        )

    def run(
        self,
        placement: Placement,
        override_kwargs: MCPToolOverrideKwargs | None = None,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        """Execute the MCP tool by calling the MCP server"""
        try:
            # Build headers with proper precedence:
            # 1. Start with additional headers from API request (filled in first, excluding denylisted)
            # 2. Override with connection config headers (from DB) - these take precedence
            # 3. Override Authorization header with OAuth token if present
            headers: dict[str, str] = {}

            # Priority 1: Additional headers from API request (filled in first)
            # Filter out denylisted headers to prevent security issues (e.g., Host Header Injection)
            if self._additional_headers:
                filtered_headers = {
                    k: v
                    for k, v in self._additional_headers.items()
                    if k.lower() not in DENYLISTED_MCP_HEADERS
                }
                if filtered_headers:
                    headers.update(filtered_headers)
                # Log if any denylisted headers were provided (for security monitoring)
                denylisted_provided = [
                    k
                    for k in self._additional_headers.keys()
                    if k.lower() in DENYLISTED_MCP_HEADERS
                ]
                if denylisted_provided:
                    logger.warning(
                        "MCP tool '%s' received denylisted headers that were filtered: %s",
                        self._name,
                        denylisted_provided,
                    )

            # Priority 2: Base headers from connection config (DB) - overrides request
            if self.connection_config and self.connection_config.config:
                config_dict = self.connection_config.config.get_value(apply_mask=False)
                headers.update(config_dict.get("headers", {}))

            # Priority 3: For pass-through OAuth, use the user's login OAuth token
            if self._user_oauth_token:
                headers["Authorization"] = f"Bearer {self._user_oauth_token}"

            # Check if this is an authentication issue before making the call
            is_passthrough_oauth = (
                self.mcp_server.auth_type == MCPAuthenticationType.PT_OAUTH
            )
            requires_auth = (
                self.mcp_server.auth_type != MCPAuthenticationType.NONE
                and self.mcp_server.auth_type is not None
            )
            has_auth_config = (
                (self.connection_config is not None and bool(headers))
                or bool(self._additional_headers)
            ) or (is_passthrough_oauth and self._user_oauth_token is not None)

            if requires_auth and not has_auth_config:
                # Authentication required but not configured
                auth_error_msg = (
                    f"The {self._name} tool from {self.mcp_server.name} requires authentication "
                    f"but no credentials have been provided. Tell the user to use the MCP dropdown in the "
                    f"chat bar to authenticate with the {self.mcp_server.name} server before "
                    f"using this tool."
                )
                logger.warning(
                    "Authentication required for MCP tool '%s' but no credentials found",
                    self._name,
                )

                error_result = {"error": auth_error_msg}
                llm_facing_response = json.dumps(error_result)

                # Emit CustomToolDelta packet
                self.emitter.emit(
                    Packet(
                        placement=placement,
                        obj=CustomToolDelta(
                            tool_name=self._name,
                            response_type="json",
                            data=error_result,
                        ),
                    )
                )

                return ToolResponse(
                    rich_response=CustomToolCallSummary(
                        tool_name=self._name,
                        response_type="json",
                        tool_result=error_result,
                    ),
                    llm_facing_response=llm_facing_response,
                )

            # For OAuth servers, construct OAuthClientProvider so the MCP SDK
            # can refresh expired tokens automatically
            auth: OAuthClientProvider | None = None
            if (
                self.mcp_server.auth_type == MCPAuthenticationType.OAUTH
                and self.connection_config is not None
                and self._user_id
            ):
                if self.mcp_server.transport == MCPTransport.SSE:
                    logger.warning(
                        "MCP tool '%s': OAuth token refresh is not supported for SSE transport — auth provider will be ignored. Re-authentication may be required after token expiry.",
                        self._name,
                    )
                else:
                    from onyx.server.features.mcp.api import make_oauth_provider
                    from onyx.server.features.mcp.api import UNUSED_RETURN_PATH

                    # user_id is the requesting user's UUID; safe here because
                    # UNUSED_RETURN_PATH ensures redirect_handler raises immediately
                    # and user_id is never consulted for Redis state lookups.
                    auth = make_oauth_provider(
                        self.mcp_server,
                        self._user_id,
                        UNUSED_RETURN_PATH,
                        self.connection_config.id,
                        None,
                    )

            tool_result = call_mcp_tool(
                self.mcp_server.server_url,
                self._mcp_tool_name,
                llm_kwargs,
                connection_headers=headers,
                transport=self.mcp_server.transport or MCPTransport.STREAMABLE_HTTP,
                auth=auth,
            )

            logger.info("MCP tool '%s' executed successfully", self._name)

            # If this server opts in, try to surface the result as citable
            # documents (chip citations + source panel). Falls back to the
            # plain custom-tool JSON path when the result isn't document-shaped.
            if self.mcp_server.emit_documents and override_kwargs is not None:
                citable_response = self._build_citable_documents_response(
                    tool_result=tool_result,
                    starting_citation_num=override_kwargs.starting_citation_num,
                    placement=placement,
                )
                if citable_response is not None:
                    return citable_response

            # Format the tool result for response
            tool_result_dict = {"tool_result": tool_result}
            llm_facing_response = json.dumps(tool_result_dict)

            # Emit CustomToolDelta packet
            self.emitter.emit(
                Packet(
                    placement=placement,
                    obj=CustomToolDelta(
                        tool_name=self._name,
                        response_type="json",
                        data=tool_result_dict,
                    ),
                )
            )

            return ToolResponse(
                rich_response=CustomToolCallSummary(
                    tool_name=self._name,
                    response_type="json",
                    tool_result=tool_result_dict,
                ),
                llm_facing_response=llm_facing_response,
            )

        except Exception as e:
            error_str = str(e).lower()
            logger.error("Failed to execute MCP tool '%s': %s", self._name, e)

            # Check for authentication-related errors
            auth_error_indicators = [
                "401",
                "unauthorized",
                "authentication",
                "auth",
                "forbidden",
                "access denied",
                "invalid token",
                "invalid api key",
                "invalid credentials",
                "please reconnect to the server",
            ]

            is_auth_error = any(
                indicator in error_str for indicator in auth_error_indicators
            )

            if is_auth_error:
                auth_error_msg = (
                    f"Authentication failed for the {self._name} tool from {self.mcp_server.name}. "
                    f"Please use the MCP dropdown in the chat bar to update your credentials "
                    f"for the {self.mcp_server.name} server. Original error: {str(e)}"
                )
                error_result = {"error": auth_error_msg}
            else:
                error_result = {"error": f"Tool execution failed: {str(e)}"}

            llm_facing_response = json.dumps(error_result)

            # Emit CustomToolDelta packet
            self.emitter.emit(
                Packet(
                    placement=placement,
                    obj=CustomToolDelta(
                        tool_name=self._name,
                        response_type="json",
                        data=error_result,
                    ),
                )
            )

            return ToolResponse(
                rich_response=CustomToolCallSummary(
                    tool_name=self._name,
                    response_type="json",
                    tool_result=error_result,
                ),
                llm_facing_response=llm_facing_response,
            )
