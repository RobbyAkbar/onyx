import os
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from onyx.db.enums import LLMModelFlowType
from onyx.db.llm import fetch_existing_llm_providers
from onyx.db.llm import fetch_llm_provider_view
from onyx.db.models import User
from onyx.llm.factory import get_default_llm
from onyx.llm.factory import llm_from_provider
from onyx.llm.interfaces import LLM
from tests.external_dependency_unit.answer.conftest import (  # noqa: F401
    mock_external_deps,
)
from tests.external_dependency_unit.answer.conftest import mock_file_store  # noqa: F401
from tests.external_dependency_unit.answer.conftest import mock_gpu_status  # noqa: F401
from tests.external_dependency_unit.answer.conftest import (  # noqa: F401
    mock_nlp_embeddings_post,
)
from tests.external_dependency_unit.answer.conftest import (  # noqa: F401
    mock_vespa_query,
)
from tests.external_dependency_unit.conftest import create_test_user

# This suite makes real LLM calls, so it is disabled by default and only runs
# when explicitly opted in.
_RUN_FLAG_ENV = "RUN_CONNECTOR_FILTER_EVAL"
_PACKAGE_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.environ.get(_RUN_FLAG_ENV):
        return
    skip = pytest.mark.skip(reason=f"manual-only eval; set {_RUN_FLAG_ENV}=1 to run")
    for item in items:
        if _PACKAGE_DIR in item.path.parents:
            item.add_marker(skip)


@pytest.fixture
def eval_user(
    db_session: Session,
    full_deployment_setup: None,  # noqa: ARG001
    mock_external_deps: None,  # noqa: ARG001, F811
) -> Generator[User, None, None]:
    """A test user, available only when an LLM provider is configured (cases
    make real LLM calls)."""
    if not fetch_existing_llm_providers(db_session, [LLMModelFlowType.CHAT]):
        pytest.skip("no LLM provider configured; set one up to run the eval")
    yield create_test_user(db_session, email_prefix="connector_filter_eval")


@pytest.fixture
def eval_llm(
    db_session: Session,
    full_deployment_setup: None,  # noqa: ARG001
    mock_external_deps: None,  # noqa: ARG001, F811
) -> LLM:
    """The LLM used for direct decide_search_scope tests. Honors the EVAL_LLM_*
    env vars (same as EvalCase) so the cheap tier is used; falls back to the
    tenant default provider."""
    provider_name = os.environ.get("EVAL_LLM_PROVIDER")
    model = os.environ.get("EVAL_LLM_MODEL")
    if provider_name and model:
        view = fetch_llm_provider_view(db_session, provider_name)
        if view is None:
            pytest.skip(f"EVAL_LLM_PROVIDER {provider_name!r} not configured")
        return llm_from_provider(model_name=model, llm_provider=view)
    return get_default_llm()
