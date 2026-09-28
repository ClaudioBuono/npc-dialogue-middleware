from __future__ import annotations
import pytest
from evals.bootstrap import configure_environment


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "eval: semantic LLM evaluation (real API calls, slow, non-deterministic). "
        "Excluded by default; run explicitly with `pytest -m eval`.",
    )


@pytest.fixture(scope="session", autouse=True)
def setup_global_settings():
    """Mirrors tests/conftest.py's fixture of the same name, plus
    ModelRegistry population -- see _bootstrap.configure_environment's
    docstring for the full rationale."""
    configure_environment()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    markexpr = config.getoption("-m") or ""
    if "eval" in markexpr:
        # The user explicitly asked for eval-marked tests; don't skip them.
        return

    skip_eval = pytest.mark.skip(reason="semantic eval: run explicitly with `pytest -m eval`")
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip_eval)