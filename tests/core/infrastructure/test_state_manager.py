"""Unit tests for core.infrastructure.state_manager – middleware lifecycle state management."""
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest

from core.infrastructure.state_manager import StateManager
from core.tools.errors import MiddlewareError, MiddlewareErrorCode
from core.types.enums import MiddlewareState


@pytest.fixture(autouse=True)
def clean_state_manager():
    """Ensure every test starts and ends with a fresh singleton state."""
    StateManager.reset_instance()
    yield
    StateManager.reset_instance()


class TestStateManagerSingleton:
    """Verification of the Singleton design pattern and initialization semantics."""

    def test_singleton_returns_same_instance(self):
        """Subsequent calls to StateManager() return the identical instance reference."""
        mgr1 = StateManager()
        mgr2 = StateManager()
        assert mgr1 is mgr2

    def test_default_initial_state_is_idle(self):
        """Newly created StateManager defaults to MiddlewareState.IDLE."""
        mgr = StateManager()
        assert mgr.state == MiddlewareState.IDLE

    def test_custom_initial_state_on_first_creation(self):
        """Specifying initial_state on initial allocation sets that state."""
        mgr = StateManager(initial_state=MiddlewareState.STARTING)
        assert mgr.state == MiddlewareState.STARTING

    def test_subsequent_calls_ignore_initial_state_when_already_initialized(self):
        """Passing initial_state to subsequent calls does not overwrite existing state."""
        mgr1 = StateManager(initial_state=MiddlewareState.IDLE)
        mgr2 = StateManager(initial_state=MiddlewareState.GENERATING)
        assert mgr1 is mgr2
        assert mgr2.state == MiddlewareState.IDLE

    def test_reset_instance_allows_fresh_reallocation(self):
        """reset_instance clears singleton reference and allows initialization with new state."""
        mgr1 = StateManager(initial_state=MiddlewareState.STARTING)
        assert mgr1.state == MiddlewareState.STARTING

        StateManager.reset_instance()
        assert StateManager._instance is None
        assert StateManager._initialized is False

        mgr2 = StateManager(initial_state=MiddlewareState.GENERATING)
        assert mgr2 is not mgr1
        assert mgr2.state == MiddlewareState.GENERATING

    def test_thread_safe_singleton_instantiation(self):
        """Concurrent instantiation across multiple threads returns the identical instance."""
        instances = []

        def worker():
            instances.append(StateManager())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        first = instances[0]
        assert all(inst is first for inst in instances)


class TestStateManagerTransitions:
    """Verification of state transitions, idempotency, and state queries."""

    @pytest.mark.parametrize(
        "target_state",
        [
            MiddlewareState.IDLE,
            MiddlewareState.STARTING,
            MiddlewareState.SETTING_CONTEXT,
            MiddlewareState.GENERATING,
        ],
    )
    def test_transition_to_valid_states(self, target_state):
        """StateManager can transition to any valid MiddlewareState member."""
        mgr = StateManager()
        mgr.transition_to(target_state)
        assert mgr.state == target_state

    def test_sequential_state_lifecycle(self):
        """Simulates typical middleware lifecycle transitions."""
        mgr = StateManager(initial_state=MiddlewareState.STARTING)
        assert mgr.state == MiddlewareState.STARTING

        mgr.transition_to(MiddlewareState.IDLE)
        assert mgr.state == MiddlewareState.IDLE

        mgr.transition_to(MiddlewareState.SETTING_CONTEXT)
        assert mgr.state == MiddlewareState.SETTING_CONTEXT

        mgr.transition_to(MiddlewareState.GENERATING)
        assert mgr.state == MiddlewareState.GENERATING

        mgr.transition_to(MiddlewareState.IDLE)
        assert mgr.state == MiddlewareState.IDLE

    def test_transition_to_same_state_is_noop(self):
        """Transitioning to the currently active state is a safe no-op."""
        mgr = StateManager(initial_state=MiddlewareState.IDLE)
        mgr.transition_to(MiddlewareState.IDLE)
        assert mgr.state == MiddlewareState.IDLE

        mgr.transition_to(MiddlewareState.GENERATING)
        mgr.transition_to(MiddlewareState.GENERATING)
        assert mgr.state == MiddlewareState.GENERATING

    def test_concurrent_state_transitions(self):
        """Concurrent state transitions from multiple threads maintain internal lock integrity."""
        mgr = StateManager()
        states = [
            MiddlewareState.IDLE,
            MiddlewareState.STARTING,
            MiddlewareState.SETTING_CONTEXT,
            MiddlewareState.GENERATING,
        ]

        def transition_cycle(thread_id: int):
            for i in range(25):
                target = states[(thread_id + i) % len(states)]
                mgr.transition_to(target)
                _ = mgr.state

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(transition_cycle, i) for i in range(8)]
            for f in futures:
                f.result()

        assert isinstance(mgr.state, MiddlewareState)


class TestStateManagerIsIn:
    """Verification of is_in helper method."""

    def test_is_in_returns_true_when_state_matches(self):
        """is_in returns True when current state matches parameter."""
        mgr = StateManager(initial_state=MiddlewareState.IDLE)
        assert mgr.is_in(MiddlewareState.IDLE) is True

    def test_is_in_returns_false_when_state_differs(self):
        """is_in returns False when current state differs from parameter."""
        mgr = StateManager(initial_state=MiddlewareState.IDLE)
        assert mgr.is_in(MiddlewareState.GENERATING) is False
        assert mgr.is_in(MiddlewareState.STARTING) is False
        assert mgr.is_in(MiddlewareState.SETTING_CONTEXT) is False

    def test_is_in_reflects_transitions_dynamically(self):
        """is_in updates dynamically after transition_to is called."""
        mgr = StateManager(initial_state=MiddlewareState.IDLE)
        assert mgr.is_in(MiddlewareState.IDLE) is True
        assert mgr.is_in(MiddlewareState.SETTING_CONTEXT) is False

        mgr.transition_to(MiddlewareState.SETTING_CONTEXT)
        assert mgr.is_in(MiddlewareState.IDLE) is False
        assert mgr.is_in(MiddlewareState.SETTING_CONTEXT) is True

    def test_is_in_with_non_enum_or_unexpected_type(self):
        """is_in returns False when compared against non-MiddlewareState objects."""
        mgr = StateManager(initial_state=MiddlewareState.IDLE)
        assert mgr.is_in(None) is False
        assert mgr.is_in(123) is False


class TestStateManagerRequireIdle:
    """Verification of require_idle context manager and error handling."""

    def test_require_idle_succeeds_when_state_is_idle(self):
        """require_idle executes enclosed block without error when state is IDLE."""
        mgr = StateManager(initial_state=MiddlewareState.IDLE)
        block_executed = False

        with mgr.require_idle():
            block_executed = True

        assert block_executed is True

    def test_require_idle_releases_lock_after_success(self):
        """Lock acquired during require_idle is released upon exiting block."""
        mgr = StateManager(initial_state=MiddlewareState.IDLE)

        with mgr.require_idle():
            pass

        # Transitioning requires acquiring _lock; verifies lock was not left locked
        mgr.transition_to(MiddlewareState.GENERATING)
        assert mgr.state == MiddlewareState.GENERATING

    def test_require_idle_releases_lock_on_internal_exception(self):
        """Lock is released even if an exception is raised within the context block."""
        mgr = StateManager(initial_state=MiddlewareState.IDLE)

        with pytest.raises(RuntimeError, match="Internal error"):
            with mgr.require_idle():
                raise RuntimeError("Internal error")

        # Lock must be released so next transition succeeds without deadlock
        mgr.transition_to(MiddlewareState.STARTING)
        assert mgr.state == MiddlewareState.STARTING

    @pytest.mark.parametrize(
        "non_idle_state",
        [
            MiddlewareState.STARTING,
            MiddlewareState.SETTING_CONTEXT,
            MiddlewareState.GENERATING,
        ],
    )
    def test_require_idle_raises_middleware_error_when_not_idle(self, non_idle_state):
        """require_idle raises MiddlewareError with GENERATING code when state is not IDLE."""
        mgr = StateManager(initial_state=non_idle_state)
        block_executed = False

        with pytest.raises(MiddlewareError) as exc_info:
            with mgr.require_idle():
                block_executed = True

        assert block_executed is False
        error = exc_info.value
        assert error.code == MiddlewareErrorCode.GENERATING
        expected_msg = f"Language cannot be changed while the middleware is '{non_idle_state.value}'."
        assert error.errors == [expected_msg]
        assert expected_msg in str(error)

    def test_require_idle_does_not_modify_current_state(self):
        """require_idle does not mutate current state on either success or failure."""
        mgr = StateManager(initial_state=MiddlewareState.IDLE)
        with mgr.require_idle():
            pass
        assert mgr.state == MiddlewareState.IDLE

        mgr.transition_to(MiddlewareState.GENERATING)
        with pytest.raises(MiddlewareError):
            with mgr.require_idle():
                pass
        assert mgr.state == MiddlewareState.GENERATING
