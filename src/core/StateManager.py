from threading import Lock
import logging
from typing import Optional

from core.types.enums import MiddlewareState


logger = logging.getLogger(__name__)


class StateManager:
    """Singleton (lazy allocation) that manages the middleware state.

    Usage:
        state_manager = StateManager()
        state_manager.transition_to(MiddlewareState.GENERATING)
        current = state_manager.state
    """

    _instance: Optional["StateManager"] = None
    _instance_lock = Lock()  # Lock protecting singleton instance creation
    _initialized: bool = False  # Flag to prevent __init__ from re-running on every call

    def __new__(cls, initial_state: MiddlewareState = MiddlewareState.IDLE) -> "StateManager":
        # Create the instance only if it doesn't already exist (singleton pattern)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, initial_state: MiddlewareState = MiddlewareState.IDLE) -> None:
        # Avoid re-initializing the state every time StateManager() is called
        if self._initialized:
            return
        self._lock = Lock()  # Separate lock protecting reads/writes of _state
        self._state = initial_state
        self._initialized = True

    @classmethod
    def reset_instance(cls) -> None:
        """Mainly useful in tests, to reset the singleton."""
        # Reset both the instance and the initialized flag,
        # so the next call to StateManager() rebuilds everything from scratch
        cls._instance = None
        cls._initialized = False

    @property
    def state(self) -> MiddlewareState:
        # Thread-safe read of the current state
        with self._lock:
            return self._state

    def transition_to(self, new_state: MiddlewareState) -> None:
        """Updates the middleware state.

        No-op if new_state matches the current state.
        """
        with self._lock:
            previous = self._state
            if previous == new_state:
                # No actual change: exit without logging the transition
                return
            self._state = new_state
