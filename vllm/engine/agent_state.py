from enum import Enum
from typing import Dict, Optional
import threading

from vllm.logger import init_logger

logger = init_logger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    IO_BLOCKED = "io_blocked"
    TOOL_CALLED = "tool_called"
    TOOL_RETURNED = "tool_returned"
    DONE = "done"

class AgentStateTracker:
    """Thread-safe singleton tracker for per-request agent FSM states."""
    
    _instance: Optional["AgentStateTracker"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "AgentStateTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._states: Dict[str, AgentState] = {}
                    cls._instance._io_bubble_counts: Dict[str, int] = {}
        return cls._instance
    
    def transition(self, request_id: str, new_state: AgentState) -> None:
        with self._lock:
            old_state = self._states.get(request_id, AgentState.IDLE)
            self._states[request_id] = new_state
            if new_state == AgentState.IO_BLOCKED:
                self._io_bubble_counts[request_id] = \
                    self._io_bubble_counts.get(request_id, 0) + 1
        logger.info("Agent FSM: %s %s -> %s (io_bubbles=%d)",
                    request_id, old_state.value, new_state.value,
                    self._io_bubble_counts.get(request_id, 0))
    
    def get_state(self, request_id: str) -> AgentState:
        with self._lock:
            return self._states.get(request_id, AgentState.IDLE)
    
    def get_io_bubble_count(self, request_id: str) -> int:
        with self._lock:
            return self._io_bubble_counts.get(request_id, 0)
    
    def remove(self, request_id: str) -> None:
        with self._lock:
            self._states.pop(request_id, None)
            self._io_bubble_counts.pop(request_id, None)
