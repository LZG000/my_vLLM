"""Agent-aware priority scoring for vLLM scheduler.

Priority weights can be tuned via environment variables:
  VLLM_PRIORITY_AGENT_DIVISOR   : agent_priority / N (default 20)
  VLLM_PRIORITY_STATE_BONUS     : bonus for TOOL_RETURNED (default 1.5)
  VLLM_PRIORITY_PATIENCE_STEP   : step divisor for patience (default 35)
  VLLM_PRIORITY_PATIENCE_CAP    : max patience score (default 8.0)
  VLLM_PRIORITY_BURST_NUMERATOR : burst = N / max_tokens (default 5.0)
"""
import os
from vllm.engine.agent_state import AgentState

# Configurable weights (read once at import time)
_AGENT_DIVISOR = float(os.environ.get("VLLM_PRIORITY_AGENT_DIVISOR", "20"))
_STATE_BONUS = float(os.environ.get("VLLM_PRIORITY_STATE_BONUS", "1.5"))
_PATIENCE_STEP = float(os.environ.get("VLLM_PRIORITY_PATIENCE_STEP", "35"))
_PATIENCE_CAP = float(os.environ.get("VLLM_PRIORITY_PATIENCE_CAP", "8.0"))
_BURST_NUM = float(os.environ.get("VLLM_PRIORITY_BURST_NUMERATOR", "5.0"))


def compute_agent_priority_score(
    agent_state: str,
    agent_priority: int = 0,
    max_tokens: int = 4096,
    waiting_steps: int = 0,
) -> float:
    """Compute a priority score for scheduler ordering.

    Score = state_bonus + burst + patience + agent

    Configurable via env vars (see module docstring).
    """
    state_bonus = _STATE_BONUS if agent_state == AgentState.TOOL_RETURNED.value else 0.0
    burst = min(_BURST_NUM / max(max_tokens, 1), 2.0)
    patience = min(float(waiting_steps) / _PATIENCE_STEP, _PATIENCE_CAP)
    agent = float(agent_priority) / _AGENT_DIVISOR

    return state_bonus + burst + patience + agent
