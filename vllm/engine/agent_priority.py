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

# --- Phase 3: Agent Tool Swap Support (Engine-layer) ---

# Per-agent EMA tool execution time history (seconds)
_DEFAULT_TOOL_TIME: float = 1.0
_agent_tool_ema: dict = {}
_EMA_ALPHA: float = 0.3


def _ema_key(agent_id: str, tool_type: str = "default") -> str:
    return f"{agent_id}:{tool_type}"


def update_agent_tool_time(agent_id: str, tool_duration_s: float,
                           tool_type: str = "default") -> None:
    """Update EMA of tool execution time for an (agent, tool_type) pair."""
    key = _ema_key(agent_id, tool_type)
    if key not in _agent_tool_ema:
        _agent_tool_ema[key] = tool_duration_s
    else:
        _agent_tool_ema[key] = (
            _EMA_ALPHA * tool_duration_s
            + (1.0 - _EMA_ALPHA) * _agent_tool_ema[key]
        )


def estimate_tool_remaining(agent_id: str,
                            tool_type: str = "default") -> float:
    """Estimate remaining tool time for an (agent, tool_type) pair."""
    key = _ema_key(agent_id, tool_type)
    return _agent_tool_ema.get(key, _DEFAULT_TOOL_TIME)


def get_swap_threshold(agent_priority: int,
                       scheduler_config=None) -> float:
    """Get swap threshold (seconds) based on agent priority.
    
    Higher priority = higher threshold = less likely to swap.
    """
    if scheduler_config is not None:
        if agent_priority >= 80:
            return scheduler_config.agent_tool_swap_threshold_vip
        elif agent_priority >= 60:
            return scheduler_config.agent_tool_swap_threshold_high
        elif agent_priority >= 30:
            return scheduler_config.agent_tool_swap_threshold_mid
        else:
            return scheduler_config.agent_tool_swap_threshold_bg
    
    if agent_priority >= 80:
        return 0.5
    elif agent_priority >= 60:
        return 0.5
    elif agent_priority >= 30:
        return 0.5
    else:
        return 0.5


def get_swap_score_boost(scheduler_config=None) -> float:
    """Get priority score boost for tool-returned requests."""
    if scheduler_config is not None:
        return getattr(scheduler_config, 'agent_tool_swap_score_boost', 3.0)
    return 3.0
