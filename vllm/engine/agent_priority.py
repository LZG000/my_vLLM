"""Agent-aware priority scoring for vLLM scheduler.

Phase 2: Compute per-request scores to order the waiting queue.
Higher score = schedule first.
"""
from vllm.engine.agent_state import AgentState


def compute_agent_priority_score(
    agent_state: str,
    agent_priority: int = 0,
    max_tokens: int = 4096,
    waiting_steps: int = 0,
) -> float:
    """Compute a priority score for scheduler ordering.

    Components:
        S_state    : +10 if TOOL_RETURNED (multi-round agent continuity)
        S_burst    : 5.0 / max(max_tokens, 1)   (short request ≈ SRPT proxy)
        S_patience : min(waiting_steps / 50, 5.0) (anti-starvation)
        S_agent    : agent_priority / 10          (agent-level static priority)

    Returns:
        float score, higher = more urgent.
    """
    # State bonus: multi-round agents should not reset to back of queue
    state_bonus = 10.0 if agent_state == AgentState.TOOL_RETURNED.value else 0.0

    # Burst proxy: short requests (small max_tokens) get slight priority
    burst = 5.0 / max(max_tokens, 1)

    # Patience: every 50 scheduling steps in waiting, gain 1 point
    patience = min(float(waiting_steps) / 50.0, 5.0)

    # Agent-level priority: scale 0-100 → 0-10 points
    agent = float(agent_priority) / 10.0

    return state_bonus + burst + patience + agent
