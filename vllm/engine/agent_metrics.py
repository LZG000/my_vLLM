# SPDX-License-Identifier: Apache-2.0

import threading
from collections import deque
from typing import Deque, Optional

import prometheus_client

from vllm.logger import init_logger

logger = init_logger(__name__)

_lock = threading.Lock()
_model_name = ""
_initialized = False

_requests = None
_execution_seconds = None
_tool_seconds = None
_queue_seconds = None
_wait_seconds = None
_tool_reports = None
_tool_duplicates = None

_seen_lock = threading.Lock()
_seen = set[str]()
_order: Deque[str] = deque()
_MAX_SEEN = 20000


def _init_metrics() -> None:
    global _initialized
    global _requests, _execution_seconds, _tool_seconds
    global _queue_seconds, _wait_seconds, _tool_reports, _tool_duplicates

    if _initialized:
        return

    with _lock:
        if _initialized:
            return

        # IMPORTANT:
        # In vLLM multiprocess mode, PROMETHEUS_MULTIPROC_DIR is set during
        # runtime startup (after module import). So we must create metrics
        # lazily, not at import time, otherwise counters are not wired into
        # multiprocess files and won't appear in /metrics.
        _requests = prometheus_client.Counter(
            "vllm:agent_requests_total",
            "Total finished requests per agent_id.",
            labelnames=["agent_id", "model_name"],
        )
        _execution_seconds = prometheus_client.Counter(
            "vllm:agent_execution_seconds_total",
            "Total end-to-end execution seconds per agent_id.",
            labelnames=["agent_id", "model_name"],
        )
        _tool_seconds = prometheus_client.Counter(
            "vllm:agent_tool_seconds_total",
            "Total tool-call/tool-wait seconds per agent_id.",
            labelnames=["agent_id", "model_name"],
        )
        _queue_seconds = prometheus_client.Counter(
            "vllm:agent_queue_seconds_total",
            "Total queue (waiting-to-first-scheduled) seconds per agent_id.",
            labelnames=["agent_id", "model_name"],
        )
        _wait_seconds = prometheus_client.Counter(
            "vllm:agent_wait_seconds_total",
            "Total wait seconds (queue + tool-wait) per agent_id.",
            labelnames=["agent_id", "model_name"],
        )
        _tool_reports = prometheus_client.Counter(
            "vllm:agent_tool_time_reports_total",
            "Total number of tool_time_ms reports received per agent_id.",
            labelnames=["agent_id", "model_name"],
        )
        _tool_duplicates = prometheus_client.Counter(
            "vllm:agent_tool_time_duplicates_total",
            "Total duplicate tool_session_id reports per agent_id.",
            labelnames=["agent_id", "model_name"],
        )
        _initialized = True
        logger.info("Agent metrics initialized.")


def _labels(agent_id: str) -> dict[str, str]:
    return {"agent_id": str(agent_id), "model_name": _model_name}


def set_model_name(model_name: str) -> None:
    _init_metrics()
    global _model_name
    with _lock:
        _model_name = model_name


def on_request_finished(agent_id: str, execution_seconds: float) -> None:
    _init_metrics()
    labels = _labels(agent_id)
    _requests.labels(**labels).inc(1.0)
    _execution_seconds.labels(**labels).inc(max(0.0, execution_seconds))
    logger.info("Agent metric update: request_finished agent_id=%s execution_seconds=%.6f",
                agent_id, max(0.0, execution_seconds))


def on_tool_time_reported(agent_id: str, tool_seconds: float,
                          duplicated: bool) -> None:
    _init_metrics()
    labels = _labels(agent_id)
    _tool_reports.labels(**labels).inc(1.0)
    if duplicated:
        _tool_duplicates.labels(**labels).inc(1.0)
        logger.info("Agent metric update: tool_report duplicate agent_id=%s", agent_id)
        return
    tool = max(0.0, tool_seconds)
    _tool_seconds.labels(**labels).inc(tool)
    logger.info("Agent metric update: tool_report agent_id=%s tool_seconds=%.6f",
                agent_id, tool)


def on_queue_and_wait_time(agent_id: str, queue_seconds: float,
                           tool_seconds: float) -> None:
    _init_metrics()
    labels = _labels(agent_id)
    queue = max(0.0, queue_seconds)
    tool = max(0.0, tool_seconds)
    _queue_seconds.labels(**labels).inc(queue)
    _wait_seconds.labels(**labels).inc(queue + tool)
    logger.info("Agent metric update: wait agent_id=%s queue_seconds=%.6f tool_seconds=%.6f",
                agent_id, queue, tool)


def should_count_tool_session(session_id: Optional[str]) -> bool:
    if session_id is None:
        return True
    sid = str(session_id)
    with _seen_lock:
        if sid in _seen:
            return False
        _seen.add(sid)
        _order.append(sid)
        while len(_order) > _MAX_SEEN:
            old = _order.popleft()
            _seen.discard(old)
        return True
