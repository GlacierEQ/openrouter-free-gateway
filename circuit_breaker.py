"""
Circuit Breaker & Model Health Manager for OpenRouter Free Gateway
=====================================================================

Implements circuit breaker pattern, health tracking, and intelligent
model routing for production-grade reliability.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CircuitBreaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for individual models to prevent cascading failures."""

    failure_threshold: int = 3
    timeout_seconds: int = 60
    reset_timeout_seconds: int = 300
    half_open_attempts: int = 2

    # Runtime state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _half_open_successes: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.timeout_seconds:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_successes = 0
                        logger.info(f"Circuit entering HALF_OPEN state")
                        return True
                return False

            # HALF_OPEN
            return self._half_open_successes < self.half_open_attempts

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._half_open_successes = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_attempts:
                    logger.warning(f"Circuit breaking to OPEN after half-open failures")
                    self._state = CircuitState.OPEN
                    self._half_open_successes = 0
            elif self._failure_count >= self.failure_threshold:
                logger.warning(f"Circuit opening after {self._failure_count} failures")
                self._state = CircuitState.OPEN

    def to_dict(self) -> Dict[str, Any]:
        """Serialize circuit breaker state."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
            "half_open_successes": self._half_open_successes,
        }


@dataclass
class ModelHealth:
    """Tracks health and performance metrics for a model."""

    model_id: str
    success_count: int = 0
    failure_count: int = 0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    avg_latency_ms: float = 0.0
    total_requests: int = 0
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    # Metadata from API
    name: str = ""
    context_length: int = 32768
    description: str = ""
    capabilities: List[str] = field(default_factory=list)

    # Status flags
    is_healthy: bool = True
    last_check_time: Optional[float] = None
    reason: str = ""

    def record_success(self, latency_ms: float = 0) -> None:
        """Record a successful request."""
        self.success_count += 1
        self.last_success_time = time.time()
        self.total_requests += 1

        # Update rolling average latency
        if self.total_requests == 1:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (
                self.avg_latency_ms * (self.total_requests - 1) + latency_ms
            ) / self.total_requests

        self.circuit_breaker.record_success()
        self.is_healthy = True

    def record_failure(self, error: str = "") -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.total_requests += 1
        self.reason = error

        self.circuit_breaker.record_failure()

        # Mark unhealthy after 3 consecutive failures
        if self.failure_count >= 3:
            self.is_healthy = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model health state."""
        return {
            "model_id": self.model_id,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_requests": self.total_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "is_healthy": self.is_healthy,
            "last_success_time": self.last_success_time,
            "last_failure_time": self.last_failure_time,
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "health_score": self.calculate_health_score(),
        }

    def calculate_health_score(self) -> float:
        """Calculate a health score between 0 and 1."""
        if self.total_requests == 0:
            return 1.0

        success_rate = self.success_count / self.total_requests

        # Penalty for high latency
        latency_penalty = min(1.0, self.avg_latency_ms / 30000)  # Normalize by 30s

        # Penalty for circuit breaker being open
        circuit_penalty = 0.5 if self.circuit_breaker._state == CircuitState.OPEN else 0

        return max(0, (success_rate * 0.7) + (latency_penalty * -0.2) - circuit_penalty)


class ModelHealthManager:
    """Manages health tracking for all models with thread-safe operations."""

    def __init__(self, config_path: Optional[str] = None):
        self.models: Dict[str, ModelHealth] = {}
        self._lock = Lock()
        self._config_path = (
            Path(config_path) if config_path else Path("model_health.json")
        )
        self._load_state()

    def register_model(self, model_info: Dict[str, Any]) -> ModelHealth:
        """Register a new model for health tracking."""
        with self._lock:
            model_id = model_info["id"]
            if model_id not in self.models:
                health = ModelHealth(
                    model_id=model_id,
                    name=model_info.get("name", model_id),
                    context_length=model_info.get("context_length", 32768),
                    description=model_info.get("description", ""),
                    capabilities=self._infer_capabilities(model_info),
                )
                self.models[model_id] = health
            return self.models[model_id]

    def _infer_capabilities(self, model_info: Dict[str, Any]) -> List[str]:
        """Infer model capabilities from description and name."""
        desc = model_info.get("description", "").lower()
        model_id = model_info.get("id", "")
        caps = []

        if any(word in desc for word in ["coding", "code", "programming", "agent"]):
            caps.append("coding")
        if any(word in desc for word in ["reasoning", "think", "logic"]):
            caps.append("reasoning")
        if any(word in desc for word in ["multimodal", "image", "video", "omni"]):
            caps.append("multimodal")
        if "inkling" in model_id.lower() or "m3" in model_id.lower():
            caps.extend(["coding", "reasoning", "agentic", "multimodal"])
        if "nemotron" in model_id.lower():
            caps.extend(["coding", "reasoning"])
        if "rag" in desc or "knowledge" in desc:
            caps.append("knowledge")

        return list(set(caps))

    def record_success(self, model_id: str, latency_ms: float = 0) -> None:
        """Record a successful request for a model."""
        with self._lock:
            if model_id in self.models:
                self.models[model_id].record_success(latency_ms)
                logger.debug(
                    f"Success recorded for {model_id}, latency: {latency_ms}ms"
                )

    def record_failure(self, model_id: str, error: str = "") -> None:
        """Record a failed request for a model."""
        with self._lock:
            if model_id in self.models:
                self.models[model_id].record_failure(error)
                logger.warning(f"Failure recorded for {model_id}: {error}")

    def can_execute(self, model_id: str) -> bool:
        """Check if a model is available for execution (passes circuit breaker)."""
        with self._lock:
            if model_id in self.models:
                return self.models[model_id].circuit_breaker.can_execute()
        return True

    def get_healthy_models(
        self, required_capability: Optional[str] = None
    ) -> List[ModelHealth]:
        """Get list of healthy models, optionally filtered by capability."""
        with self._lock:
            healthy = []
            for health in self.models.values():
                if health.is_healthy and health.circuit_breaker.can_execute():
                    if (
                        required_capability is None
                        or required_capability in health.capabilities
                    ):
                        healthy.append(health)

            # Sort by health score (highest first) and context length
            healthy.sort(
                key=lambda h: (h.calculate_health_score(), h.context_length),
                reverse=True,
            )
            return healthy

    def get_model(self, model_id: str) -> Optional[ModelHealth]:
        """Get health status for a specific model."""
        with self._lock:
            return self.models.get(model_id)

    def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """Update model metadata (context length, description, etc.)."""
        with self._lock:
            if model_id in self.models:
                health = self.models[model_id]
                health.context_length = metadata.get(
                    "context_length", health.context_length
                )
                health.description = metadata.get("description", health.description)
                health.name = metadata.get("name", health.name)

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of all model health states."""
        with self._lock:
            return {
                "total_models": len(self.models),
                "healthy_models": sum(1 for h in self.models.values() if h.is_healthy),
                "total_requests": sum(h.total_requests for h in self.models.values()),
                "overall_success_rate": self._calculate_overall_success_rate(),
                "models": {mid: h.to_dict() for mid, h in self.models.items()},
            }

    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all models."""
        total_success = sum(h.success_count for h in self.models.values())
        total_requests = sum(h.total_requests for h in self.models.values())
        return total_success / total_requests if total_requests > 0 else 1.0

    def _load_state(self) -> None:
        """Load persisted health state from disk."""
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text())
                for model_data in data.get("models", {}).values():
                    health = ModelHealth(
                        model_id=model_data["model_id"],
                        success_count=model_data.get("success_count", 0),
                        failure_count=model_data.get("failure_count", 0),
                        total_requests=model_data.get("total_requests", 0),
                        avg_latency_ms=model_data.get("avg_latency_ms", 0),
                        is_healthy=model_data.get("is_healthy", True),
                        name=model_data.get("name", ""),
                        context_length=model_data.get("context_length", 32768),
                        description=model_data.get("description", ""),
                    )
                    self.models[model_data["model_id"]] = health
                logger.info(f"Loaded health state for {len(self.models)} models")
            except Exception as e:
                logger.warning(f"Failed to load health state: {e}")

    def save_state(self) -> None:
        """Persist health state to disk."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "models": {mid: h.to_dict() for mid, h in self.models.items()},
            }
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save health state: {e}")


# Global health manager instance
_health_manager: Optional[ModelHealthManager] = None
_health_lock = Lock()


def get_health_manager(config_path: Optional[str] = None) -> ModelHealthManager:
    """Get or create the global health manager instance."""
    global _health_manager
    if _health_manager is None:
        with _health_lock:
            if _health_manager is None:
                _health_manager = ModelHealthManager(config_path)
    return _health_manager


# Model ranking based on current evaluation
MODEL_RANKINGS = {
    "thinkingmachines/inkling:free": {
        "priority": 1,
        "weight": 40,
        "score": 0.95,
        "capabilities": ["coding", "reasoning", "agentic", "multimodal"],
    },
    "thinkingmachines/inkling-small:free": {
        "priority": 2,
        "weight": 25,
        "score": 0.88,
        "capabilities": ["coding", "reasoning", "agentic"],
    },
    "poolside/laguna-s-2.1:free": {
        "priority": 3,
        "weight": 15,
        "score": 0.92,
        "capabilities": ["coding", "agentic"],
    },
    "deepseek/deepseek-r1:free": {
        "priority": 4,
        "weight": 10,
        "score": 0.89,
        "capabilities": ["reasoning", "coding"],
    },
    "minimax/minimax-m3:free": {
        "priority": 5,
        "weight": 5,
        "score": 0.85,
        "capabilities": ["multimodal", "reasoning"],
    },
}
