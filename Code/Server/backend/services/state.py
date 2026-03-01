from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from ..contracts import BehaviorState, ManualCommand


@dataclass
class RuntimeConfig:
    heartbeat_timeout_sec: float = 1.5
    control_loop_hz: float = 20.0
    max_motor_speed: int = 2800


@dataclass
class VehicleState:
    mode: BehaviorState = BehaviorState.IDLE
    controller_id: Optional[str] = None
    controller_last_seen: float = 0.0
    e_stop: bool = False
    left_motor: int = 0
    right_motor: int = 0
    ultrasonic_cm: Optional[float] = None
    infrared_value: Optional[int] = None
    camera_streaming: bool = False
    hardware_ready: bool = False
    hardware_error: Optional[str] = None
    updated_at: float = field(default_factory=lambda: time.time())


class StateStore:
    def __init__(self) -> None:
        # a reentrant lock to allow the same thread to acquire it multiple times if needed.
        self._lock = threading.RLock()
        self.config = RuntimeConfig()
        self.state = VehicleState()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": asdict(self.state),
                "config": asdict(self.config),
            }

    def update_state(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            self.state.updated_at = time.time()

    def update_config(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if value is not None and hasattr(self.config, key):
                    setattr(self.config, key, value)

    def is_controller_active(self) -> bool:
        with self._lock:
            return self.state.controller_id is not None

    def should_timeout_controller(self) -> bool:
        with self._lock:
            if self.state.controller_id is None:
                return False
            return (time.time() - self.state.controller_last_seen) > self.config.heartbeat_timeout_sec
