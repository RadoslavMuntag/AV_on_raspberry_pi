from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from ..contracts import BehaviorState, ManualCommand, PerceptionFrame, WorldState, PlannerDecision, ControlTargets


@dataclass
class RuntimeConfig:
    heartbeat_timeout_sec: float = 1.5
    control_loop_hz: float = 200.0
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
    dualsense_connected: bool = False
    hardware_ready: bool = False
    hardware_error: Optional[str] = None
    fps: Optional[float] = None
    updated_at: float = field(default_factory=lambda: time.time())


class StateStore:
    def __init__(self) -> None:
        # a reentrant lock to allow the same thread to acquire it multiple times if needed.
        self._lock = threading.RLock()
        self.config = RuntimeConfig()
        self.state = VehicleState()

        self._perception_frame: Optional[PerceptionFrame] = None
        self._world_state: Optional[WorldState] = None
        self._planner_decision: Optional[PlannerDecision] = None
        self._control_targets: Optional[ControlTargets] = None

        self._manual_command: Optional[ManualCommand] = None
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": asdict(self.state),
                "config": asdict(self.config),
            }
        
    def pipeline_snapshot(self, frame: str = "all") -> Dict[str, Any]:
        with self._lock:
            all_frames = {
                "perception": asdict(self._perception_frame) if self._perception_frame else None,
                "world": asdict(self._world_state) if self._world_state else None,
                "planner": asdict(self._planner_decision) if self._planner_decision else None,
                "control": asdict(self._control_targets) if self._control_targets else None,
                "manual": asdict(self._manual_command) if self._manual_command else None,
            }
            if frame == "all":
                return all_frames
            if frame not in all_frames:
                raise ValueError(f"unsupported frame '{frame}'")
            return {frame: all_frames[frame]}
        
    def set_perception_frame(self, frame: Optional[PerceptionFrame]) -> None:
        with self._lock:
            self._perception_frame = frame

    def set_world_state(self, world: Optional[WorldState]) -> None:
        with self._lock:
            self._world_state = world

    def set_planner_decision(self, decision: Optional[PlannerDecision]) -> None:
        with self._lock:
            self._planner_decision = decision

    def set_manual_command(self, command: Optional[ManualCommand]) -> None:
        with self._lock:
            self._manual_command = command

    def set_control_targets(self, targets: Optional[ControlTargets]) -> None:
        with self._lock:
            self._control_targets = targets

    def set_pipeline_snapshot(self, perception: Optional[PerceptionFrame], world: Optional[WorldState], decision: Optional[PlannerDecision], control: Optional[ControlTargets]) -> None:
        with self._lock:
            self._perception_frame = perception
            self._world_state = world
            self._planner_decision = decision
            self._control_targets = control

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
