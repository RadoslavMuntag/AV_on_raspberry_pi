from __future__ import annotations

import time
from typing import Optional

from .config import ControlConfig
from ..contracts import BehaviorState, ControlTargets, ManualCommand, PlannerDecision, WorldState

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class DifferentialDriveController:
    def __init__(self, cfg: Optional[ControlConfig] = None) -> None:
        self.cfg = cfg or ControlConfig()

    def _mix(self, speed: float, turn: float) -> tuple[int, int]:
        speed = _clamp(speed, -1.0, 1.0)
        turn = _clamp(turn, -1.0, 1.0)
        left = int((speed - turn) * self.cfg.max_pwm)
        right = int((speed + turn) * self.cfg.max_pwm)
        return left, right

    def step(
        self,
        decision: PlannerDecision,
        world: WorldState,
        manual: ManualCommand,
    ) -> ControlTargets:
        now = time.monotonic()

        if decision.safe_stop or decision.state == BehaviorState.SAFE_STOP:
            return ControlTargets(now, 0, 0, "blink", (255, 0, 0))

        if decision.state == BehaviorState.IDLE:
            return ControlTargets(now, 0, 0, "off", (0, 0, 0))

        if decision.state == BehaviorState.MANUAL:
            if not manual.active:
                return ControlTargets(now, 0, 0, "index", (255, 120, 0))
            left, right = self._mix(manual.throttle, manual.steer)
            return ControlTargets(now, left, right, "index", (0, 120, 255))

        if decision.state == BehaviorState.LINE_FOLLOW:
            if world.lane_detected and world.lateral_confidence >= self.cfg.min_confidence:
                turn = _clamp(self.cfg.line_kp * world.lateral_error, -1.0, 1.0)
                left, right = self._mix(decision.desired_speed, turn)
                return ControlTargets(now, left, right, "index", (0, 255, 0))
            return ControlTargets(now, 0, 0, "blink", (255, 255, 0))

        if decision.state == BehaviorState.OBSTACLE_AVOID:
            left, right = self._mix(decision.desired_speed, decision.desired_turn)
            return ControlTargets(now, left, right, "index", (255, 0, 255))

        return ControlTargets(now, 0, 0, "off", (0, 0, 0))