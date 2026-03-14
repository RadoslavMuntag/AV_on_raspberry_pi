from __future__ import annotations

import time

from backend.pipeline.config import PipelineConfig
from backend.contracts import BehaviorState, PlannerDecision, WorldState

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

class BehaviorPlanner:
    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg: PipelineConfig = cfg or PipelineConfig()
        self.current_state: BehaviorState = BehaviorState.IDLE

    def step(self, world: WorldState, requested_mode: BehaviorState, heartbeat_ok: bool) -> PlannerDecision:
        now = time.monotonic()

        if not heartbeat_ok or world.stale:
            self.current_state = BehaviorState.SAFE_STOP
            return PlannerDecision(
                ts=now,
                state=self.current_state,
                reason="heartbeat_timeout_or_stale",
                desired_speed=0.0,
                desired_turn=0.0,
                safe_stop=True,
            )

        if requested_mode == BehaviorState.MANUAL:
            self.current_state = BehaviorState.MANUAL
            return PlannerDecision(now, self.current_state, "manual_mode", 0.0, 0.0)

        if requested_mode == BehaviorState.LINE_FOLLOW:
            if world.obstacle_ahead:
                self.current_state = BehaviorState.OBSTACLE_AVOID
                return PlannerDecision(now, self.current_state, "obstacle_detected", 0.2, self.cfg.avoid_turn)
            self.current_state = BehaviorState.LINE_FOLLOW
            if not world.lane_detected:
                return PlannerDecision(now, self.current_state, "line_lost", 0.0, 0.0)
            return PlannerDecision(now, self.current_state, "line_follow_nominal", self.cfg.cruise_speed, 0.0)

        if requested_mode == BehaviorState.OBSTACLE_AVOID:
            self.current_state = BehaviorState.OBSTACLE_AVOID
            turn = self.cfg.avoid_turn if world.obstacle_ahead else 0.0
            speed = 0.2 if world.obstacle_ahead else self.cfg.cruise_speed
            return PlannerDecision(now, self.current_state, "obstacle_avoid_mode", speed, turn)

        self.current_state = BehaviorState.IDLE
        return PlannerDecision(now, self.current_state, "idle_mode", 0.0, 0.0)