from __future__ import annotations

import math
import time

from backend.pipeline.config import PipelineConfig
from backend.contracts import BehaviorState, ObstacleAvoidPhase, PlannerDecision, WorldState, SensorType

class BehaviorPlanner:
    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg: PipelineConfig = cfg or PipelineConfig()
        self.current_state: BehaviorState = BehaviorState.IDLE
        self._avoid_phase: ObstacleAvoidPhase | None = None
        self._avoid_start_left: float | None = None
        self._avoid_start_right: float | None = None

    def _reset_avoid_sequence(self) -> None:
        self._avoid_phase = None
        self._avoid_start_left = None
        self._avoid_start_right = None

    def _start_avoid_phase(self, phase: ObstacleAvoidPhase, world: WorldState) -> None:
        self._avoid_phase = phase
        self._avoid_start_left = world.left_distance
        self._avoid_start_right = world.right_distance

    def _advance_avoid_sequence(self, world: WorldState) -> None:
        if self._avoid_phase == ObstacleAvoidPhase.TURN_OUT:
            self._start_avoid_phase(ObstacleAvoidPhase.DRIVE_FORWARD_1, world)
        elif self._avoid_phase == ObstacleAvoidPhase.DRIVE_FORWARD_1:
            self._start_avoid_phase(ObstacleAvoidPhase.TURN_BACK_1, world)
        elif self._avoid_phase == ObstacleAvoidPhase.TURN_BACK_1:
            self._start_avoid_phase(ObstacleAvoidPhase.DRIVE_FORWARD_2, world)
        elif self._avoid_phase == ObstacleAvoidPhase.DRIVE_FORWARD_2:
            self._start_avoid_phase(ObstacleAvoidPhase.TURN_BACK_2, world)
        elif self._avoid_phase == ObstacleAvoidPhase.TURN_BACK_2:
            self._start_avoid_phase(ObstacleAvoidPhase.REACQUIRE_FORWARD, world)
        else:
            self._reset_avoid_sequence()

    def _encoder_feedback_ok(self, world: WorldState) -> bool:
        return (
            world.sensor_health.get(SensorType.LEFT_ENCODER, False)
            and world.sensor_health.get(SensorType.RIGHT_ENCODER, False)
        )

    def _phase_progress_cm(self, world: WorldState) -> float | None:
        if self._avoid_start_left is None or self._avoid_start_right is None:
            return None

        left_delta = abs(world.left_distance - self._avoid_start_left)
        right_delta = abs(world.right_distance - self._avoid_start_right)
        return (left_delta + right_delta) / 2.0

    def _turn_direction(self, phase: ObstacleAvoidPhase) -> float:
        if phase in (ObstacleAvoidPhase.TURN_OUT):
            return 1.0
        return -1.0

    def _turn_target_cm(self) -> float:
        return self.cfg.wheel_track * math.pi * 0.25 * self.cfg.obstacle_turn_distance_factor

    def _build_decision(
        self,
        now: float,
        state: BehaviorState,
        reason: str,
        desired_speed: float,
        desired_turn: float,
        *,
        avoid_phase: ObstacleAvoidPhase | None = None,
        avoid_progress_cm: float | None = None,
        avoid_target_cm: float | None = None,
        safe_stop: bool = False,
    ) -> PlannerDecision:
        return PlannerDecision(
            ts=now,
            state=state,
            reason=reason,
            desired_speed=desired_speed,
            desired_turn=desired_turn,
            safe_stop=safe_stop,
            avoid_phase=avoid_phase,
            avoid_progress_cm=avoid_progress_cm,
            avoid_target_cm=avoid_target_cm,
        )


    def _decision_safe_stop(self, now: float, reason: str) -> PlannerDecision:
        self._reset_avoid_sequence()
        self.current_state = BehaviorState.SAFE_STOP
        return self._build_decision(
            now,
            self.current_state,
            reason,
            0.0,
            0.0,
            safe_stop=True,
        )

    def _decision_manual(self, now: float) -> PlannerDecision:
        self._reset_avoid_sequence()
        self.current_state = BehaviorState.MANUAL
        return self._build_decision(now, self.current_state, "manual_mode", 0.0, 0.0)

    def _decision_idle(self, now: float) -> PlannerDecision:
        self._reset_avoid_sequence()
        self.current_state = BehaviorState.IDLE
        return self._build_decision(now, self.current_state, "idle_mode", 0.0, 0.0)

    def _ensure_avoid_sequence_started(self, now: float, world: WorldState, requested_mode: BehaviorState) -> PlannerDecision | None:
        if self._avoid_phase is not None:
            return None

        if requested_mode == BehaviorState.LINE_FOLLOW and world.obstacle_ahead:
            self._start_avoid_phase(ObstacleAvoidPhase.TURN_OUT, world)
            return None

        if requested_mode == BehaviorState.OBSTACLE_AVOID:
            if world.obstacle_ahead:
                self._start_avoid_phase(ObstacleAvoidPhase.TURN_OUT, world)
                return None

            self.current_state = BehaviorState.OBSTACLE_AVOID
            return self._build_decision(
                now,
                self.current_state,
                "waiting_for_obstacle",
                0.0,
                0.0,
            )

        return None

    def _handle_turn_phase(self, now: float, world: WorldState, phase: ObstacleAvoidPhase, reason: str) -> PlannerDecision:
        target_cm = self._turn_target_cm()
        progress_cm = self._phase_progress_cm(world)
        if progress_cm is None:
            return self._decision_safe_stop(now, "avoidance_state_lost")

        if progress_cm >= target_cm:
            self._advance_avoid_sequence(world)
            return self._build_decision(
                now,
                BehaviorState.OBSTACLE_AVOID,
                f"{reason}_complete",
                0.0,
                0.0,
                avoid_phase=self._avoid_phase,
                avoid_progress_cm=progress_cm,
                avoid_target_cm=target_cm,
            )

        turn_gain = self.cfg.obstacle_turn_speed * self._turn_direction(phase)
        if progress_cm >= target_cm * self.cfg.obstacle_slowdown_ratio:
            turn_gain = self.cfg.obstacle_turn_slow_speed * self._turn_direction(phase)

        self.current_state = BehaviorState.OBSTACLE_AVOID
        return self._build_decision(
            now,
            self.current_state,
            reason,
            0.0,
            turn_gain,
            avoid_phase=self._avoid_phase,
            avoid_progress_cm=progress_cm,
            avoid_target_cm=target_cm,
        )

    def _handle_drive_forward_phase(self, now: float, world: WorldState, target_cm: float, reason: str, speed: float) -> PlannerDecision:
        progress_cm = self._phase_progress_cm(world)
        if progress_cm is None:
            return self._decision_safe_stop(now, "avoidance_state_lost")

        if progress_cm >= target_cm:
            self._advance_avoid_sequence(world)
            return self._build_decision(
                now,
                BehaviorState.OBSTACLE_AVOID,
                f"{reason}_complete",
                0.0,
                0.0,
                avoid_phase=self._avoid_phase,
                avoid_progress_cm=progress_cm,
                avoid_target_cm=target_cm,
            )

        if progress_cm >= target_cm * self.cfg.obstacle_slowdown_ratio:
            speed = self.cfg.obstacle_forward_slow_speed

        self.current_state = BehaviorState.OBSTACLE_AVOID
        return self._build_decision(
            now,
            self.current_state,
            reason,
            speed,
            0.0,
            avoid_phase=self._avoid_phase,
            avoid_progress_cm=progress_cm,
            avoid_target_cm=target_cm,
        )

    def _handle_reacquire_phase(self, now: float, world: WorldState) -> PlannerDecision:
        progress_cm = self._phase_progress_cm(world)
        if progress_cm is None:
            return self._decision_safe_stop(now, "avoidance_state_lost")

        target_cm = float(self.cfg.obstacle_reacquire_min_forward_cm)
        if progress_cm >= target_cm and world.lane_detected:
            self._reset_avoid_sequence()
            return self._line_follow_nominal(now, world)

        speed = self.cfg.obstacle_reacquire_speed
        if progress_cm >= target_cm * self.cfg.obstacle_slowdown_ratio:
            speed = self.cfg.obstacle_reacquire_slow_speed

        self.current_state = BehaviorState.OBSTACLE_AVOID
        return self._build_decision(
            now,
            self.current_state,
            "reacquire_forward",
            speed,
            0.0,
            avoid_phase=self._avoid_phase,
            avoid_progress_cm=progress_cm,
            avoid_target_cm=target_cm,
        )

    def _handle_turn_back_phase(self, now: float, world: WorldState, phase: ObstacleAvoidPhase, reason: str) -> PlannerDecision:
        target_cm = self._turn_target_cm()
        progress_cm = self._phase_progress_cm(world)
        if progress_cm is None:
            return self._decision_safe_stop(now, "avoidance_state_lost")

        if progress_cm >= target_cm:
            self._advance_avoid_sequence(world)
            return self._build_decision(
                now,
                BehaviorState.OBSTACLE_AVOID,
                f"{reason}_complete",
                0.0,
                0.0,
                avoid_phase=self._avoid_phase,
                avoid_progress_cm=progress_cm,
                avoid_target_cm=target_cm,
            )

        turn_gain = self.cfg.obstacle_turn_speed * self._turn_direction(phase)
        if progress_cm >= target_cm * self.cfg.obstacle_slowdown_ratio:
            turn_gain = self.cfg.obstacle_turn_slow_speed * self._turn_direction(phase)

        self.current_state = BehaviorState.OBSTACLE_AVOID
        return self._build_decision(
            now,
            self.current_state,
            reason,
            0.0,
            turn_gain,
            avoid_phase=self._avoid_phase,
            avoid_progress_cm=progress_cm,
            avoid_target_cm=target_cm,
        )

    def _handle_active_avoid_sequence(
        self,
        now: float,
        world: WorldState,
    ) -> PlannerDecision | None:
        if self._avoid_phase is None:
            return None

        if not self._encoder_feedback_ok(world):
            return self._decision_safe_stop(now, "encoder_feedback_unavailable")

        if self._phase_progress_cm(world) is None:
            return self._decision_safe_stop(now, "avoidance_state_lost")

        if self._avoid_phase == ObstacleAvoidPhase.TURN_OUT:
            return self._handle_turn_phase(now, world, ObstacleAvoidPhase.TURN_OUT, "turn_out")
        if self._avoid_phase == ObstacleAvoidPhase.DRIVE_FORWARD_1:
            return self._handle_drive_forward_phase(now, world, float(self.cfg.obstacle_forward_distance_cm), "drive_forward_1", self.cfg.obstacle_forward_speed)
        if self._avoid_phase == ObstacleAvoidPhase.TURN_BACK_1:
            return self._handle_turn_back_phase(now, world, ObstacleAvoidPhase.TURN_BACK_1, "turn_back_1")
        if self._avoid_phase == ObstacleAvoidPhase.DRIVE_FORWARD_2:
            return self._handle_drive_forward_phase(now, world, float(self.cfg.obstacle_forward_distance_2_cm), "drive_forward_2", self.cfg.obstacle_forward_speed)
        if self._avoid_phase == ObstacleAvoidPhase.TURN_BACK_2:
            return self._handle_turn_back_phase(now, world, ObstacleAvoidPhase.TURN_BACK_2, "turn_back_2")
        if self._avoid_phase == ObstacleAvoidPhase.REACQUIRE_FORWARD:
            return self._handle_reacquire_phase(now, world)

        return None

    def _line_follow_nominal(self, now: float, world: WorldState) -> PlannerDecision:
        self.current_state = BehaviorState.LINE_FOLLOW
        if not world.lane_detected:
            return self._build_decision(now, self.current_state, "line_lost", 0.0, 0.0)

        speed = self.cfg.cruise_speed * max(
            self.cfg.line_min_speed_factor,
            1 - self.cfg.line_curvature_speed_gain * abs(world.line_curvature),
        )
        turn = self.cfg.line_kp * world.line_offset + self.cfg.line_angle_kp * (-world.line_angle)
        return self._build_decision(now, self.current_state, "line_follow_nominal", speed, turn)

    def _line_follow_with_obstacle(self, now: float, world: WorldState) -> PlannerDecision:
        self._start_avoid_phase(ObstacleAvoidPhase.TURN_OUT, world)
        self.current_state = BehaviorState.OBSTACLE_AVOID
        return self._build_decision(
            now,
            self.current_state,
            "obstacle_detected",
            0.0,
            self.cfg.obstacle_turn_speed,
            avoid_phase=self._avoid_phase,
            avoid_progress_cm=0.0,
            avoid_target_cm=self._turn_target_cm(),
        )

    def _handle_line_follow_mode(self, now: float, world: WorldState) -> PlannerDecision:
        if world.obstacle_ahead:
            return self._line_follow_with_obstacle(now, world)
        return self._line_follow_nominal(now, world)

    def _handle_obstacle_avoid_mode(self, now: float, world: WorldState) -> PlannerDecision:
        self.current_state = BehaviorState.OBSTACLE_AVOID
        if world.obstacle_ahead and self._avoid_phase is None:
            self._start_avoid_phase(ObstacleAvoidPhase.TURN_OUT, world)
        if self._avoid_phase is None:
            return self._build_decision(now, self.current_state, "obstacle_avoid_mode", 0.0, 0.0)

        return self._build_decision(
            now,
            self.current_state,
            "obstacle_avoid_mode",
            0.0,
            self.cfg.obstacle_turn_speed,
            avoid_phase=self._avoid_phase,
            avoid_progress_cm=self._phase_progress_cm(world),
            avoid_target_cm=self._turn_target_cm(),
        )

    def step(self, world: WorldState, requested_mode: BehaviorState, heartbeat_ok: bool) -> PlannerDecision:
        if self.cfg.DEBUG:
            print("DEBUG: Planner step - world:", world, "requested_mode:", requested_mode, "heartbeat_ok:", heartbeat_ok)
        now = time.monotonic()

        if not heartbeat_ok or world.stale:
            return self._decision_safe_stop(now, "heartbeat_timeout_or_stale")

        if requested_mode == BehaviorState.MANUAL:
            return self._decision_manual(now)

        if requested_mode not in (BehaviorState.LINE_FOLLOW, BehaviorState.OBSTACLE_AVOID):
            return self._decision_idle(now)

        start_decision = self._ensure_avoid_sequence_started(now, world, requested_mode)
        if start_decision is not None:
            return start_decision

        active_avoid = self._handle_active_avoid_sequence(now, world)
        if active_avoid is not None:
            return active_avoid

        if requested_mode == BehaviorState.LINE_FOLLOW:
            return self._handle_line_follow_mode(now, world)

        if requested_mode == BehaviorState.OBSTACLE_AVOID:
            return self._handle_obstacle_avoid_mode(now, world)

        return self._decision_idle(now)