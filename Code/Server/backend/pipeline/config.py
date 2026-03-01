from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class PerceptionConfig:
    obstacle_threshold_cm: float = 25.0

@dataclass(slots=True)
class FusionConfig:
    obstacle_threshold_cm: float = 25.0
    max_sensor_age_s: float = 0.25

@dataclass(slots=True)
class PlannerConfig:
    cruise_speed: float = 0.35
    avoid_turn: float = 0.6

@dataclass(slots=True)
class ControlConfig:
    max_pwm: int = 4095
    line_kp: float = 0.7
    min_confidence: float = 0.15




