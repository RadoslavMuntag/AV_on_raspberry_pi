from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class PipelineConfig:
    obstacle_threshold_cm: float = 25.0
    max_sensor_age_s: float = 0.25

    cruise_speed: float = 0.36
    no_lane_speed: float = 0.31
    avoid_turn: float = 0.6

    max_pwm: int = 2500
    line_kp: float = 0.7
    min_confidence: float = 0.15

    wheel_track: float = 14.0 # distance between tracks in cm, used for kinematic calculations



