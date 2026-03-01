from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from ..services.hardware import VehicleHardware

from ..contracts import ManualCommand, PerceptionFrame, PlannerDecision, ControlTargets, WorldState, BehaviorState
from .controller import DifferentialDriveController
from .fusion import FusionModule
from .perception import PerceptionModule
from .planner import BehaviorPlanner


@dataclass(slots=True)
class PipelineSnapshot:
    perception: PerceptionFrame
    world: WorldState
    decision: PlannerDecision
    control: ControlTargets


class ModularPipeline:
    def __init__(
        self,
        perception: Optional[PerceptionModule] = None,
        fusion: Optional[FusionModule] = None,
        planner: Optional[BehaviorPlanner] = None,
        controller: Optional[DifferentialDriveController] = None,
    ) -> None:
        self.perception = perception or PerceptionModule()
        self.fusion = fusion or FusionModule()
        self.planner = planner or BehaviorPlanner()
        self.controller = controller or DifferentialDriveController()

    def tick(
        self,
        hardware: VehicleHardware,
        requested_mode: BehaviorState,
        heartbeat_ok: bool,
        manual_cmd: ManualCommand,
    ) -> PipelineSnapshot:
        p : PerceptionFrame = self.perception.read(hardware)
        w : WorldState = self.fusion.fuse(p)
        d : PlannerDecision = self.planner.step(w, requested_mode=requested_mode, heartbeat_ok=heartbeat_ok)
        u : ControlTargets = self.controller.step(d, w, manual_cmd)

        hardware.set_motor(u.left_pwm, u.right_pwm)
        hardware.set_led(u.led_mode, u.led_rgb[0], u.led_rgb[1], u.led_rgb[2], 0)

        return PipelineSnapshot(perception=p, world=w, decision=d, control=u)