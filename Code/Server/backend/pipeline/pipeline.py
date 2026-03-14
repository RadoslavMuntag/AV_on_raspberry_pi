from __future__ import annotations
from dataclasses import dataclass

from backend.pipeline.controller import DifferentialDriveController
from backend.pipeline.planner import BehaviorPlanner
from backend.pipeline.fusion import FusionModule
from backend.pipeline.config import PipelineConfig
from backend.pipeline.perception import PerceptionModule

from backend.contracts import ManualCommand, PerceptionFrame, PlannerDecision, ControlTargets, WorldState, BehaviorState

from backend.services.hardware import VehicleHardware

@dataclass(slots=True)
class PipelineSnapshot:
    perception: PerceptionFrame
    world: WorldState
    decision: PlannerDecision
    control: ControlTargets


class ModularPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        perception: PerceptionModule | None = None,
        fusion: FusionModule | None = None,
        planner: BehaviorPlanner | None = None,
        controller: DifferentialDriveController | None = None,
    ) -> None:
        self.config : PipelineConfig = config or PipelineConfig()
        self.perception: PerceptionModule = perception or PerceptionModule(cfg=self.config)
        self.fusion: FusionModule = fusion or FusionModule(cfg=self.config)
        self.planner: BehaviorPlanner = planner or BehaviorPlanner(cfg=self.config)
        self.controller: DifferentialDriveController = controller or DifferentialDriveController(cfg=self.config)

    def tick(
        self,
        hardware: VehicleHardware,
        requested_mode: BehaviorState,
        heartbeat_ok: bool,
        manual_cmd: ManualCommand
    ) -> PipelineSnapshot:
        p : PerceptionFrame = self.perception.read(hardware)
        w : WorldState = self.fusion.fuse(p)
        d : PlannerDecision = self.planner.step(w, requested_mode=requested_mode, heartbeat_ok=heartbeat_ok)
        u : ControlTargets = self.controller.step(d, w, manual_cmd)

        hardware.set_motor(u.left_pwm, u.right_pwm)
        hardware.set_led(u.led_mode, u.led_rgb[0], u.led_rgb[1], u.led_rgb[2], 0)

        return PipelineSnapshot(perception=p, world=w, decision=d, control=u)