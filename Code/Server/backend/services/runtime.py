from __future__ import annotations

import asyncio
import time
from typing import Optional

from .hardware import VehicleHardware
from .state import StateStore
from ..pipeline.pipeline import ModularPipeline
from ..contracts import BehaviorState, ManualCommand
from ...model.dualsense.ds_device import DualSense

class RuntimeManager:
    def __init__(self, state_store: StateStore, hardware: VehicleHardware) -> None:
        self.state_store = state_store
        self.hardware = hardware
        self.pipeline = ModularPipeline()
        self._manual_cmd = ManualCommand()
        self._running = False
        self._telemetry_task: Optional[asyncio.Task] = None
        self._control_task: Optional[asyncio.Task] = None
        
        self._dualsense: Optional[DualSense] = None
        self._dualsense_connected = False

    async def start(self) -> None:
        self.hardware.start()
        self.state_store.update_state(
            hardware_ready=self.hardware.ready,
            hardware_error=self.hardware.error,
        )
        if self.hardware.ready:
            try:
                self.hardware.start_camera_stream()
                self.state_store.update_state(camera_streaming=True)
            except Exception as exc:
                self.state_store.update_state(camera_streaming=False, hardware_error=str(exc))
        self._running = True
        self._telemetry_task = asyncio.create_task(self._telemetry_loop()) 
        self._control_task = asyncio.create_task(self._control_loop())

        self._dualsense = DualSense(self.hardware, self.set_car_mode)           # Initialize the DualSense controller
        self._dualsense_connected = self._dualsense.init()  # Try to connect DualSense
        if self._dualsense_connected:
            print("DualSense controller connected.")
        else:
            print("DualSense controller not found. Continuing without controller.")

    async def stop(self) -> None:
        self._running = False
        if self._telemetry_task:
            self._telemetry_task.cancel()
        if self._control_task:
            self._control_task.cancel()
        self.hardware.stop()
        self.state_store.update_state(
            camera_streaming=False,
            hardware_ready=False,
            controller_id=None,
            left_motor=0,
            right_motor=0,
        )

    def acquire_controller(self, client_id: str) -> bool:
        snap = self.state_store.snapshot()["state"]
        current = snap["controller_id"]
        if current and current != client_id:
            return False
        self.state_store.update_state(controller_id=client_id, controller_last_seen=time.time())
        return True

    def release_controller(self, client_id: str) -> None:
        snap = self.state_store.snapshot()["state"]
        current = snap["controller_id"]
        if current == client_id:
            self.hardware.stop_motors()
            self.state_store.update_state(controller_id=None, left_motor=0, right_motor=0)

    def heartbeat(self, client_id: str) -> bool:
        snap = self.state_store.snapshot()["state"]
        if snap["controller_id"] != client_id:
            return False
        self.state_store.update_state(controller_last_seen=time.time())
        return True

    def set_mode(self, mode: BehaviorState) -> None:
        if mode == BehaviorState.SAFE_STOP:
            self.hardware.stop_motors()
            self.state_store.update_state(left_motor=0, right_motor=0, e_stop=True)

        elif mode in BehaviorState.__members__:
            if mode != BehaviorState.SAFE_STOP:
                self.state_store.update_state(e_stop=False)
        self.state_store.update_state(mode=mode)

    def drive(self, client_id: str, left: int, right: int) -> bool:
        snap = self.state_store.snapshot()
        state = snap["state"]
        cfg = snap["config"]
        
        if state["controller_id"] != client_id or state["e_stop"]:
            return False

        max_speed = cfg["max_motor_speed"]
        left = max(-max_speed, min(max_speed, left))
        right = max(-max_speed, min(max_speed, right))


        # Convert wheel command -> normalized manual command for pipeline controller
        throttle = (left + right) / (2.0 * max_speed)
        steer = (right - left) / (2.0 * max_speed)

        self._manual_cmd.throttle = max(-1.0, min(1.0, throttle))
        self._manual_cmd.steer = max(-1.0, min(1.0, steer))
        self._manual_cmd.active = True

        self.state_store.update_state(
            controller_last_seen=time.time(),
            left_motor=left,
            right_motor=right,
        )
        return True
    
    def set_car_mode(self, mode: BehaviorState) -> None:
        """Called by DualSense handler to switch between manual and autonomous mode."""
        if mode == BehaviorState.MANUAL:
            self.state_store.update_state(e_stop=False)
        self.state_store.update_state(mode=mode)

    async def _telemetry_loop(self) -> None:
        while self._running:
            self.state_store.update_state(
                ultrasonic_cm=self.hardware.read_ultrasonic(),
                infrared_value=self.hardware.read_infrared(),
                hardware_ready=self.hardware.ready,
                hardware_error=self.hardware.error,
            )
            await asyncio.sleep(0.1)

    async def _control_loop(self) -> None:
        while self._running:
            snap = self.state_store.snapshot()
            state = snap["state"]
            cfg = snap["config"]
            loop_delay = 1.0 / cfg["control_loop_hz"]

            timed_out = self.state_store.should_timeout_controller()
            if timed_out:
                self._manual_cmd = ManualCommand()  # reset manual input
                self.state_store.update_state(
                    controller_id=None,
                    mode=BehaviorState.SAFE_STOP,
                    e_stop=True,
                )

            requested_mode = state["mode"]

            # Heartbeat is required only for manual remote driving
            heartbeat_ok = True
            if requested_mode == BehaviorState.MANUAL:
                heartbeat_ok = (state["controller_id"] is not None) and (not timed_out)

            self._manual_cmd.active = requested_mode == BehaviorState.MANUAL and heartbeat_ok

            pipe = self.pipeline.tick(
                hardware=self.hardware,
                requested_mode=requested_mode,
                heartbeat_ok=heartbeat_ok and not state["e_stop"],
                manual_cmd=self._manual_cmd,
            )

            self.state_store.update_state(
                mode=pipe.decision.state.value,
                left_motor=pipe.control.left_pwm,
                right_motor=pipe.control.right_pwm,
                ultrasonic_cm=pipe.perception.ultrasonic_cm,
                infrared_value=pipe.perception.infrared_raw,
                hardware_ready=self.hardware.ready,
                hardware_error=self.hardware.error,
            )

            await asyncio.sleep(loop_delay)
