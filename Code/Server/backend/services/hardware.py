from __future__ import annotations

from typing import Optional


class VehicleHardware:
    """A wrapper around the old hardware interface. 
    The main purpose is to isolate key hardware interactions 
    and provide a clean API for the pipeline modules.
    """

    def __init__(self) -> None:
        self._car = None
        self._camera = None
        self._led = None
        self._ready: bool = False
        self._error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> Optional[str]:
        return self._error

    def start(self) -> None:
        try:
            from model.car import Car
            from model.misc.led import Led
            from model.sensors.camera import Camera

            self._car: Car = Car()
            self._led: Led = Led()
            self._camera: Camera = Camera(stream_size=(400, 300))
            self._ready = True
            self._error = None
        except Exception as exc:
            self._ready = False
            self._error = str(exc)

    def stop(self) -> None:
        try:
            if self._camera:
                try:
                    self._camera.stop_stream()
                except Exception:
                    pass
                self._camera.close()
            if self._led:
                self._led.colorWipe((0, 0, 0), 10)
            if self._car:
                self.stop_motors()
                self._car.close()
        finally:
            self._car = None
            self._camera = None
            self._led = None
            self._ready = False

    def start_camera_stream(self) -> None:
        if self._camera:
            self._camera.start_stream()

    def stop_camera_stream(self) -> None:
        if self._camera:
            self._camera.stop_stream()

    def get_jpeg_frame(self) -> Optional[bytes]:
        if self._camera:
            return self._camera.get_frame()
        return None

    def stop_motors(self) -> None:
        if self._car:
            self._car.motor.setMotorModel(0, 0)

    def set_motor(self, left: int, right: int) -> None:
        """Set motor speeds. Expects values in range [-4095, 4095]. Positive is forward."""
        if self._car:
            self._car.motor.setMotorModel(left, right)

    def set_servo(self, index: int, angle: int) -> None:
        """Set servo angle. Expects index in [0, 2] and angle in [0, 180]."""
        if self._car:
            self._car.servo.setServoAngle(index, angle)

    def set_led(self, mode: str, r: int, g: int, b: int, index: int) -> None:
        """Set LED mode and color. Mode can be 'off', 'index', 'blink', 'breathing', or 'rainbow'."""
        if not self._led:
            return
        if mode == "off":
            self._led.colorWipe((0, 0, 0), 10)
        elif mode == "index":
            self._led.ledIndex(index, r, g, b)
        elif mode == "blink":
            self._led.Blink((r, g, b), 50)
            self._led.Blink((0, 0, 0), 50)
        elif mode == "breathing":
            self._led.Breathing((r, g, b))
        elif mode == "rainbow":
            self._led.rainbowCycle()

    def read_ultrasonic(self) -> Optional[float]:
        """Returns distance in cm, or None if error."""
        if self._car:
            try:
                return float(self._car.sonic.get_distance())
            except Exception:
                return None
        return None

    def read_infrared(self) -> Optional[int]:
        """Returns raw 3-bit pattern from IR sensors, or None if error."""
        if self._car:
            try:
                return int(self._car.infrared.read_all_infrared())
            except Exception:
                return None
        return None