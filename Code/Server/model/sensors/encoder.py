from gpiozero import DigitalInputDevice

class MotorEncoder:
    def __init__(
        self,
        signal_pin: int = 16,
        ticks_per_revolution: int = 10,
        pull_up: bool = True,
        bounce_time: float = 0.001, # 1 ms debounce time to prevent false ticks
    ) -> None:
        self.encoder: DigitalInputDevice = DigitalInputDevice(
            signal_pin,
            pull_up=pull_up,
            bounce_time=bounce_time,
        )
        self.ticks_per_revolution: int = ticks_per_revolution
        self.current_ticks: int = 0

        self.encoder.when_activated = self._update_ticks

    def _update_ticks(self) -> None:
        self.current_ticks += 1

    def reset(self) -> None:
        self.current_ticks = 0

    def get_distance(self, wheel_circumference: float) -> float:
        revolutions = self.current_ticks / self.ticks_per_revolution
        return revolutions * wheel_circumference

    def close(self) -> None:
        self.encoder.close()