from core.physics import *
from core.motors import *

class rocket:

    def __init__(self, dry_mass: float = 0.0, fuel_mass: float = 0.0, motors: list = [], time_step: float = 0.0) -> None:

        self.time_step = time_step
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass

        self.motor: rocket_motor = rocket_motor(time_step)
        for x in motors:
            self.motor.add_motor(x[0], x[1])

        self.body: physics_body = physics_body()
        