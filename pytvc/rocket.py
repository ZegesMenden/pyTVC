from os import times
from pytvc.physics import *
from pytvc.data import rocketMotor

class rocket_body:

    def __init__(self, dry_mass: float = 0.0, time_step: float = 0.0) -> None:

        self.time = 0.0
        self.time_step = time_step

        self._motors = {}
        self._tvc_mounts = {}

        self.dry_mass = dry_mass
        self.engine_mass = 0.0
        self.motor_thrust = 0.0

        self.body: physics_body = physics_body()
        
    def addMotor(self, motor, name: str = "") -> None:
        """addMotor adds a motor to the rocket

        Args:
            motor (str or rocketMotor): _description_
            name (str, optional): name of the roket motor, if no name is given the name defaults to motor + the motor index. Defaults to "".

        Raises:
            ValueError: if the motor is already added
            ValueError: if the time step is not equal to the time step of the rocket
            TypeError: the motor is not a rocketMotor or string
        """
        if isinstance(motor, rocketMotor):
            if motor._timeStep != self.time_step:
                raise ValueError("The time step of the rocket and the motor must be the same")
            else:
                if name in self._motors:
                    raise ValueError("A motor with the name " + name + " already exists")
                else:
                    self._motors[name] = motor
        elif isinstance(motor, str):
            if name in self._motors:
                raise ValueError("A motor with the name " + name + " already exists")
            else:
                self._motors[name] = rocketMotor(motor, self.time_step)
        else:
            raise TypeError("motor must be a rocketMotor or a string")

    def update(self) -> None:

        self.time += self.time_step
        
        tmp_m, tmp_t = 0.0, 0.0

        for motor in self._motors.values():
            tmp_m, tmp_t = motor.update(self.time)

        self.engine_mass = tmp_m
        self.motor_thrust = tmp_t

        self.body.update(self.time_step)
        