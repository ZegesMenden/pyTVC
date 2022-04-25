from os import times
from pytvc.physics import *
from pytvc.data import rocketMotor

class rocketBody:

    def __init__(self, dry_mass: float = 0.0, time_step: float = 0.0) -> None:

        self.time = 0.0
        self.time_step = time_step

        self._tvc_mounts = {}

        self.dry_mass = dry_mass
        self.engine_mass = 0.0
        self.motor_thrust = 0.0

        self.body: physicsBody = physicsBody()

    def add_tvc_mount(self, tvc_mount: TVC, position: Vec3, name: str) -> None:
        """adds a TVC mount to the rocket

        Args:
            tvc_mount (TVC): TVC mount to add
            position (Vec3): position from the center of mass
            name (str): name of the mount

        Raises:
            Exception: _description_
            Exception: _description_
        """
        if name in self._tvc_mounts:
            raise Exception("TVC mount already exists")
        else:
            if isinstance(tvc_mount, TVC):
                self._tvc_mounts[name] = {"mount": tvc_mount, "position": position}
            else:
                raise Exception("TVC mount must be of type TVC")

    def update(self) -> None:

        self.time += self.time_step
        
        tmp_m, tmp_t = 0.0, Vec3



        self.body.update(self.time_step)
        