from pytvc.physics import *
from pytvc.data import rocketMotor

class rocket:

    def __init__(self, dry_mass: float = 1.0, time_step: float = 0.001, name: str = "") -> None:
        """initializes the rocket

        Args:
            dry_mass (float, optional): mass of the rocket without any motors. Defaults to 1.0.
            time_step (float, optional): time between steps in the simulation. Defaults to 0.001.
            name (str, optional): the name of the rocket. Defaults to ""
        """
        self.name = name
        self.time = 0.0
        self.time_step = time_step

        self._tvc_mounts = {}

        self.dry_mass = dry_mass
        self.engine_mass = 0.0
        self.motor_thrust = 0.0

        self.body: physicsBody = physicsBody()

        self._function_registry = {}

    def setup(self, func):
        self._function_registry["setup"] = func
        return func

    def update(self, func):
        self._function_registry["update"] = func
        return func
    
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

    def _update(self) -> None:

        """update the body's simulation by one time step"""

        self.time += self.time_step
        
        tmp_m = 0.0
        for mount in self._tvc_mounts:
            tm, f = self._tvc_mounts[mount]["mount"].update(self.time)
            tmp_m += tm*0.001
            self.motor_thrust += f.length()
            self.body.apply_local_point_force(f, self._tvc_mounts[mount]["position"])

        self.body.mass = self.dry_mass + tmp_m
        self.engine_mass = tmp_m
        
        self.body.update(self.time_step)
        
        if "update" in self._function_registry:
            self._function_registry["update"]()

        # datalogging?
        
        self.body.clear()
    
    def initialize(self) -> None:
        if "setup" in self._function_registry:
            self._function_registry["setup"]()

class sim:

    def __init__(self, time_step: float = 0.001, time_end: float = 60.0, rockets: dict = {}) -> None:

        self.time: float = 0.0
        self.time_step = time_step
        self.time_end = time_end
        self.nRockets = 0
        self._rockets = {}

        for r in rockets:
            if isinstance(r, rocket):
                if r.name == "":
                    r.name = "rocket" + str(self.nRockets)
                self._rockets[r.name] = r              
                self.nRockets += 1
            else:
                raise TypeError("Rocket must be a rocket class")
        return None

    def add_rocket(self, r: rocket) -> None:
        if isinstance(r, rocket):
            if r.name == "":
                r.name = "rocket" + str(self.nRockets)
            self._rockets[r.name] = r              
            self.nRockets += 1
        else:
            raise TypeError("Rocket must be a rocket class")

    def run(self) -> None:

        return None