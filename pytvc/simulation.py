from pytvc.physics import *
from pytvc.data import data_logger 

from pytvc.data import data_visualizer, plotter

class rocket:

    def __init__(self, dry_mass: float = 1.0, time_step: float = 0.001, name: str = "", log_values: bool = False) -> None:
        """initializes the rocket

        Args:
            dry_mass (float, optional): mass of the rocket without any motors. Defaults to 1.0.
            time_step (float, optional): time between steps in the simulation. Defaults to 0.001.
            name (str, optional): the name of the rocket. Defaults to ""
        """

        # timing and variables
        self.log_values = log_values
        self.name = name
        self.time = 0.0
        self.time_step = time_step

        # components
        self._n_tvc_mounts: int = 0
        self._n_parachutes: int = 0
        self._tvc_mounts = {}
        self._parachutes = {}

        self.datalogger: data_logger = data_logger()

        # physical properties
        self.dry_mass = dry_mass
        self.engine_mass = 0.0
        self.motor_thrust = 0.0

        self.body: physics_body = physics_body()

        # function registry for update and setup
        self._function_registry = {}

    def setup(self, func):
        self._function_registry["setup"] = func
        return func

    def update(self, func):
        self._function_registry["update"] = func
        return func
    
    def add_tvc_mount(self, tvc_mount: TVC, position: Vec3) -> None:
        """adds a TVC mount to the rocket

        Args:
            tvc_mount (TVC): TVC mount to add
            position (Vec3): position from the center of mass
            name (str): name of the mount

        Raises:
            Exception: _description_
            Exception: _description_
        """
        
        if isinstance(tvc_mount, TVC):
            self._n_tvc_mounts += 1
            if tvc_mount.name == "":
                tvc_mount.name = f"TVC{self._n_tvc_mounts}" 
            
            if tvc_mount.name in self._tvc_mounts:
                raise Exception(f"TVC mount with name {tvc_mount.name} already exists")
            else:
                self._tvc_mounts[tvc_mount.name] = {"mount": tvc_mount, "position": position}
        else:
            raise Exception("TVC mount must be of type TVC")
    
    def _initialize(self) -> None:
        if self.log_values:
            self.datalogger.add_datapoint("time")
            self.datalogger.add_datapoint("position_x")
            self.datalogger.add_datapoint("position_y")
            self.datalogger.add_datapoint("position_z")
            self.datalogger.add_datapoint("velocity_x")
            self.datalogger.add_datapoint("velocity_y")
            self.datalogger.add_datapoint("velocity_z")
            self.datalogger.add_datapoint("acceleration_x_local")
            self.datalogger.add_datapoint("acceleration_y_local")
            self.datalogger.add_datapoint("acceleration_z_local")
            self.datalogger.add_datapoint("acceleration_x_world")
            self.datalogger.add_datapoint("acceleration_y_world")
            self.datalogger.add_datapoint("acceleration_z_world")
            self.datalogger.add_datapoint("angular_velocity_x")
            self.datalogger.add_datapoint("angular_velocity_y")
            self.datalogger.add_datapoint("angular_velocity_z")
            self.datalogger.add_datapoint("motor_thrust")
            self.datalogger.add_datapoint("engine_mass")
            self.datalogger.add_datapoint("dry_mass")
            self.datalogger.add_datapoint("mass")

            for tvc in self._tvc_mounts:
                if isinstance(tvc, TVC):
                    if tvc.log_data:
                        self.datalogger.add_datapoint(f"{tvc.name}_position_y")
                        self.datalogger.add_datapoint(f"{tvc.name}_position_z")
                        self.datalogger.add_datapoint(f"{tvc.name}_throttle")
                        self.datalogger.add_datapoint(f"{tvc.name}_setpoint_y")
                        self.datalogger.add_datapoint(f"{tvc.name}_setpoint_z")
                        self.datalogger.add_datapoint(f"{tvc.name}_error_y")
                        self.datalogger.add_datapoint(f"{tvc.name}_error_z")

        if "setup" in self._function_registry:
            self._function_registry["setup"]()

        if self.log_values:        
            self.datalogger.fileName = f"{self.name}_log.csv"
            self.datalogger.initialize_csv(True, True)

    def _update(self) -> None:

        """update the body's simulation by one time step"""

        self.time += self.time_step
        
        tmp_m = 0.0
        for mount in self._tvc_mounts:
            self._tvc_mounts[mount]["mount"].update(self.time_step)
            tm, f = self._tvc_mounts[mount]["mount"].get_values(self.time)
            tmp_m += tm*0.001
            self.motor_thrust += f.length()
            self.body.apply_local_point_force(f, self._tvc_mounts[mount]["position"])

        temp_p_force: Vec3 = Vec3()
        for p in self._parachutes:
            if isinstance(p, parachute):
                if p._check(self.body.position, self.body.velocity):
                    temp_p_force += p.calculate_forces(self.body.mass, self.body.velocitym)
        
        self.body.mass = self.dry_mass + tmp_m
        self.engine_mass = tmp_m
        
        self.body.update(self.time_step)
        
        if "update" in self._function_registry:
            self._function_registry["update"]()

        # datalogging?
        # set datalog speed
        if self.log_values:
            self.datalogger.record_variable("time", self.time)
            self.datalogger.record_variable("position_x", self.body.position.x)
            self.datalogger.record_variable("position_y", self.body.position.y)
            self.datalogger.record_variable("position_z", self.body.position.z)
            self.datalogger.record_variable("velocity_x", self.body.velocity.x)
            self.datalogger.record_variable("velocity_y", self.body.velocity.y)
            self.datalogger.record_variable("velocity_z", self.body.velocity.z)
            self.datalogger.record_variable("acceleration_x_local", self.body.acceleration_local.x)
            self.datalogger.record_variable("acceleration_y_local", self.body.acceleration_local.y)
            self.datalogger.record_variable("acceleration_z_local", self.body.acceleration_local.z)
            self.datalogger.record_variable("acceleration_x_world", self.body.acceleration.x)
            self.datalogger.record_variable("acceleration_y_world", self.body.acceleration.y)
            self.datalogger.record_variable("acceleration_z_world", self.body.acceleration.z)
            self.datalogger.record_variable("angular_velocity_x", self.body.rotational_velocity.x)
            self.datalogger.record_variable("angular_velocity_y", self.body.rotational_velocity.y)
            self.datalogger.record_variable("angular_velocity_z", self.body.rotational_velocity.z)
            self.datalogger.record_variable("motor_thrust", self.motor_thrust)
            self.datalogger.record_variable("engine_mass", self.engine_mass)
            self.datalogger.record_variable("dry_mass", self.dry_mass)
            self.datalogger.record_variable("mass", self.body)

            for tvc in self._tvc_mounts:
                if isinstance(tvc, TVC):
                    if tvc.log_data:
                        self.datalogger.record_variable(f"{tvc.name}_position_y", tvc._rotationEulers.y * np.rad2deg)
                        self.datalogger.record_variable(f"{tvc.name}_position_z", tvc._rotationEulers.z * np.rad2deg)
                        self.datalogger.record_variable(f"{tvc.name}_throttle", tvc._throttle)
                        self.datalogger.record_variable(f"{tvc.name}_setpoint_y", tvc.targetEulers.y * np.rad2deg) 
                        self.datalogger.record_variable(f"{tvc.name}_setpoint_z", tvc.targetEulers.z * np.rad2deg) 
                        self.datalogger.record_variable(f"{tvc.name}_error_y", 0.0)
                        self.datalogger.record_variable(f"{tvc.name}_error_z", 0.0)
        self.datalogger.save_data(True)
        self.body.clear()

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
        if self.nRockets == 0:
            raise Exception("No rockets to simulate!")
        for r in self._rockets:
            self._rockets[r]._initialize()

        while self.time < self.time_end:
            self.time += self.time_step
            for r in self._rockets:
                self._rockets[r]._update()
        
        
        p: plotter = plotter()
        # p.read_header('')

        return None