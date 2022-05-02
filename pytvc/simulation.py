from pytvc.physics import *
from pytvc.data import data_logger, plotter, progress_bar
import numpy as np
import os

RAD2DEG = 180.0/np.pi

class rocket:

    def __init__(self, dry_mass: float = 1.0, time_step: float = 0.001, name: str = "", log_values: bool = False, log_frequency: int = 50, print_info: bool = False, do_aero: bool = True, drag_coeff_forewards: float = 0.0, drag_coeff_sideways: float = 0.0, refence_area: float = 0.0, cp_location: Vec3 = Vec3(), moment_of_inertia: Vec3 = Vec3(1.0, 1.0, 1.0), friction_coeff: float = 0.0) -> None:
        """__init__ initializes the rocket

        Args:
            dry_mass (float, optional): the mass of the vehicle with no motors. Defaults to 1.0.
            time_step (float, optional): time between updates. Defaults to 0.001.
            name (str, optional): name of the rocket. Defaults to "".
            log_values (bool, optional): if true the rocket will log values and export it to a CSV. Defaults to False.
            log_frequency (int, optional): frequency at which to log data. Defaults to 50.
            print_info (bool, optional): if true the rocket will print debug information while the simulation is initializing and running. Defaults to False.
            do_aero (bool, optional): enable aerodynamics simulation. Defaults to True.
            drag_coeff_forewards (float, optional): the drag coefficient when the rocket is pointed into the airstream. Defaults to 0.0.
            drag_coeff_sideways (float, optional): the drag coefficient when the rocket is at an angle of attack of 90 degrees. Defaults to 0.0.
            refence_area (float, optional): aerodynamic reference area of the rocket. Defaults to 0.0.
            cp_location (Vec3, optional): location of the center of pressure on the rocket. Defaults to Vec3().
            moment_of_inertia (Vec3, optional): moment of inertia of the rocekt. Defaults to Vec3(1.0, 1.0, 1.0).
        """

        # timing and variables
        self.print_info = print_info
        self.log_values = log_values
        self.log_delay: float = 1 / log_frequency
        self.last_log_time: float = 0
        self.name = name
        self.time = 0.0
        self.time_step = time_step
        self.apogee = 0.0

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

        self.body: physics_body = physics_body(
            use_aero=do_aero,
            drag_coefficient_forewards=drag_coeff_forewards,
            drag_coefficient_sideways=drag_coeff_sideways,
            moment_of_inertia=moment_of_inertia,
            mass=dry_mass,
            ref_area=refence_area,
            cp_location=cp_location,
            friction_coeff=friction_coeff
        )

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

    def add_parachute(self, chute: parachute, position: Vec3):

        if isinstance(chute, parachute):
            self._n_parachutes += 1
            if chute.name == "":
                chute.name = f"parachute_{self._n_parachutes}" 
            self._parachutes[chute.name] = {"parachute": chute, "position": position}
        else:
            raise Exception("Parachute must be of type parachute")
    
    def _initialize(self) -> None:
        if self.log_values:
            if self.print_info:
                print(f"    {self.name}: initializing data logger")
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
            self.datalogger.add_datapoint("rotation_x")
            self.datalogger.add_datapoint("rotation_y")
            self.datalogger.add_datapoint("rotation_z")
            self.datalogger.add_datapoint("motor_thrust")
            self.datalogger.add_datapoint("engine_mass")
            self.datalogger.add_datapoint("dry_mass")
            self.datalogger.add_datapoint("mass")

            for tvc in self._tvc_mounts:
                if self.print_info:
                    print(f"    {self.name}: adding TVC mount {tvc}")
                t = self._tvc_mounts[tvc]["mount"]
                if isinstance(t, TVC):
                    if t.log_data:
                        if self.print_info:
                            print(f"    {self.name}: initializing TVC mount {t.name} data logger")
                        self.datalogger.add_datapoint(f"{t.name}_position_y")
                        self.datalogger.add_datapoint(f"{t.name}_position_z")
                        self.datalogger.add_datapoint(f"{t.name}_throttle")
                        self.datalogger.add_datapoint(f"{t.name}_setpoint_y")
                        self.datalogger.add_datapoint(f"{t.name}_setpoint_z")
                        self.datalogger.add_datapoint(f"{t.name}_error_y")
                        self.datalogger.add_datapoint(f"{t.name}_error_z")

        if "setup" in self._function_registry:
            if self.print_info:
                print(f"    {self.name}: running user setup function")
            self._function_registry["setup"]()

        if self.log_values:        
            self.datalogger.fileName = f"{self.name}_log.csv"
            self.datalogger.initialize_csv(True, True)

    def _update(self) -> None:

        """update the body's simulation by one time step"""

        self.time += self.time_step

        if self.body.position.x < 0.01 and self.body.velocity.x < 0.0:
            self._grounded = True

        tmp_m = 0.0
        for mount in self._tvc_mounts:
            self._tvc_mounts[mount]["mount"].update(self.time_step)
            tm, f = self._tvc_mounts[mount]["mount"].get_values(self.time)
            tmp_m += tm*0.001
            self.motor_thrust += f.length()
            self.body.apply_local_point_force(f, self._tvc_mounts[mount]["position"])

        if self.body.position.x > 0.01:
            for p in self._parachutes:
                if self._parachutes[p]["parachute"]._check(self.body.position, self.body.velocity):
                    self.body.apply_force(self._parachutes[p]["parachute"].calculate_forces(self.body.mass, self.body.velocity))#, self._parachutes[p]["position"])

        self.body.mass = self.dry_mass + tmp_m
        self.engine_mass = tmp_m
        
        self.body.update(self.time_step)

        if self.body.position.x > self.apogee:
            self.apogee = self.body.position.x
        
        if "update" in self._function_registry:
            self._function_registry["update"]()

        if self.log_values and self.time > self.last_log_time + self.log_delay:
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
            self.datalogger.record_variable("rotation_x", self.body.rotation_euler.x * RAD2DEG)
            self.datalogger.record_variable("rotation_y", self.body.rotation_euler.y * RAD2DEG)
            self.datalogger.record_variable("rotation_z", self.body.rotation_euler.z * RAD2DEG)
            self.datalogger.record_variable("angular_velocity_x", self.body.rotational_velocity.x)
            self.datalogger.record_variable("angular_velocity_y", self.body.rotational_velocity.y)
            self.datalogger.record_variable("angular_velocity_z", self.body.rotational_velocity.z)
            self.datalogger.record_variable("motor_thrust", self.motor_thrust)
            self.datalogger.record_variable("engine_mass", self.engine_mass)
            self.datalogger.record_variable("dry_mass", self.dry_mass)
            self.datalogger.record_variable("mass", self.body)

            for tvc in self._tvc_mounts:
                if self._tvc_mounts[tvc]["mount"].log_data:
                    t = self._tvc_mounts[tvc]["mount"]
                    self.datalogger.record_variable(f"{t.name}_position_y", t._rotationEulers.y * RAD2DEG)
                    self.datalogger.record_variable(f"{t.name}_position_z", t._rotationEulers.z * RAD2DEG)
                    self.datalogger.record_variable(f"{t.name}_throttle", t._throttle)
                    self.datalogger.record_variable(f"{t.name}_setpoint_y", t.targetEulers.y * RAD2DEG) 
                    self.datalogger.record_variable(f"{t.name}_setpoint_z", t.targetEulers.z * RAD2DEG) 
                    self.datalogger.record_variable(f"{t.name}_error_y", 0.0)
                    self.datalogger.record_variable(f"{t.name}_error_z", 0.0)
            self.datalogger.save_data(True)
            self.last_log_time = self.time
            
        self.body.clear()

class sim:
    
    def __init__(self, plot_data: bool = True, time_step: float = 0.001, time_end: float = 60.0, rockets: dict = {}, print_times: bool = True, print_info: bool = True, wind: Vec3 = Vec3(), sim_wind: bool = False, random_wind: bool = False) -> None:
        self.time: float = 0.0
        self.time_step = time_step
        self.time_end = time_end
        self.nRockets = 0
        self._rockets = {}

        self.plot_data = plot_data
        self.print_times = print_times
        self.print_info = print_info

        self.use_wind = sim_wind

        if random_wind:
            self.wind: Vec3 = Vec3(0.0, np.random.normal(0.0, 0.5, 1)[0], np.random.normal(0.0, 0.5, 1)[0])
        else:
            self.wind: Vec3 = wind        

        for r in rockets:
            if isinstance(r, rocket):
                if r.name == "":
                    r.name = "rocket" + str(self.nRockets)
                if sim_wind:
                    r.body.wind_speed = self.wind
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

        if self.print_info:
            print("Starting simulation...")

            print(f"""sim info:

rocket count: {self.nRockets}
wind speed: {round(self.wind, 2)}""")

        if self.nRockets == 0:
            raise Exception("No rockets to simulate!")

        for r in self._rockets:
            if self.print_info:
                print(f"Initializing {r}...")
                self._rockets[r].print_info = True
            self._rockets[r]._initialize()
            if self.print_info:
                print(f"{r} initialized")

        while self.time < self.time_end:

            self.time += self.time_step
            if self.time % 0.5 < self.time_step:
                print("Running simulation " + progress_bar(self.time, self.time_end), end="\r")
                
            for r in self._rockets:
                self._rockets[r]._update()

        if self.plot_data:

            print("")

            for r in self._rockets:
                _rocket: rocket = self._rockets[r]
                if isinstance(_rocket, rocket):

                    pTmp: plotter = plotter()
                    pTmp.read_header(f"{_rocket.name}_log.csv")

                    pTmp.create_2d_graph(['time', 'position_x', 'position_y', 'position_z', 'velocity_x', 'velocity_y', 'velocity_z', 'acceleration_x_world', 'acceleration_y_world', 'acceleration_z_world'], "x", "y", True, posArg=221)
                    # pTmp.create_2d_graph(['time', 'velocity_x', 'velocity_y', 'velocity_z'], "x", "y", True)
                    pTmp.create_3d_graph(['position_x', 'position_y', 'position_z'], size=_rocket.apogee, posArg=122)
                    for tvc in _rocket._tvc_mounts:
                        t = _rocket._tvc_mounts[tvc]["mount"]
                        if isinstance(t, TVC):
                            pTmp.create_2d_graph(['time', 'rotation_x', 'rotation_y', 'rotation_z', f'{t.name}_position_y', f'{t.name}_position_z', f'{t.name}_setpoint_y', f'{t.name}_setpoint_z'], "x", "y", True, posArg=223)
                        else:
                            pTmp.create_2d_graph(['time', 'rotation_x', 'rotation_y', 'rotation_z'], "x", "y", True)
                    # pTmp.create_2d_graph(['time', 'acceleration_x_world', 'acceleration_y_world', 'acceleration_z_world'], "x", "y", True)

                    
                    pTmp.show_all_graphs()

        return None