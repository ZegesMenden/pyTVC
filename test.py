import pytvc
from pytvc.simulation import rocket, sim
from pytvc.physics import Vec3, Quat, TVC, rocket_motor, parachute
from pytvc.control import PID, torque_PID
import numpy as np

# create the simulation object
simulation: sim = sim(
    time_end=30.0,
    sim_wind=False,
    random_wind=False
)

# create our first rocket
rocket_1: rocket = rocket(
    name="rocket_1",
    log_values=True,
    do_aero=False,
    dry_mass=0.8,
    refence_area=np.pi*(0.076/2)**2,
    drag_coeff_forewards=1.35,
    drag_coeff_sideways=7.0,
    cp_location=Vec3(-0.0, 0.0, 0.0),
    friction_coeff=0.00001,
    moment_of_inertia=(Vec3(0.0404, 0.0548202733120976, 0.0548202733120976))
)

# make a tvc mount for said rocket
tvc_1: TVC = TVC(
    name="tvc_1",
    log_data=True,
    speed=13.0,
    maxPosition=Vec3(0.0, 0.1, 0.1),
    minPosition=Vec3(0.0, -0.1, -0.1),
    noise=0.00001
)

PID_y: torque_PID = torque_PID(Kp=100, Ki=0.0, Kd=28.46, inertia=rocket_1.body.moment_of_inertia.y, lever_arm=0.2)
PID_z: torque_PID = torque_PID(Kp=100, Ki=0.0, Kd=28.46, inertia=rocket_1.body.moment_of_inertia.z, lever_arm=0.2)

@tvc_1.update_func
def tvc1_update():

    target_vector: Vec3 = Vec3(1.0, 0.0, 0.0)

    if simulation.time > 1.0:
        target_vector.y += 0.1

    target_vector = rocket_1.body.rotation.conjugate().rotate(target_vector)

    PID_y.update(np.arctan2(target_vector.z, target_vector.x), tvc_1.update_delay, rocket_1.motor_thrust)
    PID_z.update(-np.arctan2(target_vector.y, target_vector.x), tvc_1.update_delay, rocket_1.motor_thrust)

    return Vec3(0.0, PID_y.getOutput(), PID_z.getOutput())

# motor for the mount
motor_1: rocket_motor = rocket_motor(
    filePath="C:/Users/Cameron/Documents/vscode/rockets/simulation/G-FIELD/motors/Estes_F15.rse", timeStep=1000)

parachute_1: parachute = parachute(diameter=0.9)

@parachute_1.deploy_func
def parachute_1_deploy_func(position: Vec3, velocity: Vec3):
    
    if position.x < 20.0 and velocity.x < -1.0:
        return True
    else:
        return False

# add the motor to the mount
tvc_1.add_motor(motor_1)

# add the mount to the rocket
rocket_1.add_tvc_mount(tvc_mount=tvc_1, position=Vec3(0.2, 0.0, 0.0))

# add the parachute to the rocket
rocket_1.add_parachute(parachute_1, Vec3(0.0, 0.0, 0.0))

# add the rocket to the sim
simulation.add_rocket(rocket_1)

# set motor ignition time
motor_1.set_ignition_time(0.0)

# and run it!
simulation.run()
