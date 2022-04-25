import pytvc
from pytvc.simulation import rocket, sim
from pytvc.physics import Vec3, Quat, TVC, rocketMotor

simulation: sim = sim()

rocket_1: rocket = rocket(name="rocket_1", log_values=True)

tvc_1: TVC = TVC(name="tvc_1")
motor_1: rocketMotor = rocketMotor(filePath="C:/Users/Cameron/Documents/vscode/rockets/simulation/G-FIELD/motors/Estes_E12.rse", timeStep=1000)
tvc_1.add_motor(motor_1)
rocket_1.add_tvc_mount(tvc_mount=tvc_1, position=Vec3(0.2, 0.0, 0.0))
simulation.add_rocket(rocket_1)

simulation.run()