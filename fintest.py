from pytvc.rigidBody import RigidBody, Vector3, Quaternion
from aero import AeroComponent, Fin
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# def dragFunction(aoa: float, flowSpeed, state: RigidBody) -> float:
#     return 0.03

# def liftFunction(aoa: float, flowSpeed, state: RigidBody) -> float:
#     return 0.05

# finArea = 0.2 * 0.1 

# fin0 = Fin(
#     Vector3(0.0, 0.0, 0.1), 
#     dragFunction, 
#     liftFunction, 
#     finArea, 
#     0 * np.pi/180
# )

# fin1 = Fin(
#     Vector3(0.0, 0.0, -0.1), 
#     dragFunction, 
#     liftFunction, 
#     finArea, 
#     180 * np.pi/180
# )

# fin2 = Fin(
#     Vector3(0.0, 0.1, 0.0), 
#     dragFunction, 
#     liftFunction, 
#     finArea, 
#     90 * np.pi/180
# )

# fin3 = Fin(
#     Vector3(0.0, -0.1, 0.0), 
#     dragFunction, 
#     liftFunction, 
#     finArea, 
#     -90 * np.pi/180
# )

# state: RigidBody = RigidBody(
#     1.0, 
#     Vector3(0.01, 0.01, 0.01), 
#     Vector3(0.0, 0.0, 0.0), 
#     Vector3(60.0, 0.0, 0.0), 
#     Quaternion.fromEulerAngles(
#         Vector3(
#             0.0, 
#             0.0, 
#             0.0
#         ) * np.pi/180
#     ),
#     Vector3(0.0, 0.0, 0.0)
# )

# fin0.setAngle(15 * np.pi/180)
# fin1.setAngle(15 * np.pi/180)
# fin2.setAngle(15 * np.pi/180)
# fin3.setAngle(15 * np.pi/180)

# t_sim = 0.0
# t_end = 1.0
# dt = 0.001

# velArr = []

# rotationArr = []
# rotVelArr = []
# forceArr = []
# torqueArr = []

# finDirectionArr = []
# finLiftArr = []
# finDragArr = []

# while t_sim < t_end:

#     forces = Vector3(0.0, 0.0, 0.0)

#     finDirs = []
#     finLifts = []
#     finDrags = []
#     for fin in [fin0, fin1, fin2, fin3]:
#         finforce = fin.getForces(state)
#         finLifts.append(finforce)
#         finDrags.append(finforce)
#         world_pos = state.rotation.rotate(fin.position)
#         state.applyForce(finforce, world_pos)

#     forceArr.append(state._accel)
#     torqueArr.append(state._torque)
#     finDirectionArr.append(finDirs)
#     finDragArr.append(finDrags)
#     velArr.append(state.velocity)
#     finLiftArr.append(state._torque)

#     state.update(dt)

#     rotationArr.append(state.rotation.toEulerAngles() * 180/np.pi)
#     rotVelArr.append(state.rotVel * 180/np.pi)

#     t_sim += dt
#     print(f"t_sim: {t_sim}")

# # plot

# rotationArr = [list(x) for x in rotationArr]
# rotVelArr = [list(x) for x in rotVelArr]
# finLiftArr = [list(x) for x in finLiftArr]

# rotationArr = np.array(rotationArr)
# rotVelArr = np.array(rotVelArr)
# finLiftArr = np.array(finLiftArr)

# fig, axs = plt.subplots(3, 1)
# axs[0].plot(rotationArr[:, 0], label='Roll')
# axs[0].plot(rotationArr[:, 1], label='Pitch')
# axs[0].plot(rotationArr[:, 2], label='Yaw')

# axs[1].plot(rotVelArr[:, 0], label='Roll')
# axs[1].plot(rotVelArr[:, 1], label='Pitch')
# axs[1].plot(rotVelArr[:, 2], label='Yaw')

# axs[2].plot(finLiftArr[:, 0], label='X')
# axs[2].plot(finLiftArr[:, 1], label='Y')
# axs[2].plot(finLiftArr[:, 2], label='Z')

# # axs[2].plot([vel.x for vel in velArr], label='X')
# # axs[2].plot([vel.y for vel in velArr], label='Y')
# # axs[2].plot([vel.z for vel in velArr], label='Z')

# axs[0].legend()
# axs[1].legend()
# axs[2].legend()
# plt.show()


import numpy as np
from pytvc.rigidBody import Vector3, Quaternion

def fin_relative_flow(
    v_inertial: Vector3,
    rotation: Quaternion,
    fin_pos_body: Vector3,
    fin_roll: float,          # rotation of fin around rocket long axis (rad)
    fin_deflection: float,    # fin deflection about its hinge axis (rad)
    q_base_BF: Quaternion,    # base body->fin rotation at zero roll/deflection
    omega_body: Vector3 | None = None,
    rho: float = 1.225,
    hinge_axis_body: Vector3 = Vector3(0, 1, 0),
):
    """
    Compute local airflow at a fin in its own coordinate frame.

    Parameters
    ----------
    v_inertial : Vector3
        Rocket translational velocity in inertial/world frame.
    rotation : Quaternion
        Rocket attitude quaternion that maps body -> inertial.
    fin_pos_body : Vector3
        Position of fin reference point (e.g. hinge) in body frame, relative to CG.
    fin_roll : float
        Fin roll angle around rocket long axis (radians).
    fin_deflection : float
        Fin deflection angle about hinge axis (radians).
    q_base_BF : Quaternion
        Base orientation of fin relative to body when roll=0, deflection=0.
        Maps body frame -> fin frame.
    omega_body : Vector3, optional
        Angular velocity of rocket in body frame (rad/s).
        If None, rotational contribution to local flow is ignored.
    rho : float
        Air density (kg/m^3).
    hinge_axis_body : Vector3, optional
        Hinge axis expressed in body coordinates.
        If None, defaults to spanwise axis lying in body Y–Z plane.

    Returns
    -------
    dict
        {
          "v_rel_F": Vector3,  # air-relative velocity in fin frame
          "speed": float,      # magnitude of v_rel_F
          "alpha": float,      # angle of attack (rad) in fin frame
          "beta": float,       # spanwise sideslip (rad)
          "q_dyn": float       # dynamic pressure at fin
        }
    """

    # 1) Air-relative velocity in inertial frame (assuming still air)
    #    v_air_inertial = 0, so v_rel = v_air - v_rocket = -v_rocket
    v_rel_inertial = Vector3(-v_inertial.x, -v_inertial.y, -v_inertial.z)

    # 2) Transform air-relative velocity into body frame
    # rotation : body -> inertial, so body = rotation_conj.rotate(world)
    q_IB_conj = rotation.conjugate()
    v_rel_body = q_IB_conj.rotate(v_rel_inertial)

    # 3) Local velocity at the fin (add contribution from angular velocity)
    if omega_body is not None:
        # v_rot = omega x r  (all in body frame)
        v_rot = omega_body.cross(fin_pos_body)
        v_fin_body = Vector3(
            v_rel_body.x + v_rot.x,
            v_rel_body.y + v_rot.y,
            v_rel_body.z + v_rot.z,
        )
    else:
        v_fin_body = v_rel_body

    # 4) Build fin orientation quaternion: body -> fin

    # 4a) Roll the fin around the rocket body X-axis
    body_x = Vector3(1.0, 0.0, 0.0)
    q_roll = Quaternion.fromAxisAngle(body_x, fin_roll)

    # 4b) Deflection about hinge axis (in body frame)
    if hinge_axis_body is None:
        # Example: hinge axis roughly spanwise, lying in body Y–Z plane.
        # You can change this to whatever your geometry actually uses.
        hinge_axis_body = Vector3(0.0, 1.0, 0.0)
    hinge_axis_body = hinge_axis_body.norm()
    q_def = Quaternion.fromAxisAngle(hinge_axis_body, fin_deflection)

    # 4c) Combine: base orientation, then roll, then deflection
    # Order: v_F = q_FB * v_B * q_FB^*
    # Composition: base (body->fin0), then roll, then deflection
    q_FB = q_def * (q_roll * q_base_BF)

    # 5) Rotate local velocity into fin frame
    v_rel_F = q_FB.rotate(v_fin_body)

    # 6) Extract speed and angles
    V = abs(v_rel_F)  # uses Vector3.__abs__

    if V > 0.0:
        # In fin frame:
        # x_F: chordwise (leading edge direction)
        # y_F: spanwise
        # z_F: normal to fin surface
        vx, vy, vz = v_rel_F.x, v_rel_F.y, v_rel_F.z

        # Angle of attack: flow vs chord in x–z plane
        alpha = np.arctan2(-vz, vx)   # sign convention: positive alpha if flow has -z component

        # Spanwise sideslip: component along span
        beta = np.arcsin(vy / V)
    else:
        alpha = 0.0
        beta = 0.0

    q_dyn = 0.5 * rho * (V**2)

    return {
        "v_rel_F": v_rel_F,
        "speed": V,
        "alpha": alpha,
        "beta": beta,
        "q_dyn": q_dyn,
    }

def q_base_from_fin_pos(fin_pos_body: Vector3) -> Quaternion:
    # 1) span/hinge direction (project onto body YZ plane)
    s = Vector3(0.0, fin_pos_body.y, fin_pos_body.z).norm()

    # 2) chord direction
    c = Vector3(1.0, 0.0, 0.0)  # body x-axis

    # 3) normal via right-hand rule
    n = c.cross(s).norm()

    # Rotation matrix: body->fin
    # Each row is a fin basis vector expressed in body frame
    R = np.array([
        [c.x, c.y, c.z],  # x_F in body frame
        [s.x, s.y, s.z],  # y_F in body frame
        [n.x, n.y, n.z],  # z_F in body frame
    ])

    return Quaternion.fromRotationMatrix(R)

q_bf = q_base_from_fin_pos(Vector3(0.0, 0.0, 0.1))

data = fin_relative_flow(
    Vector3(100, 0, 0),
    Quaternion(),
    Vector3(0, 0, 0.1),
    0*np.pi/180,
    45.0 * np.pi/180,
    q_bf,
    Vector3(),
    1.225,
    Vector3(0, 0, 1)
)

print(data)