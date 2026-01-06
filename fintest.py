from pytvc.rigidBody import RigidBody, Vector3, Quaternion
from pytvc.actor import Actor, AeroComponent, RocketBody, Fin
from pytvc.rocket import Rocket
# from aero import AeroComponent, Fin
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def _v3_to_np(v: Vector3) -> np.ndarray:
    return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)

def _copy_v3(v: Vector3) -> Vector3:
    return Vector3(float(v.x), float(v.y), float(v.z))

def _copy_q(q: Quaternion) -> Quaternion:
    return Quaternion(float(q.w), float(q.x), float(q.y), float(q.z))

def dragFunctionFin(aoa: float, flowSpeed, state: RigidBody) -> float:
    return 0.03#*min(abs(aoa), 5 * np.pi/180)

def liftFunctionFin(aoa: float, flowSpeed, state: RigidBody) -> float:
    return np.clip(abs((0.451 / 15) * (aoa * 180/np.pi)), 0, 1.2)

def liftFunctionBody(aoa: float, vel: float, state: RigidBody) -> float:
    return np.clip(abs((0.451 / 15) * (aoa * 180/np.pi)), 0, 1.2)

def dragFunctionBody(aoa: float, vel: float, state: RigidBody) -> float:
    return 0.199

def areaFunctionBody(aoa: float, vel: float, state: RigidBody) -> float:
    return 0.00331830724 + ((0.033032192 - 0.00331830724) * (np.sin(aoa)**2))

def pressureFunction(state: RigidBody) -> float:
    return 1.225

finArea = 0.2 * 0.1 

fin0 = Fin(
    Vector3(-0.2, 0.1, 0.0), 
    dragFunctionFin, 
    liftFunctionFin, 
    pressureFunction,
    finArea, 
    0 * np.pi/180
)

fin1 = Fin(
    Vector3(-0.2, -0.1, 0.0), 
    dragFunctionFin, 
    liftFunctionFin, 
    pressureFunction,
    finArea, 
    180 * np.pi/180
)

fin2 = Fin(
    Vector3(-0.2, 0.0, 0.1), 
    dragFunctionFin, 
    liftFunctionFin, 
    pressureFunction,
    finArea, 
    90 * np.pi/180
)

fin3 = Fin(
    Vector3(-0.2, 0.0, -0.1), 
    dragFunctionFin, 
    liftFunctionFin, 
    pressureFunction,
    finArea, 
    -90 * np.pi/180
)

bodyAero: RocketBody = RocketBody(
    position=Vector3(0.0, 0.0, 0.0),
    dragFunction=liftFunctionBody,
    liftFunction=dragFunctionBody,
    areaFunction=areaFunctionBody,
    presFunction=pressureFunction
)

state: RigidBody = RigidBody(
    1.0, 
    Vector3(0.1, 0.1, 0.1), 
    Vector3(0.0, 0.0, 0.0), 
    Vector3(50.0, 0.0, 0.0), 
    Quaternion.fromEulerAngles(
        Vector3(
            0.0, 
            0.0, 
            0.0
        ) * np.pi/180
    ),
    Vector3(0.0, 0.0, 0.0)
)

fin0.setAngle((1 + -4) * np.pi/180)
fin1.setAngle((1 +  4) * np.pi/180)
fin2.setAngle((1 + 0 ) * np.pi/180)
fin3.setAngle((1 + 0 ) * np.pi/180)

fins = [fin0, fin1, fin2, fin3]

cp_location_body = _copy_v3(bodyAero.position)

# Plot/animation sampling rate (points per second of simulation time).
# Set lower to keep rendering fast when dt is small.
points_per_second = 30

t_sim = 0.0
t_end = 5.0
dt = 0.001
steps = int(np.ceil(t_end / dt))
rot_history: list[Quaternion] = []
body_lift_history: list[Vector3] = []
body_drag_history: list[Vector3] = []

print(state.inertia)

for _ in range(steps):
    rot_history.append(_copy_q(state.rotation))

    bodyAero.update(state, t_sim)
    body_lift_history.append(_copy_v3(bodyAero.getLift()))
    body_drag_history.append(_copy_v3(bodyAero.getDrag()))

    body_r_world = state.rotation.rotate(cp_location_body)
    # print(bodyAero.getForce())
    if not any([np.isnan(x) for x in iter(bodyAero.getForce())]):
        state.applyForce(bodyAero.getForce(), body_r_world)
    else:
        print("invalid!")
    if t_sim < 2.0:
        state.applyLocalForce(Vector3(50.0, 0, 0), Vector3(0, 0, 0))

    for fin in fins:
        fin.update(state, t_sim)
        fin_force_world = fin.getForce()
        # RigidBody.applyForce expects a lever arm (relative to COM) in world frame,
        # not an absolute world position.
        fin_r_world = state.rotation.rotate(fin.position)
        state.applyForce(fin_force_world, fin_r_world)

    state.update(dt)
    t_sim += dt

    # Stop cleanly if the sim goes unstable.
    if not np.all(np.isfinite(_v3_to_np(state.position))):
        break
    if not np.all(np.isfinite(_v3_to_np(state.velocity))):
        break
    if not np.all(np.isfinite(_v3_to_np(state.rotVel))):
        break


lift_hist = [fin.liftForces for fin in fins]
drag_hist = [fin.dragForces for fin in fins]


def _downsample_indices(n: int, duration_s: float, pps: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if pps <= 0:
        raise ValueError("points_per_second must be > 0")
    target = int(np.ceil(max(duration_s, 0.0) * pps))
    target = max(1, min(target, n))
    idx = np.linspace(0, n - 1, target)
    idx = np.unique(np.round(idx).astype(int))
    return idx


# Trim all recorded arrays to `points_per_second` samples.
n_raw = len(rot_history)
duration_s = n_raw * dt
idx = _downsample_indices(n_raw, duration_s, points_per_second)

rot_history = [rot_history[i] for i in idx]
for fin_i in range(len(fins)):
    lift_hist[fin_i] = [lift_hist[fin_i][i] for i in idx if i < len(lift_hist[fin_i])]
    drag_hist[fin_i] = [drag_hist[fin_i][i] for i in idx if i < len(drag_hist[fin_i])]

body_lift_history = [body_lift_history[i] for i in idx if i < len(body_lift_history)]
body_drag_history = [body_drag_history[i] for i in idx if i < len(body_drag_history)]

n_frames = len(rot_history)
for i in range(len(fins)):
    n_frames = min(n_frames, len(lift_hist[i]), len(drag_hist[i]))

n_frames = min(n_frames, len(body_lift_history), len(body_drag_history))

max_force = 0.0
for forces in (lift_hist + drag_hist):
    for f in forces:
        max_force = max(max_force, abs(f))
for f in body_lift_history:
    max_force = max(max_force, abs(f))
for f in body_drag_history:
    max_force = max(max_force, abs(f))
force_scale = 1.0 if max_force == 0 else (0.25 / max_force)

# Length of the plotted direction axes (chord/root) in meters.
dir_scale = 0.12


rocket_length = 0.6

fig = plt.figure()
ax: Axes3D = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Fin Lift (red) and Drag (blue) Vectors')

body_line, = ax.plot([], [], [], color='k', linewidth=2)
fin_pts, = ax.plot([], [], [], 'ko', markersize=4)

_quivers: list = []


def _set_equal_aspect(ax: Axes3D, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

fin_positions_body = np.array([_v3_to_np(fin.position) for fin in fins], dtype=float)
fin_extent = float(np.max(np.linalg.norm(fin_positions_body, axis=1))) if len(fin_positions_body) else 0.0

center = np.zeros(3)
# Keep a stable view around the rocket body and arrows (force_scale makes arrows ~0.25 long).
radius = max(rocket_length * 0.75, fin_extent + 0.35 + dir_scale)
_set_equal_aspect(ax, center, radius)


def init():
    body_line.set_data([], [])
    body_line.set_3d_properties([])
    fin_pts.set_data([], [])
    fin_pts.set_3d_properties([])
    return [body_line, fin_pts]


def update(frame_idx: int):
    global _quivers
    for q in _quivers:
        try:
            q.remove()
        except Exception:
            pass
    _quivers = []

    rot = rot_history[frame_idx]

    # Rocket COM stays fixed at the origin, but the rocket (and attached forces)
    # rotate in the view based on the simulated orientation.
    p0 = _v3_to_np(rot.rotate(Vector3(-rocket_length * 0.5, 0.0, 0.0)))
    p1 = _v3_to_np(rot.rotate(Vector3( rocket_length * 0.5, 0.0, 0.0)))
    body_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
    body_line.set_3d_properties([p0[2], p1[2]])

    fin_world_positions = np.array(
        [_v3_to_np(rot.rotate(fin.position)) for fin in fins],
        dtype=float,
    )
    fin_pts.set_data(fin_world_positions[:, 0], fin_world_positions[:, 1])
    fin_pts.set_3d_properties(fin_world_positions[:, 2])

    for fin_i, fin_pos in enumerate(fin_world_positions):
        lift = _v3_to_np(lift_hist[fin_i][frame_idx]) * force_scale
        drag = _v3_to_np(drag_hist[fin_i][frame_idx]) * force_scale

        # Fin chord and root/span directions (rotate with rocket + fin orientation).
        fin_rot_body = fins[fin_i].getRotationBody()
        chord_dir = rot.rotate(fin_rot_body.rotate(Vector3(1.0, 0.0, 0.0))).norm()
        root_dir = rot.rotate(fin_rot_body.rotate(Vector3(0.0, 1.0, 0.0))).norm()

        chord_tip = fin_pos + _v3_to_np(chord_dir) * dir_scale
        root_tip = fin_pos + _v3_to_np(root_dir) * dir_scale

        lift_tip = fin_pos + lift
        drag_tip = fin_pos + drag

        (line_lift,) = ax.plot(
            [fin_pos[0], lift_tip[0]],
            [fin_pos[1], lift_tip[1]],
            [fin_pos[2], lift_tip[2]],
            color='r', linewidth=1.5,
        )
        (line_drag,) = ax.plot(
            [fin_pos[0], drag_tip[0]],
            [fin_pos[1], drag_tip[1]],
            [fin_pos[2], drag_tip[2]],
            color='b', linewidth=1.5,
        )

        (line_chord,) = ax.plot(
            [fin_pos[0], chord_tip[0]],
            [fin_pos[1], chord_tip[1]],
            [fin_pos[2], chord_tip[2]],
            color='tab:orange', linewidth=1.0, linestyle='--',
        )
        (line_root,) = ax.plot(
            [fin_pos[0], root_tip[0]],
            [fin_pos[1], root_tip[1]],
            [fin_pos[2], root_tip[2]],
            color='tab:green', linewidth=1.0, linestyle='--',
        )

        _quivers.extend([line_lift, line_drag, line_chord, line_root])

    # Body aerodynamic lift/drag plotted at the CP location.
    cp_world = _v3_to_np(rot.rotate(cp_location_body))
    body_lift = _v3_to_np(body_lift_history[frame_idx]) * force_scale
    body_drag = _v3_to_np(body_drag_history[frame_idx]) * force_scale

    body_lift_tip = cp_world + body_lift
    body_drag_tip = cp_world + body_drag

    (line_body_lift,) = ax.plot(
        [cp_world[0], body_lift_tip[0]],
        [cp_world[1], body_lift_tip[1]],
        [cp_world[2], body_lift_tip[2]],
        color='r', linewidth=2.0,
    )
    (line_body_drag,) = ax.plot(
        [cp_world[0], body_drag_tip[0]],
        [cp_world[1], body_drag_tip[1]],
        [cp_world[2], body_drag_tip[2]],
        color='b', linewidth=2.0,
    )

    _quivers.extend([line_body_lift, line_body_drag])

    return [body_line, fin_pts] + _quivers


ani = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    interval=1000 / points_per_second,
    blit=False,
)

# plt.ion()
plt.show()
# plt.pause(0.01)
# --- Save animation to a video file (requires ffmpeg installed) ---
import matplotlib.animation as animation

output_path = "fin_forces.mp4"
fps = int(points_per_second)

try:
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
    ani.save(output_path, writer=writer, dpi=200)
    print(f"Saved video to: {output_path}")
except Exception as e:
    print(f"Failed to save MP4 via ffmpeg ({e}). Trying GIF instead...")
    gif_path = "fin_forces.gif"
    ani.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=150)
    print(f"Saved GIF to: {gif_path}")
