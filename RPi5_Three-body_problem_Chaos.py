import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ==================================
# Three body simulation gravity dynamic by chaotic
# 1. Physical Constants
# ==================================
G = 1.0
dt = 0.001
steps = 40000

# Perturbation parameters
process_noise_strength = 0.05
measurement_noise_strength = 0.02
drag_coefficient = 0.0005
external_field_strength = 0.001

np.random.seed(42)

# ==================================
# 2. Initial Conditions
# ==================================
masses = np.array([1.0, 1.0, 1.0])

positions = np.array([
    [-1.0, 0.0],
    [ 1.0, 0.0],
    [ 0.0, 0.5]
], dtype=float)

velocities = np.array([
    [ 0.0, -0.3],
    [ 0.0,  0.3],
    [ 0.5,  0.0]
], dtype=float)

# ==================================
# 3. Acceleration Function
# ==================================
def compute_acceleration(pos, vel):

    acc = np.zeros_like(pos)

    # Mutual gravity
    for i in range(3):
        for j in range(3):
            if i != j:
                r = pos[j] - pos[i]
                dist = np.linalg.norm(r) + 1e-5
                acc[i] += G * masses[j] * r / dist**3

    # Weak external harmonic-like field
    acc -= external_field_strength * pos

    # Drag force
    acc -= drag_coefficient * vel

    # Stochastic perturbation (Euler-Maruyama style scaling)
    acc += process_noise_strength * np.sqrt(dt) * np.random.randn(3, 2)

    return acc

# ==================================
# 4. RK4 Integrator
# ==================================
def rk4_step(pos, vel, dt):

    def derivatives(p, v):
        return v, compute_acceleration(p, v)

    k1_v, k1_a = derivatives(pos, vel)
    k2_v, k2_a = derivatives(pos + 0.5*dt*k1_v,
                             vel + 0.5*dt*k1_a)
    k3_v, k3_a = derivatives(pos + 0.5*dt*k2_v,
                             vel + 0.5*dt*k2_a)
    k4_v, k4_a = derivatives(pos + dt*k3_v,
                             vel + dt*k3_a)

    pos_new = pos + dt/6*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    vel_new = vel + dt/6*(k1_a + 2*k2_a + 2*k3_a + k4_a)

    return pos_new, vel_new

# ==================================
# 5. Run Simulation
# ==================================
trajectory = np.zeros((steps, 3, 2))

print("Running three-body chaotic simulation...")

for i in range(steps):
    positions, velocities = rk4_step(positions, velocities, dt)
    trajectory[i] = positions

# Add measurement noise (observation effect)
measured_trajectory = trajectory + \
    measurement_noise_strength * np.random.randn(*trajectory.shape)

print("Simulation complete.")

# ==================================
# 6. Animation Setup
# ==================================
fig, ax = plt.subplots(figsize=(7,7))
ax.set_title("Three-Body Problem (Perturbed Chaotic System)")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(alpha=0.3)

lines = [ax.plot([], [], lw=1)[0] for _ in range(3)]
points = [ax.plot([], [], 'o')[0] for _ in range(3)]

def update(frame):
    for i in range(3):
        lines[i].set_data(
            measured_trajectory[:frame, i, 0],
            measured_trajectory[:frame, i, 1]
        )

        # FIX: wrap scalar in list
        points[i].set_data(
            [measured_trajectory[frame, i, 0]],
            [measured_trajectory[frame, i, 1]]
        )

    return lines + points

ani = FuncAnimation(
    fig,
    update,
    frames=np.arange(1, steps, 20),
    interval=1,
    blit=True
)

# ==================================
# 7. Save Animation
# ==================================
writer = FFMpegWriter(fps=60)
ani.save("three_body_realistic.mp4", writer=writer)
plt.close()

# ==================================
# 8. Save Final Trajectory Image
# ==================================
plt.figure(figsize=(7,7))
for i in range(3):
    plt.plot(measured_trajectory[:, i, 0],
             measured_trajectory[:, i, 1])

plt.title("Three-Body Problem - Realistic Chaotic Trajectories")
plt.gca().set_aspect('equal')
plt.grid(alpha=0.3)
plt.savefig("three_body_realistic.jpg", dpi=300)
plt.show()

print("MP4 and JPEG saved successfully.")
