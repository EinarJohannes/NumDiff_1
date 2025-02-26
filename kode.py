import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
beta = 3
gamma = 1
mu_S = 0.01
mu_I = 0.01
Nx, Ny = 50, 50
T = 10
dt = 0.01
dx = 1
dy = 1

# Initialize grids
S = np.ones((Nx, Ny))
I = np.zeros((Nx, Ny))
R = np.zeros((Nx, Ny))
# Initial condition: small infection in the center
I[Nx//2, Ny//2] = 0.01
S -= I

def laplacian(U):
    return (np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0) - 2 * U) / dx**2 + \
           (np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1) - 2 * U) / dy**2

# Create a figure for plotting
fig, ax = plt.subplots()
im = ax.imshow(I, cmap='hot', interpolation='nearest')

# Time-stepping loop
frames = []
for _ in range(int(T / dt)):
    S_new = S + dt * (-beta * S * I + mu_S * laplacian(S))
    I_new = I + dt * (beta * S * I - gamma * I + mu_I * laplacian(I))
    S, I = S_new, I_new
    R = 1 - S - I  # since S + I + R = 1
    frames.append((I.copy(),))

def update(frame):
    im.set_data(frame[0])
    return [im]

# Animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)
plt.colorbar(im, ax=ax)
plt.show()
