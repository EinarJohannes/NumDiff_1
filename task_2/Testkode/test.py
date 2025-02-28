import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
beta = 0.8
gamma = 0.01
mu_S = 0.3
mu_I = 0.3
Lx, Ly = 1.0, 1.0
Nx, Ny = 50, 50
hx, hy = Lx / (Nx - 1), Ly / (Ny - 1)
dt = 0.01
T = 2.0  # Total time
Nt = int(T / dt)

# Initial conditions
S = np.ones((Nx, Ny))
I = np.zeros((Nx, Ny))
I[Nx//2, Ny//2] = 1.0  # Initial infection at the center

# Function to compute the Laplacian
def laplacian(U):
    return (np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0) +
            np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1) -
            4 * U) / (hx * hy)

# Storage for results
S_solution = np.zeros((Nt, Nx, Ny))
I_solution = np.zeros((Nt, Nx, Ny))
R_solution = np.zeros((Nt, Nx, Ny))

# Time-stepping loop
for t in range(Nt):
    # Store the current state
    S_solution[t, :, :] = S
    I_solution[t, :, :] = I
    R_solution[t, :, :] = 1 - S - I  # Ensure S + I + R = 1
    
    # Compute the Laplacian
    lap_S = laplacian(S)
    lap_I = laplacian(I)
    
    # Update S and I using the discretized equations
    S += dt * (-beta * I * S + mu_S * lap_S)
    I += dt * (beta * I * S - gamma * I + mu_I * lap_I)

# Print shape of solutions:
print(S_solution.shape)
print(I_solution.shape)
print(R_solution.shape)

# Animation of infected, susceptible, and recovered over time
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cax_S = axes[0].imshow(S_solution[0, :, :], cmap='Blues', interpolation='nearest')
fig.colorbar(cax_S, ax=axes[0], label='Susceptible Population')
axes[0].set_title('Susceptible Population')

cax_I = axes[1].imshow(I_solution[0, :, :], cmap='Reds', interpolation='nearest')
fig.colorbar(cax_I, ax=axes[1], label='Infected Population')
axes[1].set_title('Infected Population')

cax_R = axes[2].imshow(R_solution[0, :, :], cmap='Greens', interpolation='nearest')
fig.colorbar(cax_R, ax=axes[2], label='Recovered Population')
axes[2].set_title('Recovered Population')

def update(frame):
    cax_S.set_array(S_solution[frame, :, :])
    cax_I.set_array(I_solution[frame, :, :])
    cax_R.set_array(R_solution[frame, :, :])
    axes[0].set_title(f'Susceptible Population at Time {frame*dt:.2f}')
    axes[1].set_title(f'Infected Population at Time {frame*dt:.2f}')
    axes[2].set_title(f'Recovered Population at Time {frame*dt:.2f}')
    return cax_S, cax_I, cax_R

ani = animation.FuncAnimation(fig, update, frames=Nt, interval=300, blit=True)

plt.show()

#CHecking for largest value of integer in S_solution
print(np.max(S_solution))