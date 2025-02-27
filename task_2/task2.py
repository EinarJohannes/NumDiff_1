import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.8
gamma = 0.1
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

# Time-stepping loop
for t in range(Nt):
    # Store the current state
    S_solution[t, :, :] = S
    I_solution[t, :, :] = I
    
    # Compute the Laplacian
    lap_S = laplacian(S)
    lap_I = laplacian(I)
    
    # Update S and I using the discretized equations
    S += dt * (-beta * I * S + mu_S * lap_S)
    I += dt * (beta * I * S - gamma * I + mu_I * lap_I)

"""
# Visualization of results at the final time point
plt.imshow(S_solution[-1, :, :], cmap='Blues', interpolation='nearest')
plt.colorbar(label='Susceptible Population')
plt.title('Susceptible Population at Final Time')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.imshow(I_solution[-1, :, :], cmap='Reds', interpolation='nearest')
plt.colorbar(label='Infected Population')
plt.title('Infected Population at Final Time')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""

#Print shape of solutions:
print(S_solution.shape)
print(I_solution.shape)

#Visualising infected over time by animation. Color plot or such that changes over time.
#Animation of infected over time
import matplotlib.animation as animation

# Animation of infected over time
fig, ax = plt.subplots()
cax = ax.imshow(I_solution[0, :, :], cmap='Reds', interpolation='nearest')
fig.colorbar(cax, label='Infected Population')

def update(frame):
    cax.set_array(I_solution[frame, :, :])
    ax.set_title(f'Infected Population at Time {frame*dt:.2f}')
    return cax,

ani = animation.FuncAnimation(fig, update, frames=Nt, interval=300, blit=True)

plt.show()