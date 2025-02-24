import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


SIR_IV = [0.99,0.01,0]

T = 12 # Max time
h = 0.01 # Step length in time
beta = 4 # Transmission rate
gamma = 1 # Recovery rate

t_eval = np.arange(0, T, h) # Time vector: from 0 to T in steps of h

# Define the ODE system for SIR, made to fit the solve_ivp()-function
def SIR_ODE(t, y):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# Solve the ODE system
sol = solve_ivp(SIR_ODE, [0, T], SIR_IV, t_eval=t_eval)


# Plotting the results
plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], label="Susceptible")
plt.plot(sol.t, sol.y[1], label="Infected")
plt.plot(sol.t, sol.y[2], label="Recovered")
plt.xlabel("Time")
plt.ylabel("Population Fraction")
plt.title("SIR Model Dynamics")
plt.legend()
plt.grid(True)
plt.show()