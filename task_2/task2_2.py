import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SIR:
    def __init__(self, M, S0, I0, tspan, beta, gamma, mu_I, mu_S, k=0.01):
        self.M = M
        self.S0 = S0
        self.I0 = I0
        self.tspan = tspan
        self.beta = beta
        self.gamma = gamma
        self.mu_I = mu_I
        self.mu_S = mu_S
        self.k = k

def laplacian(M):
    """Discrete Laplacian (2D)"""
    Mi = M - 1
    Mi2 = (M-1)**2
    B = diags([-4*np.ones(Mi), np.ones(Mi-1), np.ones(Mi-1)], [0, -1, 1])
    A = kron(eye(Mi), B)
    c = diags([np.ones(Mi2 - Mi), np.ones(Mi2 - Mi)], [-Mi, Mi])
    A += c
    return A

def RHS_SIR(t, U, p):
    """Helper function for SIR_2D. Differential equations for SIR-model"""
    beta, gamma, mu_I, mu_S, h, M, A = p
    S = U[:M**2]
    I = U[M**2:]
    
    dS = -beta * I * S + (mu_S / h**2) * (A @ S)
    dI = beta * I * S - gamma * I + (mu_I / h**2) * (A @ I)
    
    return np.concatenate([dS, dI])

def SIR_2D(M, S0, I0, tspan, beta, gamma, mu_I, mu_S):
    """Solve the SIR-model differential equations in two dimensions"""
    assert S0.shape == I0.shape == (M, M)
    
    if isinstance(beta, np.ndarray):
        assert beta.shape == (M, M)
        beta = beta.flatten()
    elif isinstance(beta, (int, float)):
        pass
    else:
        raise ValueError("beta must be either a scalar or a (M, M) array")
    
    if isinstance(gamma, np.ndarray):
        assert gamma.shape == (M, M)
        gamma = gamma.flatten()
    elif isinstance(gamma, (int, float)):
        pass
    else:
        raise ValueError("gamma must be either a scalar or a (M, M) array")
    
    I0 = I0.flatten()
    S0 = S0.flatten()
    
    h = 1 / (M + 1)
    A = laplacian(M + 1)
    p = [beta, gamma, mu_I, mu_S, h, M, A]
    
    U0 = np.concatenate([S0, I0])
    
    sol = solve_ivp(RHS_SIR, tspan, U0, args=(p,), method='RK45')
    
    num_steps = len(sol.t)
    S_vec = [sol.y[:M**2, i] for i in range(num_steps)]
    I_vec = [sol.y[M**2:, i] for i in range(num_steps)]
    
    S_matrix = [S.reshape(M, M) for S in S_vec]
    I_matrix = [I.reshape(M, M) for I in I_vec]
    
    R_matrix = [1 - (S + I) for S, I in zip(S_matrix, I_matrix)]
    
    return sol.t, S_matrix, I_matrix, R_matrix

def animate_SIR(M, t, Sm, Im, Rm, suptitle="", FPS=10):
    """Function to animate SIR model"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    r = np.linspace(0, 1, M)
    
    def update(frame):
        for ax in axes:
            ax.clear()
        axes[0].imshow(Sm[frame], extent=[0, 1, 0, 1], vmin=0, vmax=1)
        axes[0].set_title("Susceptible")
        axes[1].imshow(Im[frame], extent=[0, 1, 0, 1], vmin=0, vmax=1)
        axes[1].set_title("Infected")
        axes[2].imshow(Rm[frame], extent=[0, 1, 0, 1], vmin=0, vmax=1)
        axes[2].set_title("Recovered")
        fig.suptitle(suptitle)
    
    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=1000/FPS)
    plt.show()
    
# Example usage
M = 30
S0 = np.ones((M, M)) * 0.99  # 99% susceptible
I0 = np.zeros((M, M))
I0[2, 2] = 0.1  # Øk initial infeksjon
I0[15, 15] = 0.1  # Legg til en annen infeksjon

tspan = (0.0, 5.0)
beta = 3
gamma = 1
mu_I = 0.01
mu_S = 0.01

t, Sm, Im, Rm = SIR_2D(M, S0, I0, tspan, beta, gamma, mu_I, mu_S)
animate_SIR(M, t, Sm, Im, Rm, "SIR-model")

"""
# Example usage
M = 30
S0 = np.ones((M, M)) * 0.99  # 99% susceptible
I0 = np.zeros((M, M))
I0[2, 2] = 0.01  # Small initial infection

tspan = (0.0, 5.0)
beta = 3
gamma = 1
mu_I = 0.01
mu_S = 0.01

t, Sm, Im, Rm = SIR_2D(M, S0, I0, tspan, beta, gamma, mu_I, mu_S)
animate_SIR(M, t, Sm, Im, Rm, "SIR-model")
"""