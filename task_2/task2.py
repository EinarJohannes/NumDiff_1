import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.integrate import solve_ivp
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
    Mi = M - 1
    diagonals = [-4 * np.ones(Mi), np.ones(Mi-1), np.ones(Mi-1)]
    offsets = [0, -1, 1]
    B = diags(diagonals, offsets).toarray()
    A = kron(eye(Mi), B) + diags([np.ones(Mi-1)], [-Mi]).toarray() + diags([np.ones(Mi-1)], [Mi]).toarray()
    return A

def grid(M):
    r = np.linspace(0, 1, M)
    X, Y = np.meshgrid(r, r)
    return X, Y


def RHS_SIR(t, U, p):
    beta, gamma, mu_I, mu_S, h, M, A = p
    S = U[:M**2]
    I = U[M**2:]
    dSdt = -beta * I * S + (mu_S / h**2) * (A @ S)
    dIdt = beta * I * S - gamma * I + (mu_I / h**2) * (A @ I)
    return np.concatenate((dSdt, dIdt))

def SIR_2D(M, S0, I0, tspan, beta, gamma, mu_I, mu_S):
    h = 1 / (M + 1)
    A = laplacian(M + 1)
    p = [beta, gamma, mu_I, mu_S, h, M, A]
    U0 = np.concatenate((S0.flatten(), I0.flatten()))

    sol = solve_ivp(RHS_SIR, tspan, U0, args=(p,), method='RK45', t_eval=np.linspace(tspan[0], tspan[1], 500))
    
    S_matrix = sol.y[:M**2].T.reshape(-1, M, M)
    I_matrix = sol.y[M**2:].T.reshape(-1, M, M)
    R_matrix = 1 - S_matrix - I_matrix
    
    return sol.t, S_matrix, I_matrix, R_matrix

def CN_step(Sn, In, p):
    A, h, k, beta, gamma, r_S, r_I = p

    S_LHS = np.eye(len(Sn)) - (r_S / 2) * A
    I_LHS = np.eye(len(In)) - (r_I / 2) * A

    S_RHS = (np.eye(len(Sn)) + (r_S / 2) * A) @ Sn - k * beta * In * Sn
    I_RHS = (np.eye(len(In)) + (r_I / 2) * A) @ In + k * beta * In * Sn - k * gamma * In

    S_star = np.linalg.solve(S_LHS, S_RHS)
    I_star = np.linalg.solve(I_LHS, I_RHS)

    S_next = S_star - k * beta * (I_star * S_star + In * Sn)
    I_next = I_star + k * beta * (I_star * S_star + In * Sn) - k * gamma * I_star

    return S_next, I_next

def SIR_2D_CN(M, k, S0, I0, tspan, beta, gamma, mu_I, mu_S):
    h = 1 / (M + 1)
    A = laplacian(M + 1)
    r_S = mu_S * k / (h**2)
    r_I = mu_I * k / (h**2)

    p = [A, h, k, beta, gamma, r_S, r_I]
    t = np.arange(tspan[0], tspan[1], k)

    S_arr = np.zeros((len(t), M**2))
    I_arr = np.zeros((len(t), M**2))

    S_arr[0, :] = S0.flatten()
    I_arr[0, :] = I0.flatten()

    for i in range(1, len(t)):
        S_arr[i, :], I_arr[i, :] = CN_step(S_arr[i-1, :], I_arr[i-1, :], p)

    S_matrix = [S_arr[i, :].reshape(M, M) for i in range(S_arr.shape[0])]
    I_matrix = [I_arr[i, :].reshape(M, M) for i in range(I_arr.shape[0])]
    R_matrix = [1 - (S + I) for S, I in zip(S_matrix, I_matrix)]

    return t, S_matrix, I_matrix, R_matrix


def solvePDE(model, method=""):
    if method == "CN":
        return SIR_2D_CN(model.M, model.k, model.S0, model.I0, model.tspan, model.beta, model.gamma, model.mu_I, model.mu_S)
    else:
        return SIR_2D(model.M, model.S0, model.I0, model.tspan, model.beta, model.gamma, model.mu_I, model.mu_S)



def manhattan_beta(M, beta_road, beta_block, road_spacing):
    beta = np.full((M, M), beta_block)
    for i in range(M):
        if i % road_spacing == 0:
            beta[i, :] = beta_road
            beta[:, i] = beta_road
    return beta


def Animate(M, t, Sm, Im, Rm, suptitle="", FPS=10):
    r = np.linspace(0, 1, M)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def update(frame):
        for ax, data, title in zip(axes, [Sm, Im, Rm], ["Susceptible", "Infected", "Recovered"]):
            ax.clear()
            ax.set_title(title)
            heatmap = ax.imshow(data[frame], cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        fig.suptitle(suptitle)
        return heatmap,

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=1000/FPS, blit=False)
    plt.show()

    return ani


if __name__ == "__main__":
    M = 30
    S0 = np.full((M, M), 0.99)
    I0 = np.zeros((M, M))
    I0[2, 2] = 0.01

    tspan = (0.0, 5.0)
    beta = 3
    gamma = 1
    mu_I = 0.01
    mu_S = 0.01

    t, Sm, Im, Rm = SIR_2D(M, S0, I0, tspan, beta, gamma, mu_I, mu_S)
    Animate(M, t, Sm, Im, Rm, "SIR-model")
