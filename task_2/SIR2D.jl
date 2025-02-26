using DifferentialEquations
using Plots
using LinearAlgebra
using SparseArrays
using LinearSolve


struct SIR
    """Model to solve"""
    M::Int
    S0::Matrix{Float64}
    I0::Matrix{Float64}
    tspan::Tuple{Float64, Float64}
    beta
    gamma
    mu_I::Float64
    mu_S::Float64
    k::Float64
end

#Constructor for when k=0
SIR(M, S0, I0, tspan, beta, gamma, mu_I, mu_S) = SIR(M, S0, I0, tspan, beta, gamma, mu_I, mu_S, 0.01)

function laplacian(M::Int)
    """"Discrete Laplacian (2D)"""
    Mi = M - 1
    Mi2 = (M-1)^2
    B = sparse(SymTridiagonal(-4*ones(Mi), ones(Mi-1)))
    A = sparse(kron(I(Mi), B))
    c = spdiagm(-Mi => ones(Mi2 - Mi), Mi => ones(Mi2 - Mi))
    A += c
    return A
end

function grid(M::Int)
    """Helper function; equivalent to 'np.meshgrid(np.linspace(0,1,M),np.linspace(0,1,M))'  """
    r = range(0,1, length=M)
    X = reshape(r, :, 1) 
    Y = reshape(r, 1, :) 
    return X .+ zero(Y), Y .+ zero(X)
end


function RHS_SIR(dU, U, p, t)
    """Helper function for SIR_2D. Differential equations for SIR-model"""
    beta, gamma, mu_I, mu_S, h, M, A = p  # Parameters
                           
    # Extract S and I from U
    S = U[1:M^2]
    I = U[M^2+1:end]

    # Apply the PDE terms
    dU[1:M^2] .= .- beta .* I .* S + (mu_S/(h^2)) .* (A * S)
    dU[M^2+1:end] .= beta .* I .* S - gamma .* I + (mu_I/(h^2)) .* (A * I)
end


function SIR_2D(M::Int, S0, I0, tspan, beta, gamma, mu_I, mu_S)
    """Solve the SIR-model differential equations in two dimensions
    
        INPUT:

        OUTPUT:
    t [Array]: Time intervals
    S [Array]: Suceptible at each timestamp
    I [Array]: Infected at each timestamp
    R [Array]: Recovered at each timestamp

    """
    @assert size(I0) == size(S0)
    @assert size(I0) == (M,M)

    #Check that beta, gamma format are ok
    if isa(beta, AbstractArray)
        @assert size(beta) == (M, M) "If beta is an array, it must have size (M, M)"
        beta = vec(beta)
    elseif isa(beta, Number)
        # No assertion needed for scalars
    else
        error("beta must be either a scalar or a (M, M) array")
    end

    if isa(gamma, AbstractArray)
        @assert size(gamma) == (M, M) "If gamma is an array, it must have size (M, M)"
        gamma = vec(gamma)
    elseif isa(gamma, Number)
        # No assertion needed for scalars
    else
        error("gamma must be either a scalar or a (M, M) array")
    end

    #Flatten IC-arrays
    I0 = vec(I0)
    S0 = vec(S0)

    #Create Laplacian, and define parameter arrays
    h = 1/(M+1)
    A = laplacian(M+1)
    p = [beta, gamma, mu_I, mu_S, h, M, A]

    U0 = vcat(S0, I0) 

    #Solve ODE
    prob = ODEProblem(RHS_SIR, U0, tspan, p)
    sol = solve(prob, AutoTsit5(Rosenbrock23()))

    #Separate results into 'Susceptible' and 'Infected'
    num_steps = length(sol.u)
    S_vec = [sol.u[i][1:M^2] for i in 1:num_steps]   # Collect S over time
    I_vec = [sol.u[i][M^2+1:end] for i in 1:num_steps] # Collect I over time

    #Reshape vectors into 2-dimensional arrays
    S_matrix = [reshape(S_matrix, M, M) for S_matrix in S_vec]
    I_matrix = [reshape(I_matrix, M, M) for I_matrix in I_vec]

    #Compute 'Recovered' using S + I + R = 1
    R_matrix = [1 .- (S_mat .+ I_mat) for (S_mat, I_mat) in zip(S_matrix, I_matrix)]

    return sol.t,  S_matrix, I_matrix, R_matrix
end

function CN_step(Sn, In, p)
    A, h, k, beta, gamma, r_S, r_I = p

    S_LHS = I - r_S/2 .* A
    I_LHS = I - r_I/2 .* A

    S_RHS = (I + r_S/2 .* A)*Sn - k .* beta .* In .* Sn
    I_RHS = (I + r_I/2 .* A)*In + beta .* In .* Sn - gamma .* In

    S_star = solve(LinearProblem(S_LHS,S_RHS)).u
    I_star = solve(LinearProblem(I_LHS,I_RHS)).u

    S_next = S_star - k * beta .* (I_star .* S_star + In .* Sn)
    I_next = I_star + k * beta .* (I_star .* S_star + In .* Sn) - gamma .* (I_star - In)


    return S_next, I_next
end


function SIR_2D_CN(M::Int, k, S0, I0, tspan, beta, gamma, mu_I, mu_S)
    """Solve the SIR-model differential equations in two dimensions using Crank-Nicholson
    
        INPUT:

        OUTPUT:
    t [Array]: Time intervals
    S [Array]: Suceptible at each timestamp
    I [Array]: Infected at each timestamp
    R [Array]: Recovered at each timestamp

    """
    @assert size(I0) == size(S0)
    @assert size(I0) == (M,M)

    #Check that beta, gamma format are ok
    if isa(beta, AbstractArray)
        @assert size(beta) == (M, M) "If beta is an array, it must have size (M, M)"
        beta = vec(beta)
    elseif isa(beta, Number)
        # No assertion needed for scalars
    else
        error("beta must be either a scalar or a (M, M) array")
    end

    if isa(gamma, AbstractArray)
        @assert size(gamma) == (M, M) "If gamma is an array, it must have size (M, M)"
        gamma = vec(gamma)
    elseif isa(gamma, Number)
        # No assertion needed for scalars
    else
        error("gamma must be either a scalar or a (M, M) array")
    end

    #Flatten IC-arrays
    I0 = vec(I0)
    S0 = vec(S0)

    #Create Laplacian, and define parameter arrays
    h = 1/(M+1)
    A = laplacian(M+1)

    r_S = mu_S *k/(h^2)
    r_I = mu_I *k/(h^2)

    p = [A, h, k, beta, gamma, r_S, r_I]

    t = tspan[1]:k:tspan[2]

   
    #solve ODE
    S_arr = zeros(length(t), M^2)
    I_arr = zeros(length(t), M^2)

    S_arr[1, :] = S0
    I_arr[1, :] = I0
   
    for i in 2:length(t)
        S_arr[i, :], I_arr[i, :] = CN_step(S_arr[i-1, :], I_arr[i-1, :], p)
    end


    #Reshape vectors into 2-dimensional arrays
    S_matrix = [reshape(S_arr[i, :], M, M) for i in 1:size(S_arr, 1)]
    I_matrix = [reshape(I_arr[i, :], M, M) for i in 1:size(I_arr, 1)]

    #Compute 'Recovered' using S + I + R = 1
    R_matrix = [1 .- (S_mat .+ I_mat) for (S_mat, I_mat) in zip(S_matrix, I_matrix)]

    return t,  S_matrix, I_matrix, R_matrix
end


function solvePDE(m::SIR, method="")
    """Wrapper for solving PDE using SIR-struct"""
    if method=="CN"
        return SIR_2D_CN(m.M, m.k, m.S0, m.I0, m.tspan, m.beta, m.gamma, m.mu_I, m.mu_S)
    else
        return SIR_2D(m.M, m.S0, m.I0, m.tspan, m.beta, m.gamma, m.mu_I, m.mu_S)
    end
end

function manhattan_beta(M::Int, beta_road::Float64, beta_block::Float64, road_spacing::Int)
    """Create Manhattan-grid like structure, with different values of beta for "roads" and "blocks" """
    beta = fill(beta_block, M, M)

    # Assign beta to roads
    for i in 1:M
        if mod(i, road_spacing) == 0
            beta[i, :] .= beta_road  # Horizontal road
            beta[:, i] .= beta_road  # Vertical road
        end
    end
    
    return beta
end

function Animate(M::Int, t, Sm, Im, Rm, suptitle="", FPS=10)
    """Function to animate SRI model

        INPUT:
    M [int] 

        OUTPUT:
    None
    
    """
    r = range(0,1,length=M)

    gr()

    time_diffs = diff(t)
    # Normalize time differences to determine frame repetition
    min_time_diff = minimum(time_diffs)
    frame_counts = round.(Int, time_diffs ./ min_time_diff)

    # Create side-by-side animation
    anim = @gif for i in 1:length(t)-1
        for _ in 1:frame_counts[i]
            p1 = heatmap(r, r, Sm[i]', title="Susceptible", colorbar=false, clim=(0,1), titlefontsize=10)
            p2 = heatmap(r, r, Im[i]', title="Infected", colorbar=false, clim=(0,1), titlefontsize=10)
            p3 = heatmap(r, r, Rm[i]', title="Recovered", colorbar=true, clim=(0,1), titlefontsize=10, colorbar_title="Proportion of Population")
            
            plot(p1, p2, p3, layout=(1, 3), size=(900, 300), 
            plot_title=suptitle,
            plot_titlefontsize=12,
            bottom_margin=5Plots.px,
            top_margin = 20Plots.px,
            subplot_padding=5)
        end
    end fps=FPS  # Adjust frame rate if needed


end 

"""
M = 30
S0 = ones(M, M) * 0.99  # 99% susceptible
I0 = zeros(M, M)
I0[2, 2] = 0.01       # Small initial infection

tspan = (0.0, 5.0)
beta = 3
gamma = 1
mu_I = 0.01
mu_S = 0.01

t, Sm, Im, Rm = SIR_2D(M, S0, I0, tspan, beta, gamma, mu_I, mu_S)
Animate(M, t, Sm, Im, Rm, "SIR-model")
"""

