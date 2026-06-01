module DeePCUtils

using LinearAlgebra, Statistics, FFTW

export DiscreteSystem, step!, reset!
export build_data_matrix, hankellize
export estimate_SP_hinf

# Discrete-time LTI system  x[t+1] = A x[t] + B u[t],  y[t] = C x[t] + D u[t]
mutable struct DiscreteSystem
    A::Matrix{Float64}; B::Matrix{Float64}; C::Matrix{Float64}; D::Matrix{Float64}
    x::Vector{Float64}
    u_hist::Union{Nothing, Matrix{Float64}}
    y_hist::Union{Nothing, Matrix{Float64}}
    noise_std::Float64

    function DiscreteSystem(A, B, C, D; noise_std=0.0, x0=nothing)
        nx = size(A, 1)
        new(A, B, C, D, isnothing(x0) ? zeros(nx) : x0, nothing, nothing, noise_std)
    end
end

# Advance one step; accumulate (u, y) history.
function step!(sys::DiscreteSystem, u_in::Matrix{Float64})
    y_pure = (sys.C * sys.x + sys.D * u_in')'
    y_noisy = y_pure .+ sys.noise_std * randn(size(y_pure))
    sys.x = sys.A * sys.x + vec(sys.B * u_in')
    sys.u_hist = isnothing(sys.u_hist) ? u_in : vcat(sys.u_hist, u_in)
    sys.y_hist = isnothing(sys.y_hist) ? y_noisy : vcat(sys.y_hist, y_noisy)
    return y_noisy
end

# Reset state and clear history; optionally set x0.
function reset!(sys::DiscreteSystem; x0=nothing)
    sys.x = isnothing(x0) ? zeros(size(sys.A, 1)) : x0
    sys.u_hist, sys.y_hist = nothing, nothing
end

# Depth-L Hankel matrix from a multi-channel time series U ∈ ℝ^{m×T}.
# Returns H_L(U) ∈ ℝ^{mL × (T−L+1)}  (paper Eq. 432–441).
function hankellize(U::Matrix{Float64}, L::Int)
    m, T = size(U)
    H = zeros(m*L, T-L+1)
    for row = 1:L
        H[(row-1)*m+1 : row*m, :] = U[:, row : T-L+row]
    end
    return H
end

# Build DeePC data matrices (Up, Uf, Yp, Yf) from input U and output Y.
# Splits the depth-(Tini+N) Hankel matrix into past (Tini rows) and future (N rows) blocks.
function build_data_matrix(U::Matrix{Float64}, Y::Matrix{Float64}, Tini::Int, N::Int)
    Hu = hankellize(U, Tini + N)
    Hy = hankellize(Y, Tini + N)
    m, p = size(U, 1), size(Y, 1)
    return (Up=Hu[1:Tini*m, :], Uf=Hu[Tini*m+1:end, :], Yp=Hy[1:Tini*p, :], Yf=Hy[Tini*p+1:end, :])
end

# Estimate empirical peak gain Ĝ_∞ via the Empirical Transfer Function Estimate (ETFE).
# Used to set the Phase-1 stealth bound δ₁ ≤ (ε_ids − ε_ss) / Ĝ_∞  (paper Eq. 390–394).
function estimate_SP_hinf(y_phase1::Vector{Float64},
                          u_a_phase1::Vector{Float64},
                          y_ref::Float64)
    N     = length(y_phase1)
    y_c   = y_phase1 .- y_ref   # remove DC tracking component

    Y     = fft(y_c)
    U_a   = fft(u_a_phase1)

    tol      = 1e-8 * maximum(abs.(U_a))
    valid    = abs.(U_a) .> tol
    SP_mag   = abs.(Y[valid]) ./ abs.(U_a[valid])

    return maximum(SP_mag)
end

end
