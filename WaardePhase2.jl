module WaardePhase2

using LinearAlgebra
import Main.DeePCUtils: step!, DiscreteSystem

export hankellize_vec
export build_M_io, candidate_col_io, choose_u_waarde_siso
export online_rank_io

function hankellize_vec(v::Vector{Float64}, L::Int)
    T = length(v)
    @assert T >= L "Need at least L samples, got T=$T and L=$L"
    H = zeros(L, T - L + 1)
    for i in 1:L
        H[i, :] = v[i:T-L+i]
    end
    return H
end

"""
Matrix from Theorem 3 structure:
M_t = [ H_{L-1}(y[1:end-1]) ;
        H_L(u)             ]
Columns are aligned by truncating to common number of columns.
"""
function build_M_io(u_hist::Vector{Float64}, y_hist::Vector{Float64}, L::Int)
    @assert L >= 2
    @assert length(u_hist) >= L
    @assert length(y_hist) >= L - 1

    Hy = hankellize_vec(y_hist, L - 1)
    Hu = hankellize_vec(u_hist, L)

    ncols = min(size(Hy, 2), size(Hu, 2))
    return vcat(Hy[:, 1:ncols], Hu[:, 1:ncols])
end

"""
Candidate new column:
c_t = [ y[t-L+2 : t] ;
        u[t-L+2 : t+1] ]
In implementation, we use current histories and a trial u_new.
"""
function candidate_col_io(u_hist::Vector{Float64}, y_hist::Vector{Float64}, u_new::Float64, L::Int)
    @assert length(u_hist) >= L - 1
    @assert length(y_hist) >= L - 1

    y_block = y_hist[end-(L-2):end]             # length L-1
    u_block = vcat(u_hist[end-(L-2):end], u_new) # length L
    return vcat(y_block, u_block)
end

function online_rank_io(u_hist::Vector{Float64}, y_hist::Vector{Float64}, L::Int; atol=1e-9)
    M = build_M_io(u_hist, y_hist, L)
    return rank(M; atol=atol), M
end

"""
Returns:
u_id, mode, rank_before, residual_zero
mode ∈ (:arbitrary, :kernel_guided, :fallback_random)
"""
function choose_u_waarde_siso(
    u_hist::Vector{Float64},
    y_hist::Vector{Float64},
    L::Int;
    amp::Float64 = 0.1,
    atol::Float64 = 1e-9
)
    rank_before, M = online_rank_io(u_hist, y_hist, L; atol=atol)

    c0 = candidate_col_io(u_hist, y_hist, 0.0, L)
    α = M \ c0
    r0 = c0 - M * α
    residual_zero = norm(r0)

    # If c0 is already outside im(M), any input works in the theorem
    if residual_zero > 1e-8
        u_id = 0
        return u_id, :arbitrary, rank_before, residual_zero
    end

    LK = nullspace(M')
    if size(LK, 2) == 0
        u_id = amp * (2rand(Bool) - 1)
        return u_id, :fallback_random, rank_before, residual_zero
    end

    idx = findfirst(j -> abs(LK[end, j]) > 1e-10, 1:size(LK, 2))
    if isnothing(idx)
        u_id = amp * (2rand(Bool) - 1)
        return u_id, :fallback_random, rank_before, residual_zero
    end

    ξ = LK[:, idx]
    ξ_head = ξ[1:end-1]
    ξ_last = ξ[end]

    # enforce ξ' * c(u) != 0
    c_head = c0[1:end-1]
    u_star = -dot(ξ_head, c_head) / ξ_last

    # shift away from cancellation and saturate
    u_id = clamp(u_star + 0.5 * amp * sign(ξ_last), -amp, amp)

    return u_id, :kernel_guided, rank_before, residual_zero
end


end