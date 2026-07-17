#!/usr/bin/env julia

using LinearAlgebra, Statistics, Printf, Random, JuMP, OSQP, DelimitedFiles
using MathOptInterface
const MOI = MathOptInterface
using Plots
using LaTeXStrings
gr()

include("./DeePCUtils.jl");    using .DeePCUtils
include("./WaardePhase2.jl");  using .WaardePhase2

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIGURATION  (paper Section 10.2)
# ─────────────────────────────────────────────────────────────────────────────
Random.seed!(1234)

# ── Simulation mode ───────────────────────────────────────────────────────
SYS_NOISE_STD = 0.0       
LEN_WARM_UP   = 200       
ID_REF        = 1.0        

# ── Simulation parameters ─────────────────────────────────────────────────
T_ini = 2
N     = 20
L     = T_ini + N    # = 22

# ── Plant (DC motor, minimal, n=2, m=1, p=1) ─────────────────────────────
A_p = [1.5462 -0.5646; 1.0 0.0]
B_p = [1.0; 0.0;;]
C_p = [0.3379 0.2793]
D_p = [0.0;;]

# ── PI controller ─────────────────────────────────────────────────────────
A_c = [1.0 -0.1673; 0.0 0.0]
B_c = [0.1701; 1.0;;]
C_c = [1.0 -0.1673]
D_c = [0.1701;;]

# ── Derived constants ─────────────────────────────────────────────────────
n  = size(A_p, 2)   # = 2
m  = size(B_p, 2)   # = 1
p  = size(C_p, 1)   # = 1
q  = m + p          # = 2

T_star           = n + (m + 1)*L - 1     # total samples for full id. T* = n+(m+1)L−1 (Eq. 7)  = 45
FULL_RANK_TARGET = n + m*L               # TC1 target rank n+mL (Eq. 9)                          = 24
T_2              = FULL_RANK_TARGET - 1  # Phase-2 steps T₂ = n+mL−1 (Eq. 8)                    = 23

# ── Phase 1 & 2 injection amplitudes ─────────────────────────────────────
# δ₁ must satisfy δ₁ ≤ (ε_ids − ε_ss) / Ĝ_∞  (Assumption 1 / Eq. 571–577)
PRBS_AMP_P1 = 0.001
# δ_k must satisfy |δ_k| ≤ (ε_ids − ε_ss) / Ĝ_∞  (Eq. 707–713)
PRBS_AMP_P2 = 0.01

# ── Phase 3 parameters ───────────────────────────────────────────────────
LEN_PHASE_3 = 100
λ_G    = 1e-5   # g-regularization weight λ_g in DeePC QP (Eq. 1056)
λ_U    = 1e-3   # input regularization weight λ_u in DeePC QP (Eq. 1056)
U_MIN  = -2.0
U_MAX  =  2.0

# ── Sinusoidal attack reference  y*(t) = ȳ + A_SIN·sin(2π t / P_SIN) ────
A_SIN = 0.5
P_SIN = 50.0

println("="^72)
println("AUDIT: Incremental Behavioral Covert Misappropriation")
println("="^72)
@printf("Parameters: T_ini=%d  N=%d  L=%d  n+mL=%d  T_2=%d  T*=%d\n",
        T_ini, N, L, FULL_RANK_TARGET, T_2, T_star)
println()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  AUXILIARY — closed-loop eigenvalues
# ─────────────────────────────────────────────────────────────────────────────
function closed_loop_A(Ap, Bp, Cp, _, Ac, Bc, Cc, Dc)
    # Controller input is error e = r - y, so D_c·C_p and B_c·C_p enter negated.
    # Homogeneous CL (stability) matrix, state = col(x_p, x_c):
    #   x_p[t+1] = (Ap - Bp*Dc*Cp)*x_p + Bp*Cc*x_c
    #   x_c[t+1] = -Bc*Cp*x_p          + Ac*x_c
    Acl = [Ap - Bp*Dc*Cp   Bp*Cc;
           -Bc*Cp            Ac  ]
    return Acl
end

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
plant      = DiscreteSystem(A_p, B_p, C_p, D_p; noise_std=SYS_NOISE_STD)
legit_ctrl = DiscreteSystem(A_c, B_c, C_c, D_c)

reset!(plant); reset!(legit_ctrl)

Y_full  = Float64[];  U_total = Float64[]
U_id    = Float64[];  Y_fake  = Float64[]

# ── Phase 0: Warm-up ─────────────────────────────────────────────────────
# Run closed-loop at setpoint r = ID_REF until steady state.
# From the final window estimate: ȳ (nominal setpoint), ū, and ε_ss
# (steady-state output band, Assumption 2 / Section 4).
for t in 1:LEN_WARM_UP
    y_t = isnothing(plant.y_hist) ? [0.0;;] : plant.y_hist[end:end, :]
    u_c = step!(legit_ctrl, [ID_REF;;] .- y_t)[1]
    step!(plant, [u_c;;])
    push!(Y_full, plant.y_hist[end,1])
    push!(U_total, u_c);  push!(U_id, 0.0);  push!(Y_fake, plant.y_hist[end,1])
end

SS_WIN = 50
y_bar  = mean(Y_full[LEN_WARM_UP - SS_WIN + 1 : LEN_WARM_UP])    # ȳ: nominal steady-state output
u_bar  = mean(U_total[LEN_WARM_UP - SS_WIN + 1 : LEN_WARM_UP])   # ū: nominal steady-state input
ε_ss   = maximum(abs.(Y_full[LEN_WARM_UP - SS_WIN + 1 : LEN_WARM_UP] .- y_bar))  # Assumption 2

idx_learn_start = LEN_WARM_UP + 1

# ── Phase 1: Rademacher seeding (Section 5) ──────────────────────────────
# Inject T₁ = L steps of small-amplitude Rademacher (±δ₁) noise with no
# sensor masking (y_{a,t} = 0).  This seeds exactly one Hankel column and
# keeps ‖y_t − ȳ‖_∞ below ε_ids because δ₁ Ĝ_∞ ≤ ε_ids − ε_ss (Eq. 571–577).
LEN_PHASE_1 = L    # T₁ = L  →  exactly 1 Hankel column
for t in 1:LEN_PHASE_1
    y_t = plant.y_hist[end:end, :]
    u_c = step!(legit_ctrl, [ID_REF;;] .- y_t)[1]
    u_a = PRBS_AMP_P1 * (2.0*(rand() > 0.5) - 1.0)   # Rademacher ±δ₁ (Eq. 6)
    step!(plant, [u_c + u_a;;])
    new_y = plant.y_hist[end,1]
    push!(Y_full, new_y); push!(U_total, u_c + u_a)
    push!(U_id, u_a);     push!(Y_fake, new_y)         # no masking: y_{a,t} = 0
end

# ── Empirical peak gain Ĝ_∞ (Eq. 390–394, Section 4.3) ───────────────────
# Estimated as the realized peak-to-peak (ℓ∞-induced) ratio over Phase-1 data,
# matching Definition 1 directly (see DeePCUtils.estimate_peak_gain).
# Used to set ε_ids: any δ₁ satisfying δ₁ ≤ (ε_ids − ε_ss) / Ĝ_∞ keeps the
# unmasked output deviation below the IDS threshold (Assumption 1).
G_hat_inf = estimate_peak_gain(
    Y_full[idx_learn_start:end],
    U_id[idx_learn_start:end],
    Float64(ID_REF))

δ_budget = (G_hat_inf > 0) ? G_hat_inf : 1.0   # gain headroom factor
EPS_IDS  = PRBS_AMP_P1 * G_hat_inf * 4         # ε_ids: IDS alarm threshold (Eq. 571–577)

# ── Build learning set and initial Hankel matrix H^(0) ───────────────────
# H^(0) ∈ ℝ^{qL × 1} is rank-1; its single column is the Phase-1 trajectory.
# The stacked signal w_t = col(u_t, y_t) ∈ ℝ^q, so H_L(w) ∈ ℝ^{qL × N_k}
# (paper Eq. 432–441).
u_learn = copy(U_total[idx_learn_start:end])
y_learn = copy(Y_full[idx_learn_start:end])

H_k      = vcat(hankellize(Matrix{Float64}(u_learn'), L),
                hankellize(Matrix{Float64}(y_learn'), L))   # qL × 1, rank 1
H0_saved = copy(H_k)

# ── Target trajectory w* ∈ B_L  (shadow simulation — sinusoidal reference) ─
# Run a closed-loop shadow simulation driven by the sinusoidal setpoint for
# 4 full periods so the system reaches periodic steady state, then take the
# last L consecutive samples.  The resulting w* is a genuine plant trajectory
# and therefore lies in B_L; after TC1 it is exactly representable by H^(K)
# (Fundamental Lemma, Theorem 1; projection error d_K(w*) = 0, Corollary 4).
let
    sp = DiscreteSystem(A_p, B_p, C_p, D_p; noise_std=0.0)
    sc = DiscreteSystem(A_c, B_c, C_c, D_c)
    n_warmup = round(Int, 4 * P_SIN) + L
    for t in 1:n_warmup
        y_s  = isnothing(sp.y_hist) ? [0.0;;] : sp.y_hist[end:end, :]
        ref  = ID_REF + A_SIN * sin(2π * (t - 1) / P_SIN)
        us   = step!(sc, [ref;;] .- y_s)[1]
        step!(sp, [us;;])
    end
    global u_star_traj = sp.u_hist[end-L+1:end, 1]
    global y_star_traj = sp.y_hist[end-L+1:end, 1]
end
w_star = vcat(u_star_traj, y_star_traj)   # qL-vector, w* ∈ B_L

# Verify w* ∈ B_L: it must lie in the full-rank Hankel's column space.
# We verify this post-hoc after Phase 2.

d0 = norm(w_star - H_k * (H_k \ w_star))

# ── Phase 2 history accumulators ─────────────────────────────────────────
rank_before_hist    = Int[]
rank_after_hist     = Int[]
residual_hist       = Float64[]
u_ctrl_hist         = Float64[]
y_a_hist            = Float64[]
unmasked_resid_hist = Float64[]
masked_resid_hist   = Float64[]
proj_error_hist     = Float64[]

# ── Phase 2: iterative rank-increasing identification (Section 6) ─────────
# At each step choose_u_waarde_siso checks whether the left-kernel condition
# (Eq. 670–673) would be satisfied by the next Hankel column.  If so, it
# selects u_{a,t} satisfying (ξ_L^u)ᵀ u_{a,t} ≠ α_t (Proposition 2)
# to guarantee rank(H^(k+1)) = rank(H^(k)) + 1.
k_phase2 = 0
while true
    u_id_now, _, rank_before, residual0 = choose_u_waarde_siso(
        u_learn, y_learn, L; amp=PRBS_AMP_P2, atol=1e-9)

    rank_before >= FULL_RANK_TARGET && break
    global k_phase2 += 1
    k_phase2 > 3*FULL_RANK_TARGET && (@warn "safety limit at k=$k_phase2"; break)

    # Controller sees masked output ỹ_{t-1} (attacker controls sensor channel)
    u_c = step!(legit_ctrl, [Float64(ID_REF);;] .- [Y_fake[end];;])[1]

    # Causal sensor masking: y_{a,t} = −(ŷ_t − ȳ)  (Eq. 735–738)
    # where ŷ_t = (H_f^(k) g_t)[1] is the one-step-ahead behavioral predictor,
    # g_t = H_p^(k)† w_{ini,t}  (minimum-norm coefficients, Eq. 746–753).
    y_a   = 0.0
    y_hat = y_bar   # fallback: predict ȳ before enough data for predictor
    if length(u_learn) >= T_ini + N
        dm    = build_data_matrix(Matrix{Float64}(u_learn'),
                                  Matrix{Float64}(y_learn'), T_ini, N)
        H_p   = vcat(dm.Up, dm.Yp)
        w_ini = vcat(u_learn[end-T_ini+1:end], y_learn[end-T_ini+1:end])
        g_t   = H_p \ w_ini
        y_hat = (dm.Yf * g_t)[1]
        y_a   = -(y_hat - y_bar)
    end

    # Apply rank-increasing injection; observe true plant output
    u_applied  = u_c + u_id_now
    step!(plant, [u_applied;;])
    new_y      = plant.y_hist[end, 1]
    y_fake_new = new_y + y_a

    push!(Y_full, new_y);       push!(U_total, u_applied)
    push!(U_id,   u_id_now);    push!(Y_fake,  y_fake_new)
    push!(u_ctrl_hist, u_c);    push!(y_a_hist, y_a)

    push!(u_learn, u_applied);  push!(y_learn, new_y)

    # Append new Hankel column
    new_col = vcat(u_learn[end-L+1:end], y_learn[end-L+1:end])
    global H_k = hcat(H_k, new_col)

    rank_after, _ = online_rank_io(u_learn, y_learn, L; atol=1e-9)
    # Projection error d_k(w*) = ‖w* − P_{B_L^(k)} w*‖₂  (Eq. 806–809, Section 7)
    d_k           = norm(w_star - H_k * (H_k \ w_star))

    push!(rank_before_hist,    rank_before)
    push!(rank_after_hist,     rank_after)
    push!(residual_hist,       residual0)
    push!(unmasked_resid_hist, abs(new_y      - y_bar))
    push!(masked_resid_hist,   abs(y_fake_new - y_bar))
    push!(proj_error_hist,     d_k)
end

# ── Phase 3: DeePC attack ─────────────────────────────────────────────────
dm_K    = build_data_matrix(Matrix{Float64}(u_learn'), Matrix{Float64}(y_learn'),
                             T_ini, N)
H_p_K   = vcat(dm_K.Up, dm_K.Yp)
H_f_u_K = dm_K.Uf
H_f_y_K = dm_K.Yf
N_K_p3  = size(H_p_K, 2)

# DeePC attack QP  (paper Eq. 1056–1072, Section 9).
# Minimizes tracking error + input regularization + g-norm penalty subject to:
#   • behavioral consistency  H^(K) g = [w_ini; w_f]
#   • input bounds
#   • IDS stealth  ‖ỹ_{t+i} − ȳ‖_∞ ≤ ε_ids  (Eq. 1061)
#     In the noise-free full-rank case ỹ = y_f + y_a = y_f − (y_f − ȳ) = ȳ,
#     so this constraint is always satisfied and does not alter the optimum.
function solve_attack_qp(H_p, H_f_u, H_f_y, w_ini, y_star_v, u_star_v,
                          u_min, u_max, λ_g, λ_u, y_bar, eps_ids)
    NK = size(H_p, 2);  Nh = length(y_star_v)
    model = Model(OSQP.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "eps_abs", 1e-8)
    set_optimizer_attribute(model, "eps_rel", 1e-8)
    set_optimizer_attribute(model, "max_iter", 20_000)
    @variable(model, g[1:NK])
    @variable(model, u_f[1:Nh])
    @variable(model, y_f[1:Nh])
    # Behavioral consistency: H^(K) g = [w_ini; u_f; y_f]
    @constraint(model, H_p   * g .== w_ini)
    @constraint(model, H_f_u * g .== u_f)
    @constraint(model, H_f_y * g .== y_f)
    # Input bounds
    @constraint(model, u_f .>= u_min)
    @constraint(model, u_f .<= u_max)
    # IDS stealth constraint (Eq. 1061): ‖ỹ_{t+i} − ȳ‖_∞ ≤ ε_ids
    # Masking: y_a = −(y_f − ȳ)  ⟹  ỹ = y_f + y_a = ȳ (trivially satisfied)
    y_tilde = y_f .- (y_f .- y_bar)   # = ȳ · 1, algebraic identity
    @constraint(model, y_tilde .- y_bar .<= eps_ids)
    @constraint(model, y_tilde .- y_bar .>= -eps_ids)
    @objective(model, Min,
        sum((y_f[i] - y_star_v[i])^2 for i in 1:Nh) +
        λ_u * sum((u_f[i] - u_star_v[i])^2 for i in 1:Nh) +
        λ_g * sum(g[j]^2 for j in 1:NK))
    optimize!(model)
    ok = termination_status(model) ∈ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
    return ok ? (value.(u_f), value.(y_f)) : (nothing, nothing)
end

u_star_vec = fill(u_bar, N)

u_a_p3       = Float64[];  y_a_p3    = Float64[]
y_true_p3    = Float64[];  y_fake_p3 = Float64[]
u_c_p3       = Float64[];  phase3_ok = Bool[]
y_ref_p3     = Float64[]   # instantaneous reference y*(t) at each step
w_ini_hist   = Vector{Vector{Float64}}()  # past window at each solve, for the
                                           # receding-horizon audit below

# Receding-horizon Phase-3 loop (Section 9): at each step, solve the DeePC
# attack QP, apply the first optimal input u_{a,t} = u_{f,0} − u_{c,t}, and
# mask the sensor: y_{a,t} = −(ŷ_{f,0} − ȳ)  (Eq. 735–738).
for k in 1:LEN_PHASE_3
    t_local = k - 1   # Phase-3 local time, 0-indexed

    # Build receding-horizon sinusoidal reference y*(t+i), i = 0,…,N−1
    y_star_vec = [y_bar + A_SIN * sin(2π * (t_local + i) / P_SIN) for i in 0:N-1]
    y_ref_now  = y_bar + A_SIN * sin(2π * t_local / P_SIN)
    push!(y_ref_p3, y_ref_now)

    u_c   = step!(legit_ctrl, [Float64(ID_REF);;] .- [Y_fake[end];;])[1]
    w_ini = vcat(u_learn[end-T_ini+1:end], y_learn[end-T_ini+1:end])
    push!(w_ini_hist, copy(w_ini))

    uf_opt, yf_opt = solve_attack_qp(
        H_p_K, H_f_u_K, H_f_y_K,
        w_ini, y_star_vec, u_star_vec,
        U_MIN, U_MAX, λ_G, λ_U, y_bar, EPS_IDS)

    ok = !isnothing(uf_opt)
    if ok
        u_a_now = uf_opt[1] - u_c              # actuator injection u_{a,t}
        y_a_now = -(yf_opt[1] - y_bar)         # sensor masking y_{a,t} = −(ŷ_{f,0} − ȳ)
    else
        u_a_now = 0.0;  y_a_now = 0.0
    end

    u_total    = u_c + u_a_now
    step!(plant, [u_total;;])
    new_y      = plant.y_hist[end, 1]
    y_fake_now = new_y + y_a_now

    push!(Y_full, new_y);      push!(U_total, u_total)
    push!(U_id,   u_a_now);    push!(Y_fake,  y_fake_now)
    push!(u_learn, u_total);   push!(y_learn, new_y)

    push!(u_a_p3, u_a_now);    push!(y_a_p3, y_a_now)
    push!(y_true_p3, new_y);   push!(y_fake_p3, y_fake_now)
    push!(u_c_p3, u_c);        push!(phase3_ok, ok)
end

t_phase1_end = LEN_WARM_UP + LEN_PHASE_1
t_phase2_end = t_phase1_end + k_phase2
t_phase3_end = length(Y_full)


# =============================================================================
# PART 1 — CONSISTENCY AUDIT
# =============================================================================
println("\n", "="^72)
println("PART 1: CONSISTENCY AUDIT")
println("="^72)

pass_count = 0
fail_count = 0

function audit(label, cond, detail="")
    global pass_count, fail_count
    status = cond ? "PASS" : "FAIL"
    cond ? (pass_count += 1) : (fail_count += 1)
    marker = cond ? "✓" : "✗"
    @printf("  [%s] %s %s\n", status, marker, label)
    if !isempty(detail)
        println("        → ", detail)
    end
end

# ── 0. Parameter audit ───────────────────────────────────────────────────
println("\n── Parameters (Section 10.2) ────────────────────────────────────────")

audit("n=2, m=p=1, q=2",
      n==2 && m==1 && p==1 && q==2,
      "n=$n, m=$m, p=$p, q=$q")

audit("T_ini = $T_ini",
      T_ini == 2,
      "T_ini=$T_ini")

audit("N = $N",
      N == 20,
      "N=$N")

audit("L=22, qL=44",
      L==22 && q*L==44,
      "L=$L, qL=$(q*L)")

audit("FULL_RANK_TARGET = n+mL = 24",
      FULL_RANK_TARGET == 24,
      "n+mL=$(n+m*L) = FULL_RANK_TARGET=$FULL_RANK_TARGET")

audit("T* = n+(m+1)L-1 = 45",
      T_star == 45,
      "T*=$T_star")

audit("T_2 = n+mL-1 = 23  (Phase-2 duration, Remark 3)",
      T_2 == 23,
      "T_2=$T_2")

A_p_paper = [1.5462 -0.5646; 1.0 0.0]
B_p_paper = [1.0; 0.0]
C_p_paper = [0.3379, 0.2793]
D_p_paper = [0.0]
audit("Plant A matches paper",  norm(A_p - A_p_paper) < 1e-10, "‖ΔA‖=$(norm(A_p - A_p_paper))")
audit("Plant B matches paper",  norm(vec(B_p) - B_p_paper) < 1e-10)
audit("Plant C matches paper",  norm(vec(C_p) - C_p_paper) < 1e-10)
audit("D=0",                    norm(D_p) < 1e-10)

audit("λ_u = 1e-3, λ_g = 1e-5  (tuned for visible transient)",
      λ_U ≈ 1e-3 && λ_G ≈ 1e-5,
      "λ_u=$λ_U, λ_g=$λ_G")

# ── Closed-loop stability ────────────────────────────────────────────────
Acl = closed_loop_A(A_p, B_p, C_p, D_p, A_c, B_c, C_c, D_c)
cl_eigs = eigvals(Acl)
cl_stable = all(abs.(cl_eigs) .< 1.0)
audit("Closed-loop stable (all |λ| < 1)",
      cl_stable,
      "Eigenvalues: " * join([@sprintf("|λ|=%.4f", abs(e)) for e in cl_eigs], ", "))

# ── Phase 1 audit ────────────────────────────────────────────────────────
println("\n── Phase 1 (Section 5) ──────────────────────────────────────────────")

audit("Noise-free: SYS_NOISE_STD = 0",
      SYS_NOISE_STD == 0.0,
      "SYS_NOISE_STD=$SYS_NOISE_STD")

audit("T_1 = L = $L  (one Hankel column)",
      LEN_PHASE_1 == L,
      "LEN_PHASE_1=$LEN_PHASE_1")

# Check Rademacher: u_id ∈ {−δ₁, +δ₁} during Phase 1
u_id_p1 = U_id[idx_learn_start:idx_learn_start+LEN_PHASE_1-1]
is_rademacher = all(abs.(abs.(u_id_p1) .- PRBS_AMP_P1) .< 1e-12)
audit("Injection is Rademacher: u_{a,t} ∈ {-δ₁, +δ₁}",
      is_rademacher,
      "δ₁=$PRBS_AMP_P1, all |u_id| ≈ δ₁: $is_rademacher")

# No masking in Phase 1: Y_fake[idx] == Y_full[idx]
y_fake_p1 = Y_fake[idx_learn_start:idx_learn_start+LEN_PHASE_1-1]
y_full_p1 = Y_full[idx_learn_start:idx_learn_start+LEN_PHASE_1-1]
audit("No masking in Phase 1: ỹ_t = y_t",
      norm(y_fake_p1 - y_full_p1) < 1e-12,
      "max|ỹ-y| = $(maximum(abs.(y_fake_p1 - y_full_p1)))")

# Stealth: δ₁ ≤ (ε_ids − ε_ss)/Ĝ_∞
stealth_budget = (EPS_IDS - ε_ss) / max(G_hat_inf, 1e-8)
p1_stealth_amp = PRBS_AMP_P1 <= stealth_budget + 1e-10
max_p1_resid   = maximum(abs.(y_full_p1 .- y_bar))
p1_stealth_out = max_p1_resid <= EPS_IDS + 1e-10
audit("Phase 1 amplitude within stealth budget: δ₁ ≤ (ε_ids−ε_ss)/Ĝ_∞",
      p1_stealth_amp,
      @sprintf("δ₁=%.4f  budget=(ε_ids-ε_ss)/Ĝ=%.4f  ε_ss=%.4e  Ĝ_∞=%.4f",
               PRBS_AMP_P1, stealth_budget, ε_ss, G_hat_inf))
audit("Phase 1 output within IDS: max‖y_t−ȳ‖_∞ ≤ ε_ids",
      p1_stealth_out,
      @sprintf("max‖y_t−ȳ‖=%.4e  ε_ids=%.4f", max_p1_resid, EPS_IDS))

# H^(0) rank
rank_H0  = rank(H0_saved; atol=1e-9)
audit("H^(0) has exactly 1 column",
      size(H0_saved, 2) == 1,
      "H^(0) ∈ R^{$(size(H0_saved,1))×$(size(H0_saved,2))}")
audit("rank(H^(0)) = 1",
      rank_H0 == 1,
      "rank=$rank_H0")

# ── Phase 2 audit ────────────────────────────────────────────────────────
println("\n── Phase 2 (Section 6) ──────────────────────────────────────────────")

K = k_phase2
audit("Phase-2 duration K = T_2 = n+mL-1 = $T_2 steps",
      K == T_2,
      "K=$K  (expected T_2=$T_2)")

audit("Total samples T_1+T_2 = T* = $T_star",
      LEN_PHASE_1 + K == T_star,
      "T_1+T_2=$(LEN_PHASE_1+K)  T*=$T_star")

# Rank staircase: +1 per step
rank_seq = vcat(1, rank_after_hist)
rank_diffs = diff(rank_seq)
audit("Rank increases by exactly +1 at each iteration (Proposition 2)",
      all(rank_diffs .== 1),
      "Δrank values: $(unique(rank_diffs))")

audit("Final rank = n+mL = $FULL_RANK_TARGET  (TC1)",
      isempty(rank_after_hist) ? false : rank_after_hist[end] == FULL_RANK_TARGET,
      "rank(H^(K)) = $(isempty(rank_after_hist) ? 1 : rank_after_hist[end])")

# Projection error monotone decreasing
d_vals = vcat(d0, proj_error_hist)
d_diffs = diff(d_vals)
mono_strict = all(d_diffs .< 0)
audit("d_k(w*) strictly monotone decreasing (Theorem 3)",
      mono_strict,
      mono_strict ? "All Δd_k < 0 ✓" :
      @sprintf("Non-monotone at steps: %s", string(findall(d_diffs .>= 0))))

# d_K = 0 at TC1
d_K = isempty(proj_error_hist) ? d0 : proj_error_hist[end]
audit("d_K(w*) = 0 at TC1 (Corollary 4): d_K < 1e-8",
      d_K < 1e-8,
      @sprintf("d_K = %.4e", d_K))

# Behavioral membership: H^(K) H^(K)† w* ≈ w*
proj_wstar = H_k * (H_k \ w_star)
membership_err = norm(proj_wstar - w_star)
audit("w* ∈ B_L: ‖H^(K)H^(K)†w* − w*‖ < 1e-7  (shadow sim)",
      membership_err < 1e-7,
      @sprintf("membership error = %.4e", membership_err))

# Order recovery: n = rank(H^(K)) − mL
n_recovered = FULL_RANK_TARGET - m*L
audit("Order recovery: n = rank(H^(K)) − mL = $n_recovered  (TC1)",
      n_recovered == n,
      "n_recovered=$n_recovered, n=$n")

# Nested subspaces: check B^(k) ⊆ B^(k+1) numerically for k=1,2,3
nested_ok = let
    ok = true
    Hr = copy(H0_saved)
    for ki in 1:min(3, K)
        new_col = vcat(u_learn[LEN_PHASE_1 + ki - L + 1 : LEN_PHASE_1 + ki],
                       y_learn[LEN_PHASE_1 + ki - L + 1 : LEN_PHASE_1 + ki])
        H_prev = copy(Hr)
        Hr     = hcat(Hr, new_col)
        for c in eachcol(H_prev)
            norm(c - Hr * (Hr \ c)) > 1e-8 && (ok = false)
        end
    end
    ok
end
audit("Nested subspaces B^(k) ⊆ B^(k+1) (checked k=1…min(3,K))",
      nested_ok)

# Masking: y_{a,t} = −(ŷ_t − ȳ)  — verify sign/formula
# (The code explicitly computes y_a = -(y_hat - y_bar); we verify non-trivial values exist)
if !isempty(y_a_hist)
    audit("Masking active (‖y_a‖ > 0) for ≥1 Phase-2 step",
          maximum(abs.(y_a_hist)) > 1e-12,
          @sprintf("max|y_a| = %.4e", maximum(abs.(y_a_hist))))
end

# Masked ≤ unmasked residual for a majority of Phase-2 steps. The analytical
# sufficient condition for this (formerly Proposition 5, "Sufficient Condition
# for Residual Reduction") was cut from the 6pp camera-ready (see CHANGELOG
# entry 16); it is not a numbered result in the current paper, only in
# main_draft.tex / an eventual extended version. This check is now a plain
# empirical sanity check on masking effectiveness, not a verification of a
# cited theorem.
if !isempty(masked_resid_hist)
    prop2_frac = count(masked_resid_hist .< unmasked_resid_hist) / length(masked_resid_hist)
    audit("Masked residual < unmasked for majority of Phase-2 steps (masking effectiveness)",
          prop2_frac > 0.5,
          @sprintf("%.0f%% of steps have ‖r_c‖ < ‖r_u‖", 100*prop2_frac))
end

# ── Phase 3 audit ────────────────────────────────────────────────────────
println("\n── Phase 3 (Section 9) ──────────────────────────────────────────────")

n_ok_qp = count(phase3_ok)
audit("QP solved optimally at every step",
      all(phase3_ok),
      @sprintf("%d / %d optimal", n_ok_qp, LEN_PHASE_3))

# Tracking: plant IS oscillating with significant amplitude (last 50 steps).
# For a dynamic sinusoidal reference the relevant check is not a pointwise
# bound (which conflates phase delay with tracking error) but that the plant
# output has sinusoidal character with amplitude ≥ A_SIN/2.
y_p3_ss   = y_true_p3[end-49:end]
amp_actual = maximum(abs.(y_p3_ss .- mean(y_p3_ss)))
audit("Phase-3 plant oscillates: amplitude ≥ A_SIN/2 (last 50 steps)",
      amp_actual >= A_SIN / 2,
      @sprintf("actual amplitude=%.4f  A_SIN/2=%.4f", amp_actual, A_SIN/2))

# Stealth: all ỹ_t within IDS bounds
max_fake_err = maximum(abs.(y_fake_p3 .- y_bar))
audit("Stealth: max‖ỹ_t−ȳ‖_∞ ≤ ε_ids throughout Phase 3",
      max_fake_err <= EPS_IDS + 1e-10,
      @sprintf("max‖ỹ_t−ȳ‖ = %.4e  ε_ids = %.4f", max_fake_err, EPS_IDS))

# Receding horizon: the QP is genuinely re-solved each step, not solved once
# and replayed — verify the past window w_ini (which conditions each solve)
# actually changes step to step, since it always contains the most recent
# T_ini samples.
w_ini_changes = count(k -> w_ini_hist[k] != w_ini_hist[k-1], 2:length(w_ini_hist))
audit("Receding horizon: past window w_ini updates every step (not a fixed open-loop plan)",
      w_ini_changes == length(w_ini_hist) - 1,
      @sprintf("%d / %d consecutive steps show a changed w_ini",
               w_ini_changes, length(w_ini_hist) - 1))

# QP structure: behavioral consistency + IDS satisfied by masking
# With noise-free full-rank H^(K): ỹ = y_f + y_a = y_f − (y_f − ȳ) = ȳ exactly
yt_exact = all(abs.(y_fake_p3 .- y_bar) .< 1e-8)
audit("IDS satisfied trivially: ỹ_t = ȳ (eq. deepc-ids, noise-free full-rank)",
      yt_exact,
      @sprintf("max|ỹ_t − ȳ| = %.2e", maximum(abs.(y_fake_p3 .- y_bar))))

# ── Causal prediction error at TC1 ──────────────────────────────────────
println("\n── Causal Prediction Error at TC1 (Corollary 4) ────────────────────")

# Generate 5 fresh test trajectories from the plant (not used in training)
# and check that H^(K) predicts their futures exactly (noise-free + full rank).
# The causal predictor conditions on BOTH past w_ini AND future input u_f
# (i.e. the two-block DeePC solve [H_p; H_f_u] g = [w_ini; u_f]):
#   g = [H_p_K; H_f_u_K] \ [w_ini; u_f],   y_f_pred = H_f_y_K * g
# This is the exact two-step implied by eq.(deepc-consistency + deepc-split).
# At TC1 (full rank, noise-free) this should give machine-epsilon error.
pred_errs = let
    errs = Float64[]
    sp_test = DiscreteSystem(A_p, B_p, C_p, D_p; noise_std=0.0)
    sc_test = DiscreteSystem(A_c, B_c, C_c, D_c)
    for trial in 1:5
        reset!(sp_test); reset!(sc_test)
        for _ in 1:50
            y_s = isnothing(sp_test.y_hist) ? [0.0;;] : sp_test.y_hist[end:end,:]
            u_s = step!(sc_test, [ID_REF + 0.1*sin(trial);;] .- y_s)[1]
            step!(sp_test, [u_s;;])
        end
        u_traj = Float64[]; y_traj = Float64[]
        for _ in 1:L
            y_s = sp_test.y_hist[end:end,:]
            u_s = step!(sc_test, [ID_REF;;] .- y_s)[1]
            step!(sp_test, [u_s + 0.01*sin(trial*pi/3);;])
            push!(u_traj, sp_test.u_hist[end,1])
            push!(y_traj, sp_test.y_hist[end,1])
        end
        w_ini_test = vcat(u_traj[1:T_ini], y_traj[1:T_ini])
        u_f_true   = u_traj[T_ini+1:end]
        y_f_true   = y_traj[T_ini+1:end]
        # Two-block solve: condition on past AND future input
        Z      = vcat(H_p_K, H_f_u_K)
        z      = vcat(w_ini_test, u_f_true)
        g_pred = Z \ z
        y_f_pred = H_f_y_K * g_pred
        push!(errs, norm(y_f_pred - y_f_true))
    end
    errs
end
max_pred_err = maximum(pred_errs)
audit("Causal prediction error at TC1 < 1e-6 (two-block solve, Corollary 4)",
      max_pred_err < 1e-6,
      @sprintf("max prediction error (5 fresh trajectories) = %.4e", max_pred_err))

# ── H_p dimensions ───────────────────────────────────────────────────────
println("\n── Structural Checks ────────────────────────────────────────────────")
hp_rows_expected = q * T_ini
audit("H_p ∈ R^{q·T_ini × N_K} = R^{$(hp_rows_expected) × N_K}",
      size(H_p_K, 1) == hp_rows_expected,
      @sprintf("H_p ∈ R^{%d×%d}", size(H_p_K)...))

hf_rows_expected = p * N    # output rows only (H_f_y)
audit("H_f_y ∈ R^{p·N × N_K} = R^{$(hf_rows_expected) × N_K}",
      size(H_f_y_K, 1) == hf_rows_expected,
      @sprintf("H_f_y ∈ R^{%d×%d}, H_f_u ∈ R^{%d×%d}",
               size(H_f_y_K)..., size(H_f_u_K)...))

println()
println("─"^72)
@printf("  TOTAL:  %d PASS  |  %d FAIL\n", pass_count, fail_count)
println("─"^72)


# =============================================================================
# PART 2 — PUBLICATION FIGURES  (Plots.jl + plotlyjs backend)
# =============================================================================
println("\n", "="^72)
println("PART 2: GENERATING FIGURES")
println("="^72)

mkpath("figures")

# ── Color palette (Wong 2011, colorblind-safe) ────────────────────────────
const C_BLUE   = "#0072B2"   # primary: y_t, rank, d_k
const C_ORANGE = "#E69F00"   # primary: ỹ_t (masked signal)
const C_GRAY   = "#777777"   # reference lines: y*, IDS bounds
const C_BLACK  = "#000000"   # ȳ, zero lines
const C_RED    = "#D55E00"   # TC1 target, unmasked residual

# ── Publication style: 10 pt fonts, thin grid, white background ──────────
COMMON = (
    linewidth               = 1.5,
    framestyle              = :box,
    grid                    = true,
    gridalpha               = 0.2,
    minorgrid               = false,
    legendfontsize          = 8,
    tickfontsize            = 8,
    guidefontsize           = 9,
    titlefontsize           = 9,
    background_color        = :white,
    background_color_inside = :white,
    margin                  = 6Plots.mm,
)


# Each figure gets its own subfolder: figures/<stem>/<stem>.pdf plus the
# CSV(s) backing it, so a reader can regenerate the plot from raw numbers
# without rerunning the simulation.
function save_fig(p, stem)
    dir = "figures/$(stem)"
    mkpath(dir)
    savefig(p, "$(dir)/$(stem).pdf")
    println("     saved $(dir)/$(stem).pdf")
end

function save_csv(stem::AbstractString, filename::AbstractString,
                   header::Vector{String}, cols...)
    dir = "figures/$(stem)"
    mkpath(dir)
    path = joinpath(dir, filename)
    n = length(cols[1])
    data = Array{Any}(undef, n, length(cols))
    for (j, col) in enumerate(cols), i in 1:n
        data[i, j] = col[i]
    end
    open(path, "w") do io
        println(io, join(header, ","))
        writedlm(io, data, ',')
    end
    println("     saved $(path)")
end

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2  (fig2_unified) — unified time-domain plot: all three phases
# Two vertically stacked panels sharing the x-axis (absolute time index).
# Top panel:    ỹ_t  — what the IDS/operator sees  (boring/flat)
# Bottom panel: y_t  — true plant output            (dramatic sinusoid)
# ─────────────────────────────────────────────────────────────────────────────
println("  → fig2_unified …")

# ── x-axis: Phase 1 starts at t=1, Phase 3 ends at t=T_total
T_p1 = LEN_PHASE_1                              # = 22
T_p2 = k_phase2                                 # = 23
T_p3 = LEN_PHASE_3                              # = 100
T_total = T_p1 + T_p2 + T_p3                   # = 145

t_axis = collect(1:T_total)
y_full_plot = Y_full[idx_learn_start : idx_learn_start + T_total - 1]
y_fake_plot = Y_fake[idx_learn_start : idx_learn_start + T_total - 1]

# Sinusoidal reference — only for Phase 3 region
t_p3_start = T_p1 + T_p2 + 1
y_ref_full  = fill(NaN, T_total)
for t in t_p3_start:T_total
    t_local = t - t_p3_start   # 0-indexed local Phase-3 time
    y_ref_full[t] = y_bar + A_SIN * sin(2π * t_local / P_SIN)
end

# ── IDS residuals over full timeline ─────────────────────────────────────
resid_masked   = abs.(y_fake_plot .- y_bar)   # ‖ỹ_t − ȳ‖: what IDS sees
resid_unmasked = abs.(y_full_plot .- y_bar)   # ‖y_t − ȳ‖: counterfactual

# ── y-axis extents ────────────────────────────────────────────────────────
y_hi_u = max(maximum(resid_unmasked), EPS_IDS) * 1.18
y_lo_u = 0.0
y_lo_r = min(y_bar - EPS_IDS * 1.4, minimum(y_full_plot) - 0.05)
y_hi_r = max(y_bar + A_SIN + 0.08,  maximum(y_full_plot) + 0.05)

# ── Phase boundary x-positions ───────────────────────────────────────────
x_p12 = T_p1 + 0.5        # Phase 1 | Phase 2 boundary
x_p23 = T_p1 + T_p2 + 0.5 # Phase 2 | Phase 3 boundary

# ── TOP PANEL: IDS residual — masked vs unmasked ─────────────────────────
# ylim extended above data to create a header zone for the phase labels.

FLOOR = 1e-5   # anything ≤ this reads as "≈ 0"; sits just below Phase-1's ~1e-3
rm = max.(resid_masked,   FLOOR)
ru = max.(resid_unmasked, FLOOR)

p_top = plot(t_axis, rm;
    label="‖ỹₜ − ȳ‖  (masked)", color=C_BLUE, lw=2.5,
    xlabel="",
    ylabel="Residual (log)",
    title="IDS view: masked vs unmasked residual",
    yscale=:log10,
    ylims=(FLOOR*0.7, y_hi_u * 6.0),   # headroom is MULTIPLICATIVE on a log axis
    COMMON...)
plot!(p_top, t_axis, ru;
    label="‖yₜ − ȳ‖  (unmasked)", color=C_ORANGE, lw=2.5, ls=:dash)
hline!(p_top, [EPS_IDS];
    color="#BBBBBB", lw=1.0, ls=:dash, label="ε_ids")
vline!(p_top, [x_p12, x_p23]; color=C_GRAY, lw=1.7, ls=:dot, label="")
# Phase labels in the header zone above the data traces (y > y_hi_u, inside extended ylim)
y_lbl = y_hi_u * 3.0
annotate!(p_top, T_p1/2,                y_lbl, text("Phase 1", 8, :center, C_GRAY))
annotate!(p_top, T_p1 + T_p2/2,         y_lbl, text("Phase 2", 8, :center, C_GRAY))
annotate!(p_top, T_p1 + T_p2 + T_p3/2,  y_lbl, text("Phase 3", 8, :center, C_GRAY))

# ── BOTTOM PANEL: true plant output ──────────────────────────────────────
p_bot = plot(t_axis, y_full_plot;
    label="yₜ", color=C_BLUE, lw=2.5,
    xlabel="Time step",
    ylabel="yₜ",
    title="True plant output",
    ylims=(y_lo_r, y_hi_r),
    COMMON...)
# Sinusoidal reference (Phase 3 only)
plot!(p_bot, t_axis, y_ref_full;
    label="y*(t)", color=C_RED, lw=1.2, ls=:dashdot)
hline!(p_bot, [y_bar];   color=C_BLACK, lw=0.8, ls=:dot,  label="ȳ")
vline!(p_bot, [x_p12, x_p23]; color=C_GRAY, lw=1.7, ls=:dot, label="")

fig2_unified = plot(p_top, p_bot;
    layout=grid(2, 1; heights=[0.5, 0.5]),
    size=(916, 500),   # PNG = 2148×1200 px at 300 DPI → 7.16"×4.0" (IEEE double-col)
    legend=:topright,
    link=:x)
save_fig(fig2_unified, "fig2_unified")
save_csv("fig2_unified", "residuals.csv",
    ["t", "residual_masked", "residual_unmasked"],
    t_axis, resid_masked, resid_unmasked)
save_csv("fig2_unified", "output_tracking.csv",
    ["t", "y", "y_ref"],
    t_axis, y_full_plot, y_ref_full)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3  (fig:phase2) — 3 panels side by side
# ─────────────────────────────────────────────────────────────────────────────
println("  → fig3_phase2 …")

k_axis     = collect(0:K)
rank_stair = vcat(1, rank_after_hist)
d_all      = vcat(d0, proj_error_hist)
d_norm_all = d_all ./ max(d0, 1e-12)

# (a) Rank growth staircase 1 → n+mL
p3a = plot(k_axis, rank_stair;
    label="rank H⁽ᵏ⁾", color=C_BLUE,
    seriestype=:steppost,
    xlabel="Phase 2 iteration k",
    ylabel="rank H⁽ᵏ⁾",
    title="(a) Rank growth",
    ylims=(0, FULL_RANK_TARGET + 3),
    legend=false,
    gridalpha=0.08,
    COMMON...)
hline!(p3a, [FULL_RANK_TARGET];
    color=C_RED, lw=1.5, ls=:dash,
    label="n+mL = $FULL_RANK_TARGET  (TC1)")

# (b) Normalised projection error on linear scale.
# w* is the sinusoidal shadow trajectory (consistent with Phase 3 execution).
# Strictly decreasing a.s. at each rank increase (Theorem 3).
# Reaches 0 exactly at TC1 (Corollary 4).
p3b = plot(k_axis, d_norm_all;
    label="dₖ(w*) / d₀(w*)", color=C_BLUE,
    marker=:circle, markersize=4, lw=1.5,
    xlabel="Phase 2 iteration k",
    ylabel="dₖ(w*) / d₀(w*)",
    title="(b) Monotone projection-error decrease  (Cor. 1)",
    ylims=(-0.03, 1.05),
    legend=false,
    gridalpha=0.08,
    COMMON...)
hline!(p3b, [0.0]; color=C_BLACK, lw=0.8, ls=:dot, label="")

fig3a = plot(p3a, size=(700, 550))
save_fig(fig3a, "fig3_rank_growth")
save_csv("fig3_rank_growth", "rank_growth.csv",
    ["k", "rank"], k_axis, rank_stair)

fig3b = plot(p3b, size=(700, 550))
save_fig(fig3b, "fig4_projection_error")
save_csv("fig4_projection_error", "projection_error.csv",
    ["k", "d_raw", "d_normalized"], k_axis, d_all, d_norm_all)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
println("\n", "="^72)
println("SUMMARY")
println("="^72)
println()
@printf("  Simulation parameters used:  T_ini=%d  N=%d  L=%d\n", T_ini, N, L)
@printf("  Full-rank target:            n+mL = %d\n", FULL_RANK_TARGET)
@printf("  Phase-2 steps K:             %d  (expected T_2=%d)\n", K, T_2)
@printf("  d_K(w*):                     %.2e  (paper: = 0)\n", d_K)
@printf("  Phase-3 QP failures:         %d / %d\n",
        count(.!phase3_ok), LEN_PHASE_3)
@printf("  Phase 3 sinusoidal ref:      A=%.2f, P=%.0f steps\n", A_SIN, P_SIN)
@printf("  Max |y_t − y*(t)|:           %.4e  (last 50 steps)\n",
        maximum(abs.(y_true_p3[end-49:end] .- y_ref_p3[end-49:end])))
@printf("  Max |ỹ_t − ȳ|:              %.4e  (ε_ids = %.4f)\n",
        maximum(abs.(y_fake_p3 .- y_bar)), EPS_IDS)
println()
println("  Figures → ./figures/<stem>/<stem>.pdf (+ CSV data alongside each)")
println("    fig2_unified/          top+middle panels of \\label{fig:unified} (camera-ready)")
println("    fig4_projection_error/ bottom panel of \\label{fig:unified} (camera-ready)")
println("    fig3_rank_growth/      not in the 6pp camera-ready; kept for the extended version")
println()
@printf("  Audit: %d PASS, %d FAIL\n", pass_count, fail_count)
println("="^72)
