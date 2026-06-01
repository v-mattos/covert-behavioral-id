# Iterative Behavioral Identification for Covert Misappropriation Attacks

Julia implementation of a three-phase, model-free behavioral framework for
**covert misappropriation attacks** against closed-loop networked control systems
under a man-in-the-middle (MITM) threat model.

The attacker has access only to intercepted closed-loop input–output data and must
remain undetected by an infinity-norm intrusion-detection system (IDS) throughout.
The framework combines:

- **Behavioral systems theory** (Willems' Fundamental Lemma) for model-free plant
  representation via Hankel matrices
- **Online rank-increasing experiment design** (van Waarde et al., 2021) adapted
  to the closed-loop MITM setting
- **DeePC-style convex QP** (Coulson et al., 2019) for covert attack synthesis
  under explicit IDS constraints

The three phases are:

| Phase | Purpose | Duration |  
|-------|---------|----------|
| **Phase 1** — Initial covert probing | Low-amplitude Rademacher injection seeds a rank-1 Hankel matrix without triggering the IDS | `T₁ = L` samples |
| **Phase 2** — Iterative behavioral ID | Rank-increasing injections with predictor-based sensor masking expand the behavioral subspace to full rank | `T₂ = n + mL − 1` steps |
| **Phase 3** — Covert misappropriation | DeePC optimization drives the plant to a malicious reference while the masked sensor signal stays below the IDS threshold | Receding horizon |

The repository reproduces the simulation outputs underlying the manuscript figures. The final plots in the paper were composed in LaTeX using those outputs as the basis for Figures 2-4.

***

## Repository structure

```
├── main.jl       # Main script — runs all three phases, consistency audit, and figure generation
├── DeePCUtils.jl       # Discrete-time LTI simulator, Hankel/page matrix utilities, peak-gain estimation
└── WaardePhase2.jl     # Online rank-increasing experiment design (van Waarde et al., 2021)
```

***

## Requirements

Julia **1.10+** and the following registered packages:

```julia
using Pkg
Pkg.add([
    "LinearAlgebra", "Statistics", "FFTW",
    "JuMP", "OSQP",
    "Plots"
])
```

> The script also calls `python3` with `Pillow` to embed 300 DPI metadata in the
> PNG figures. Install with `pip install Pillow` if needed (optional — the figures
> are saved correctly without it, only the DPI tag is affected).

***

## Reproducing the results

```julia
include("main.jl")
```

The script will:

1. Simulate a 200-step nominal warm-up to estimate steady-state bounds
2. Run **Phase 1** (22 steps, Rademacher injection, `δ₁ = 10⁻³`)
3. Run **Phase 2** (23 rank-increasing steps, sensor masking active, `δ₂ = 10⁻²`)
4. Run **Phase 3** (100-step receding-horizon QP, sinusoidal reference `A = 0.5`, `P = 50`)
5. Print a full consistency audit (parameter checks, rank staircase, projection-error
   monotonicity, stealth verification)
6. Save the following figures to `figures/`:

| File | Content |
|------|---------|
| `fig2_unified.{pdf,png}` | Full attack timeline — masked vs. unmasked IDS residual (top) and true plant output tracking the sinusoidal reference (bottom) |
| `fig3a_phase2.{pdf,png}` | Phase 2 rank growth staircase: rank `H⁽ᵏ⁾` from 1 to `n + mL = 24` |
| `fig3b_phase2.{pdf,png}` | Phase 2 projection-error decrease: normalized `dₖ(w*)/d₀(w*)` to zero at full rank |

All results are deterministic with `Random.seed!(1234)`.

***

## Plant and controller (DC motor example)

The simulation uses a second-order discrete-time DC motor (`n = 2`, `m = p = 1`)
in closed loop with a PI controller. The matrices are used **only to generate
simulation data** — the attacker has no access to them.

```
Plant:       A = [1.5462  -0.5646;  1.0  0.0],  B = [1.0; 0.0],  C = [0.3379  0.2793]
Controller:  Aₒ = [1.0  -0.1673; 0.0  0.0],     Bₒ = [0.1701; 1.0]
             Cₒ = [1.0  -0.1673],                Dₒ = 0.1701
```

Key parameters: `Tᵢₙᵢ = 2`, `N = 20`, `L = 22`, full behavioral rank target `= 24`,
total identification length `T* = 45` samples.

***
