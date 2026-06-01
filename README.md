# Iterative Behavioral Identification for Covert Misappropriation Attacks

Companion code for the paper:

> **"Iterative Behavioral Identification for Covert Misappropriation Attacks
> against Closed-Loop Control Systems"**

Julia implementation of the three-phase, model-free behavioral attack framework
described in the manuscript. The code reproduces Figures 2–4.

---

## Overview

The attacker is a man-in-the-middle (MITM) who intercepts the actuator and sensor
channels of a closed-loop networked control system. Having access only to the
intercepted input–output data, the attacker must:

- learn a usable predictor for the plant (requiring *persistent excitation*), and
- remain undetected by an ∞-norm intrusion-detection system (IDS) throughout.

These two objectives are in fundamental tension. The framework resolves this via
three sequential phases:

| Phase | Purpose | Duration |
|-------|---------|----------|
| **Phase 1** — Initial covert probing | Small-amplitude Rademacher injection (`±δ₁`) seeds a rank-1 Hankel matrix without triggering the IDS; no sensor masking is applied | `T₁ = L` samples |
| **Phase 2** — Iterative behavioral identification | Rank-increasing injections guided by the left-kernel condition (Proposition 1) expand the identified behavioral subspace one dimension at a time; predictor-based sensor masking keeps the IDS residual flat | `T₂ = n + mL − 1` steps |
| **Phase 3** — Covert misappropriation | A DeePC-style convex QP (receding-horizon) drives the plant to a malicious sinusoidal reference while the masked sensor signal remains within the IDS threshold | 100 steps |

The key theoretical tools are Willems' Fundamental Lemma (behavioral systems theory),
the online rank-increasing experiment design of van Waarde et al. (2021), and
DeePC-style data-driven predictive control (Coulson et al., 2019).

---

## Repository structure

```
├── main.jl           # Entry point — phases 1–3, consistency audit, figure generation
├── DeePCUtils.jl     # Discrete-time LTI simulator, Hankel matrix utilities, peak-gain estimation
└── WaardePhase2.jl   # Online rank-increasing experiment design (van Waarde et al., 2021)
```

---

## Requirements

Julia **1.10** or later. Install the required packages from the Julia REPL:

```julia
using Pkg
Pkg.add(["FFTW", "JuMP", "OSQP", "Plots", "LaTeXStrings"])
```

(`LinearAlgebra`, `Statistics`, `Printf`, and `Random` are Julia standard-library
packages and require no installation.)

---

## Reproducing the results

```
julia main.jl
```

or from the Julia REPL:

```julia
include("main.jl")
```

The script runs the full simulation and then prints a consistency audit. All results
are deterministic with `Random.seed!(1234)`.

### What the script does

1. **Warm-up** (200 steps) — closed-loop simulation at the nominal setpoint to
   estimate the steady-state output `ȳ`, input `ū`, and steady-state band `ε_ss`
   (Assumption 2).
2. **Phase 1** (22 steps) — Rademacher injection with amplitude `δ₁ = 10⁻³`;
   no sensor masking; seeds a single Hankel column.
3. **Phase 2** (23 steps) — rank-increasing injections with amplitude `δ₂ = 10⁻²`;
   predictor-based sensor masking active; rank grows from 1 to `n + mL = 24`.
4. **Phase 3** (100 steps) — receding-horizon DeePC QP; sinusoidal attack reference
   `y*(t) = ȳ + 0.5 sin(2πt/50)`; IDS stealth constraint enforced at every step.
5. **Consistency audit** — checks parameter values, closed-loop stability, rank
   staircase, projection-error monotonicity, masking effectiveness, QP feasibility,
   and causal prediction accuracy at TC1.
6. **Figure generation** — saves the following to `figures/`:

| File | Paper label | Content |
|------|------------|---------|
| `fig2_unified.pdf` | `\label{fig:unified}` | Full attack timeline: masked vs. unmasked IDS residual (log scale, top panel) and true plant output tracking the sinusoidal reference (bottom panel) |
| `fig3_rank_growth.pdf` | `\label{fig:rank-growth}` | Phase 2 rank-growth staircase: `rank H⁽ᵏ⁾` from 1 to `n + mL = 24` |
| `fig4_projection_error.pdf` | `\label{fig:projection-error}` | Phase 2 normalized projection error `dₖ(w*)/d₀(w*)` decreasing to zero at full rank (Corollary 1) |

---

## Plant and controller (simulation example)

The simulation uses a second-order discrete-time DC motor (`n = 2`, `m = p = 1`)
in closed loop with a discrete-time PI controller. These matrices serve only to
generate data for the simulation — the attacker has no access to them.

**Plant (DC motor):**
```
A = [1.5462  -0.5646;  1.0  0.0]
B = [1.0;  0.0]
C = [0.3379  0.2793]
```

**Controller (PI):**
```
Ac = [1.0  -0.1673;  0.0  0.0],  Bc = [0.1701;  1.0]
Cc = [1.0  -0.1673],             Dc = 0.1701
```

**Key parameters:**

| Symbol | Value | Description |
|--------|-------|-------------|
| `T_ini` | 2 | DeePC past horizon |
| `N` | 20 | DeePC future horizon |
| `L = T_ini + N` | 22 | Hankel depth |
| `n + mL` | 24 | Full behavioral rank target (TC1) |
| `T* = n + (m+1)L − 1` | 45 | Total samples for complete identification |
| `δ₁` | 10⁻³ | Phase 1 Rademacher amplitude |
| `δ₂` | 10⁻² | Phase 2 rank-increasing amplitude |
| `λ_g` | 10⁻⁵ | DeePC regularization on `g` |
| `λ_u` | 10⁻³ | DeePC regularization on `u_f` |

---

## References

- J. C. Willems et al., "A note on persistency of excitation," *Systems & Control Letters*, 2005.
- H. J. van Waarde et al., "Willems' fundamental lemma for state-space systems and its extension to multiple datasets," *IEEE Control Systems Letters*, 2020.
- J. Coulson, J. Lygeros, F. Dörfler, "Data-enabled predictive control: In the shallows of the DeePC," *ECC*, 2019.
