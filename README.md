# Acoustic Project


## Project Overview

This project solves the **1D acoustic inverse problem** (full waveform inversion). The goal is to recover an unknown wave speed field from pressure measurements at a few detector positions. The forward problem (wave propagation) uses a Chebyshev expansion of the dynamical propagator, which is spectrally accurate and efficient.

## Running the Code

```bash
# Run a forward simulation (evaluate the acoustic model, plot initial and final pressure)
python evaluation.py

# Run the inversion model (sets up real model + optimization model, prints error)
python simple_inversion_model.py

# Run individual check scripts from the tests/ directory
python tests/num_deriv_check.py
python tests/numerical_integrators_check.py
python tests/solve_ivp_check.py
```

## Architecture

### Forward Problem (wave propagation)

The system solves the 1D acoustic wave equation expressed as two coupled first-order PDEs:
- State vector: `[pressure (size,), velocity (size,)]` → concatenated as `(2*size,)`
- Spatial derivatives computed via FFT with a linear-correction trick (`deriv_n_gen`) to handle non-zero boundary conditions
- Time propagation via **Chebyshev expansion** of the operator `exp(O*dt)`, where coefficients are modified Bessel functions `besseli(n, R)` (Kosloff 1994 style). Coefficients are precomputed once per propagation run and truncated when they fall below `1e-17`.

### Class Hierarchy

```
simulation/
  AcousticModel          — grid, speed field, state, FFT derivatives, generator O
  ChebyshevPropagator    — precomputes Chebyshev coefficients, time-steps via three-term recurrence
  Detector               — records pressure at specified (time, position) pairs during propagation

utilities/
  derivative_with_fft    — standalone FFT derivative functions (deriv_n, deriv_n_gen)
  utility_functions      — gaussian, besseli, Plot, cv/rv helpers
  simple_wave_eq_solver  — finite-difference wave solver (older/reference)
  simple_wave_eq_solver_fft — FFT-based wave solver (older/reference)
```

### Inverse Problem (`simple_inversion_model.py`)

`OptModel` subclasses `AcousticModel` and adds:
- `run()` — propagates with current `opt_speed_field`, records detector data
- `evaluate()` — L2 cost function between predicted and observed data
- `GD()` / `grad()` — gradient descent via adjoint state method (**incomplete**, marked TODO)

The inversion workflow:
1. Run real model → get `observed_data` (dict keyed by `(time, position)`)
2. Initialize `OptModel` with a flat speed guess
3. Iteratively: `run()` → `evaluate()` → `grad()` → `update_speed_field()`


The `simulation/__init__` file is misspelled as `__initi__.py` (missing `t`) — the package still works because Python finds modules by directory, but the init file is not loaded.

## Dependencies

`numpy`, `scipy`, `matplotlib`.

