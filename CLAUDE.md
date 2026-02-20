# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Solves the acoustic inverse problem in 1D: given measured pressure data at sensor positions, recover the unknown spatial sound speed field `c(x)`. The forward problem (pressure field evolution) is solved using a Chebyshev spectral expansion of the time-evolution operator, with FFT-based spatial derivatives.

## Running the Code

Always activate the virtual environment first:
```bash
source .venv/bin/activate
```

Run the forward model (propagates a wave and plots initial/final pressure):
```bash
python evaluation.py
```

Run the inversion model (forward pass + cost evaluation; gradient descent is TODO):
```bash
python simple_inversion_model_2.py
```

Run a validation script (stand-alone, visual output — no pytest):
```bash
python tests/num_deriv_check.py
python tests/solve_ivp_check.py
python tests/numerical_integrators_check.py
```

There is no formal test framework configured. Tests are stand-alone scripts that produce plots for visual verification.

## Architecture

### Package Structure

**`simulation/`** — forward model (the physics engine)
- `acoustic_model.py` — `AcousticModel`: grid, speed field, state `[p; v]`, FFT-based spatial derivatives, normalized generator O
- `chebyshev_propator.py` — `ChebyshevPropagator`: Chebyshev expansion of `exp(O·dt)` using modified Bessel coefficients (Kosloff 1994); three-term recurrence for time stepping
- `detector.py` — `Detector`: records pressure at specified (time, position) points during propagation

**`utilities/`** — helper functions
- `utility_functions.py` — `gaussian`, `besseli`, `Plot`, `cv`/`rv` (column/row vector conversion)
- `derivative_with_fft.py` — standalone FFT differentiation (same logic as in `AcousticModel`)

### State Convention

The system state is a concatenated array: `state = [p_1, …, p_N, v_1, …, v_N]` where `p` is pressure and `v` is particle velocity. All classes follow this layout; `state[:size]` = pressure, `state[size:]` = velocity.

### Standard Workflow

```python
model = AcousticModel(size, L)
model.initialize(speed_field, initial_state)   # sets spectral range lam_min/lam_max/dE

detector = Detector(model)
propagator = ChebyshevPropagator(model, detector, T0=T0)
detector.setup_default(propagator)             # 2 positions at L/4 and 3L/4, 5 time samples

observed_data = detector.get_data(propagator)  # runs propagation; returns {(t, x): pressure}
```

### Inversion Model (`simple_inversion_model_2.py`)

`OptModel(AcousticModel)` wraps the forward workflow to support optimization:
- `opt_speed_field` — current guess for `c(x)` (initialized to flat `0.1`)
- `run()` — runs forward model with `opt_speed_field`
- `evaluate()` — computes L2 cost between predicted and observed data
- `GD()`, `update_speed_field()`, `grad()` — gradient descent via adjoint state method (**not yet implemented**)

### Key Parameters

| Parameter | Typical value | Meaning |
|-----------|--------------|---------|
| `size` | `2**8 = 256` | Grid points |
| `L` | `1.0` | Domain length |
| `dt` | `0.1` | Time step |
| `T0` | `1`–`20` | Total simulation time |
| `base_speed` | `0.01`–`0.1` | Background sound speed |

CFL stability limit (conservative): `dt ≤ dx / (c_max · π)`. `ChebyshevPropagator` can enforce this with `cfl_check=True`.

### Known Filename Typo

The Chebyshev propagator file is named `chebyshev_propator.py` (missing 'g'). Import it as:
```python
from simulation.chebyshev_propator import ChebyshevPropagator
```
