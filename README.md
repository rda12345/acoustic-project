# Acoustic Project


## Project Overview

This project solves the **1D acoustic inverse problem** (full waveform inversion). The goal is to recover an unknown wave speed field from pressure measurements at a few detector positions. It perfoms gradiant descent, to optimize over the speed field, evaluating the gradient with the adjoint method. To estimate the gradient efficiently, the method combines information from the (forward) solution of the current estimation of the speed field, together with the solution of the adjoint equation, which depends on both the observed and predicted data. 
The solution of the forward and adjoint problems (wave propagation) employs a Chebyshev expansion of the dynamical propagator, which is spectrally accurate and efficient.
The basic theory behind the method is described in appendix A of Ettore's thesis which appears in the theory folder.

The theory is described in detail in the `theory/` folder, along with analytical limits utilized to benchmark the code.
The code is organized into two main folders: `simulation/` contains the core classes for the forward and inverse problems, while `utilities/` contains helper functions and legacy code. The `tests/` folder includes scripts for validating the implementation against analytical results and numerical properties. Plots and associated code are stored in the `results/` folder for analysis. 



## Architecture

### Inverse Problem 

An `InverseEngine` instance accepts model parameters, defining the physics of the problem, and observed measurements, obtained from a "real" experiment. The engine aims to optimize the speed field so the predicted measurements, obtained from its model, coincide with the observed measurements. 

The `InverseEngine` class has two main components:
1. Forward propagation - the `ForwardSolver` is given a model, defining the physics of the problem, a total running time, an initial system state (pressure field + rate of change of the pressure field), and the current prediction for the speed field. It propagates the model and produces a dictionary of `predicted measurements.
2. The model (identical one to the forward solver), current speed field, and predicted measurements are then fed into the `AdjointSolver`. The solver employs the **Adjoint method** to evaluate the gradiant of the L2 cost function with respect to the speed field and outputs it.

The engine updates the speed field via **gradient descent** and feeds it back to the forward solver iteratively, until convergence. 

### Forward Problem (wave propagation with a time-dependent source term)

The system solves the 1D acoustic wave equation expressed as two coupled first-order PDEs:
- The `ForwardSolver` has three components, it begins by building an `AcousticModel` instance, propagates it with the `ChebyshevPropagator`, and records the measurements at specified time utilizing a `Detector` instance.
- The system state is given by the state vector including ca concatination of the pressure and its time-derivative, at all the grid points.
- Spatial derivatives computed via FFT.
- Time propagation via **Chebyshev expansion** of the operator `exp(O*dt)`, where coefficients are modified Bessel functions `besseli(n, R)`. Coefficients are precomputed once per propagation run and truncated when they fall below a defined threshold.
- **Source term**: the additional convolution integral `∫₀ᵀ exp(O(T−τ)) s(τ) dτ` is computed using the Chebyshev propagator and **Trapezodial rule** with `Nt+1` quadrature points.

### Adjoint Problem (backpropagation of the wave equation with a time-dependent source)

The adjoint equation for the wave-equation has the same generator as the model, with a source term that depends on the difference between the observed and predicted data. The equation is backpropagated from the final time to initial time by performing a variable transformation `τ = T-t` and then forward propagating.


## Comments
- Due to scattering, and the zero base pressure, the pressure may obtain negative values, this not unphysical.
- The source is initialized in the middle of the grid, in order to maximize the scattering. If the source away from the perturbation from the speed field the direct pulse dominates the measurements which inhibits the ability to recover the speed field. This is in a consequence of the geometry of the problem, and the fact that the direct pulse is not muted or compensated for.


## Running the Code
Install dependencies:
```bash
pip install numpy scipy matplotlib
```

Run the tests:
```bash
cd python_files
pytest tests/
```

Run the example inversion:
```bash
cd python_files
python simulation/example_run.py
```

## Minimal Example
The `example_run.py` script demonstrates an end-to-end example of the code. It defines a speed field with a Gaussian perturbation, simulates the forward problem to generate synthetic measurements, and then runs the inverse engine to recover the speed field from the measurements. The results are saved at the end.


## Complexity Analysis
The forward propagator has a number of components.
- The FFT-based spatial derivative has a complexity of `O(N log N)` per time step, where `N` is the number of spatial grid points.
- The Chebyshev expansion requires computing the coefficients, which involves evaluating modified Bessel functions. The number of coefficients scales with the spectral radius `R`, where `dk = π/dx` is the maximum wavenumber:
  ```
  R  ~  dE · dt  ~  (c_max · dk) · (T0/Nt)  ~  (N/L) · (T0/Nt)
  ```
  So `N_coeff = O(R) = O( (T0 · N) / (L · Nt) )`.
- The convolution integral is computed with the trapezoidal rule over `Nt` steps, each of cost `N_coeff`:
  ```
  O(Nt · N_coeff) = O((T0/L) · N)
  ```
- The total complexity of the forward propagator is `O(Nt · N log N + (T0/L) · N) = O(Nt · N log N)`.
- The adjoint propagator has the same complexity as the forward propagator, `O(Nt · N log N)`.
- The gradient evaluation involves integrating over time, which has a complexity of `O(Nt · N)`.

Overall, the complexity of one iteration of the inverse problem is `O(Nt · N log N)` due to the forward and adjoint propagations, which dominate the cost. The gradient evaluation adds an additional `O(Nt · N)` term, but this is typically smaller than the propagation cost.

