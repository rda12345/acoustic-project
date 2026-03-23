The file contains a TODO list for the project.

Theoretical TODOs:
1. Derive the trapezoidal rule in the convolution solution.


Large TODOs:
1. Test the forward solver agains an analytical example, of a
    gaussian in time and space, do the integral with some built in 
    method.
2. Test the adjoint solver against an analytical example.
3. Test the residual, against an analytical result
4. Test the gradiant update angains analytics.
7. Test that the inverse engine converges to the right solution for a simple examples.
8. Organize the code, push to github.
9. Write documentation and orgranize the theory.
10. Organize all the dimensions, mapping the paper notations to the actual simulation values.


--- Added ---


Missing implementations:
- AcousticModel: implement `dH_dm(speed_field, pressure)` method (called by forward_solver.evaluate_dH_dm but not defined)

Tests needed:
- evaluation.py: assert that ForwardSolver and ChebyshevPropagator give identical results (assert already in place, needs passing)
- adjoint_solver: unit test against a simple scalar ODE with known adjoint solution


