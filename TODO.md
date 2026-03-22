The file contains a TODO list for the project.



Large TODOs:
1. Organize the solution of the adjoint equation so it is effectively backpropagating (modify the integral term arising from the source.)
2. Organize the residual function.
3. Implement the adjoint solver.
4. Test the adjoint solver against an analytical example.
5. Do the analytics to check the result is the same.
6. Implement the inverse engine.
7. Test that the inverse engine converges to the right solution for a simple example.
8. Organize the code, push to github.
9. Write documentation and orgranize the theory.
10. Organize all the dimensions, mapping the paper notations to the actual simulation values.


--- Added ---


Missing implementations:
- AcousticModel: implement `dH_dm(speed_field, pressure)` method (called by forward_solver.evaluate_dH_dm but not defined)

Tests needed:
- evaluation.py: assert that ForwardSolver and ChebyshevPropagator give identical results (assert already in place, needs passing)
- adjoint_solver: unit test against a simple scalar ODE with known adjoint solution


