The file contains a TODO list for the project.

Theoretical TODOs:
1. Derive the trapezoidal rule in the convolution solution.


Large TODOs:
1. finish all the check in engine_evaluation.py (testing predicted measurement, residual function and gradient)
2. Test that the inverse engine converges to the right solution for a simple examples.
3. Organize the code, push to github.
4. Write documentation (readme, comments, theory files turn to pdfs) and orgranize the theory.
5. Organize all the dimensions, mapping the paper notations to the actual simulation values.


--- Added ---


Missing implementations:
- AcousticModel: implement `dH_dm(speed_field, pressure)` method (called by forward_solver.evaluate_dH_dm but not defined)




