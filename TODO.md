The file contains a TODO list for the project.

Theoretical TODOs:
1. Derive the trapezoidal rule in the convolution solution.
2. Organize the theory equations.


Large TODOs:
0. Modify lines 67-68 in acoustic_model.py to make the speed field real, and modify the rest of the code accordingly (e.g., in the inverse engine, make sure to take the real part of the speed field when computing the gradient, and when updating the speed field).
2. Test that the inverse engine converges to the right solution for a simple examples.
3. Test a example_run, demonstrating the code.
4. Organize the code.
5. Turn the check files into tests
5. Get a code review from claude, fix the comments
6. Write documentation (readme, comments, theory files turn to pdfs) and orgranize the theory.
7. Organize a "Theory" folder in the github with all the theory pdfs.
8. Add complexity analysis to the README.md file
9. Organize all the dimensions, mapping the paper notations to the actual simulation values.


--- Added ---


Missing implementations:
- AcousticModel: implement `dH_dm(speed_field, pressure)` method (called by forward_solver.evaluate_dH_dm but not defined)




