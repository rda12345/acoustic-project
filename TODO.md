The file contains a TODO list for the project.

Large TODOs:
1. Implement the adjoint solver.
2. Test the adjoint solver against an analytical example.
3. Do the analytics to check the result is the same.
4. Implement the inverse engine.
5. Implement an inverse_problem file that uses the inverse engine to solve a simple problem.
6. Test that the inverse engine converges to the right solution for a simple example.
7. Organize the code, push to github.
8. Write documentation and orgranize the theory.
9. Make the code visible to the public.


Small TODOs:
- Claculate the analytical solution and test propagation_w_source against it.
- Calculate an analytical solution with and test the forward and adjoint solvers against it.
- In the Chebyshev propagator, modify the time steps and dt in integrate_source_term to improve accuracy.
- In the inverse engine, compute the residual and implement the gradient descent step.