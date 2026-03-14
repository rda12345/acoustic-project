The file contains a TODO list for the project.
1. Write the pseudo code for the step integration of the source term, decompose it to iterative operations propagating in time....
1. Solve a time-dependent source wave equation and compare to the numerics, maybe a sin(omega * t) would be good.
2. Solve for a guassian source, integrating the analytical solution, with the integral.


Large TODOs:
1. Fix the propagation of the forward solver to include the detector recording
2. Test the forward propagation, so it makes sense.
3. Organize the solution of the adjoint equation so it is effectively backpropagating (modify the integral term arising from the source.)
4. Organize the residual function.

1. Implement the adjoint solver.
2. Test the adjoint solver against an analytical example.
3. Do the analytics to check the result is the same.
4. Implement the inverse engine.
5. Implement an inverse_problem file that uses the inverse engine to solve a simple problem.
6. Test that the inverse engine converges to the right solution for a simple example.
7. Organize the code, push to github.
8. Write documentation and orgranize the theory.


Small TODOs:
- In the inverse engine, compute the residual and implement the gradient descent step.