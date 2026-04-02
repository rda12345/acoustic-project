The file contains a TODO list for the project.

Theoretical TODOs:
1. Derive the trapezoidal rule in the convolution solution.
2. Organize the theory equations.


Large TODOs:
1. Increase the number of detectors and test that the inverse engine converges to the right solution for a simple examples.
2. Test a example_run, demonstrating the code.
3. Finish the code review, and make sure all the comments are fixed.
4. Organize the code.
5. Get a code review from claude, two comments left, one is that the detector is created on every run of forward solver:
Every call to `run()` discards the previous detector and propagator. This means you cannot configure the detector externally (e.g., set custom positions) before calling `run()` — it will be overwritten. It also means the Chebyshev coefficients are recomputed from scratch every time even if nothing changed.
and the other
The "default" position is buried inside a method with no documentation explaining *why* 3/4 of the domain. This makes it hard to change or understand. Either make the position a parameter, or add a comment explaining it is an arbitrary default.
6. Write documentation (readme, comments, theory files turn to pdfs) and orgranize the theory.
7. Organize a "Theory" folder in the github with all the theory pdfs.
8. Add complexity analysis to the README.md file, note that performing a real dft rfft/irfft is twice faster compared to the general complex dft fft/ifft, which exploits the hermitian symmetry of the Fourier coefficients (for the present I chose not to utilize this symmetry.)
9. Organize all the dimensions, mapping the paper notations to the actual simulation values.