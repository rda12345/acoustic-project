The file contains a TODO list for the project.

Theoretical TODOs:
1. Derive the trapezoidal rule in the convolution solution.
2. Organize the theory equations.


Large TODOs:

2. Write documentation (readme, comments, theory files turn to pdfs) and orgranize the theory.


Documentation TODOs (pre-publication):

Inaccuracies to fix:
- AcousticModel.__init__ docstring says grid is [0, L] — should say [0, L) (endpoint excluded)
- Detector.__init__ references a setup_specific method that doesn't exist — remove or replace
- InverseEngine.get_gradient() has typo "operatoration"
- dH_dc vs dH_dm naming is inconsistent between code and docstrings — pick one and apply consistently

Missing inline documentation:
- propagation_step: add explanation of the three-term Chebyshev recurrence and the fi variable
- Source callable convention (zeros in first half, physical source in second half) should be documented inline in ForwardSolver.run() and ChebyshevPropagator.propagate_with_source(), not only in CLAUDE.md
- observed_data key format (time, position) should be explained at point of use in InverseEngine

README TODOs:
- Add "Running the Code" section with concrete commands
- Add minimal end-to-end example (generate data → run inversion → plot result)

