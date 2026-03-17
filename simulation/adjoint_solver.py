"""
Adjoint Solver for acoustic wave equation inversion.
Implements the adjoint state method for computing the gradient of the loss
function with respect to the speed field in an acoustic isotropic wave
equation inversion problem.
"""
import numpy as np
from simulation.acoustic_model import AcousticModel
from simulation.chebyshev_propagator import ChebyshevPropagator


class AdjointSolver:
    """
    An adjoint solver for the acoustic wave equation, which uses the adjoint state method to compute the gradient of the loss function with respect to the speed field. 
    The loss function is defined as the L2 norm of the difference between the observed data and the predicted measurements at the defined space-time points. 
    The predicted measurements are obtained by propagating the initial state with the Chebyshev propagator.
     """
    def __init__(self, model: AcousticModel, T0: float) -> None:
        """
        Parameters
        ----------
        model: AcousticModel, the acoustic model, which defines the generator of the adjoint state equation.
        T0: float, total propagation time
        """
        self.model = model
        self.T0 = T0
        self.propagator = ChebyshevPropagator(model=self.model, T0=self.T0)

    def solve_adjoint_equation(self, speed_field: np.ndarray, residual: callable) -> np.ndarray:
        """
        Solves the adjoint state equation, with boundary conditions
        u^dagger(t=T0) = 0, and  \pd{u^dagger(t=T0)}{t} = 0, where T0 is the total simulation time.
        The solution is the adjoint state u^dagger. 
        In order to propagate backward in time from time t=T0 to t=0, we define tau = T0-t, and integrate over
        tau. The generator of the wave equation remains the same, but the adjoint source function is modified.

        
        Parameters
        ----------
        speed_field: np.ndarray (size,), the speed field of the acoustic model
        residual: callable, residual function, the function outputs an np.ndarray of size (2* model.size,)
        
        Return
        ------
        np.ndarray (size,), the adjoint state u^dagger, which is the solution of the adjoint state equation.
        """
        adjoint_source = lambda tau: speed_field**2 * residual(self.T0 - tau)
        u_dagger = self.propagator.propagate_with_source(source=adjoint_source)  # solve the adjoint state equation to get u^dagger
        return u_dagger








    