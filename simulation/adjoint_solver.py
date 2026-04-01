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
    def __init__(self, model: AcousticModel, T0: float, Nt: int) -> None:
        """
        Parameters
        ----------
        model: AcousticModel, the acoustic model, which defines the generator of the adjoint state equation.
        T0: float, total propagation time
        Nt: int, number of time steps
        """
        self.model = model
        self.T0 = T0
        self.Nt = Nt
        self.propagator = ChebyshevPropagator(model=self.model, T0=self.T0, Nt=self.Nt)
        self.state = None   # will store the propagated adjoint state and corresponding adjoint state time-derivative
        self.history = None     # will store the dynamics of the adjoint state and corresponding adjoint state time-derivative

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
        initial_state = np.zeros(2*self.model.size)
        self.model.initialize(speed_field, initial_state)   # initialize acoustic model
        def adjoint_source(tau):
            return np.concatenate([np.zeros(self.model.size), - residual(self.T0 - tau)])   # setting the adjoint source so the integration is equivalent to back propagation from time T0.
        self.state, self.history = self.propagator.propagate_with_source(source=adjoint_source)  # solve the adjoint state equation to get u^dagger
        


    def get_adjoint_state(self) -> np.ndarray:
        """
        Returns the adjoint state
        
        Returns
        -------
        np.ndarray (2*size,): adjoint state at time T0.
        """
        assert self.state is not None, "First solve adjoint equation"
        return self.state[:self.model.size]
    
    def get_history(self) -> np.ndarray:
        """
        Returns the adjoint state dynamics

        Returns
        -------
        np.ndarray (size, Nt): adjoint state at the different time points
        """
        assert self.history is not None, "First solve adjoint equation"
        return self.history[:self.model.size,:]








    