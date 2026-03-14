"""
Inverse engine for acoustic wave equation. 
"""
import numpy as np
from simulation.acoustic_model import AcousticModel
from simulation.forward_solver import ForwardSolver
from simulation.adjoint_solver import AdjointSolver


class InverseEngine:
    """
    An inverse engine for the acoustic wave equation, which uses the adjoint state method to compute the gradient of the loss function with respect to the speed field. 
    The loss function is defined as the L2 norm of the difference between the observed data and the predicted measurements at the defined space-time points. 
    The predicted measurements are obtained by propagating the initial state with the Chebyshev propagator.
    """
    def __init__(self,
                 observed_data: dict,
                 size: int,
                 L: float,
                 T0: float,
                 ) -> None:
        
            self.observed_data = observed_data
            self.model = AcousticModel(L=L, size=size)
            self.forward_solver = ForwardSolver(model=self.model, T0=T0)
            self.adjoint_solver = AdjointSolver(model=self.model, T0=T0)
            base_speed = 0.01
            self.speed_field = base_speed * np.ones(self.model.size)  # the speed field to be optimized, initialized to a constant initial guess.
            self.learning_rate = 0.01  # learning rate for the optimization, can be tuned.
            self.max_iters = 100  # maximum number of iterations for the optimization, can be tuned.
            self.tol = 1e-6  # tolerance for convergence, can be tuned
            self.initial_state = np.zeros(2*self.model.size)    # set to coincide with the assumptions of the adjoint equation method

    def set_speed_field_guess(self, speed_field) -> None:
        self.speed_field = speed_field

    def optimize(self) -> np.ndarray:
        """
        Optimizes the speed field by minimizing the loss function, which is the L2 norm of the difference
        between the observed data and the predicted measurements at the defined space-time points. 
        The predicted measurements are obtained by propagating the initial state with the Chebyshev propagator.
        
        Returns
        -------
        np.ndarray (size,), the optimized speed field.
        """
        for _ in range(self.max_iters):
            self.forward_solver.run(self.speed_field, self.initial_state)  # propagate the initial state to get the predicted measurements
            predicted_measurements = self.forward_solver.get_predicted_data() 
            residual = self.get_residual_function(predicted_measurements, self.observed_data)  # compute the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
            gradient = self.adjoint_solver.get_gradient(residual)  # compute the gradient of the loss function with respect to the speed field using the adjoint state method
            self.speed_field = self.speed_field - self.learning_rate * gradient  # update the speed field using gradient descent
            if np.linalg.norm(gradient) < self.tol:  # check for convergence
                break
        return self.speed_field

    def get_residual_function(self, predicted_data: dict, observed_data: dict) -> np.ndarray:
        """
        Computes the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
        
        Parameters
        ----------
        predicted_data: dict, the predicted measurements at the defined space-time points, obtained from the forward solver.
        observed_data: dict, the observed measurments at the defined space-time points.
        
        Returns
        -------
        callable, the residual function, which returns a np.ndarray (p,), where is the number of measurement points.
        """
        
        def residual(t):
            r = np.zeros(self.model.size)
            keys_at_t = [k for k in observed_data.keys() if k[0]==t]
            for k in keys_at_t:
                position = k[1]    # the keys of observed and predicted data are tuples, of the form (time, position)
                idx = np.where(self.model.grid == position)     # find the array element of x in the grid      
                r[idx] = predicted_data
                return r
        
        return  residual
       
    def set_learning_rate(eta: float) -> None:
        """
        Sets the learning rate for the optimization.
        
        Parameters
        ----------
        eta: float, the learning rate for the optimization.
        """
        self.eta = eta

    
    def set_max_iters(max_iters: int) -> None:
        """
        Sets the maximum number of iterations for the optimization.
        
        Parameters
        ----------
        max_iters: int, the maximum number of iterations for the optimization.
        """
        self.max_iters = max_iters
    

    def set_tol(tol: float) -> None:
        """
        Sets the tolerance for convergence.
        
        Parameters
        ----------
        tol: float, the tolerance for convergence.
        """
        self.tol = tol