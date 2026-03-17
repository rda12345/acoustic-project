"""
Inverse engine for acoustic wave equation. 
"""
import numpy as np
from simulation.acoustic_model import AcousticModel
from simulation.forward_solver import ForwardSolver
from simulation.adjoint_solver import AdjointSolver
from utilities.utility_functions import simpson_integrator


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
            predicted_measurements = self.forward_solver.get_predicted_data()   # dict mapping (time, position) points to pressure values
            residual = self.get_residual_function(predicted_measurements, self.observed_data)  # compute the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
            gradient = self.get_gradient(self.speed_field, residual)  # compute the gradient of the loss function with respect to the speed field using the adjoint state method
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
                r[idx] = predicted_data[k] - observed_data[k]
                return r
   
        return  residual
    
    def get_gradient(self, speed_field, residual=None) -> np.ndarray:
        """
        Evaluates the gradiant of the cost function with respect to the speed field, which is the product of the adjoint of the derivative of the Hamiltonian with respect to the speed field and the adjoint state.
        
        Parameters
        ----------
        speed_field: np.ndarray (size,), the speed field of the acoustic model
        residual: callable,   
        Return
        ------
        np.ndarray (size,), the gradiant of the cost function with respect to the speed field, which is a vector representing the diagonal elements of the matrix grad_m phi(m) = (pd{H}{m})^* u^dagger, where * is the adjoint operatoration.
        """
        if not residual:
            raise ValueError('get_gradiant requires a residual functions as an input.')
        
        # forward propagate and get the pressure field dynamics
        self.forward_solver.run(
            speed_field=self.speed_field,
            source=self.source)
        u_history = self.forward_solver.get_history()
        Nt = u_history.shape[1]     # number of time-points

        # backward propagate the adjoint equation to evaluate the adjoint state dynamics
        self.adjoint_solver.solve_adjoint_equation(speed_field, residual)  # solve the adjoint state equation to get u^dagger
        u_dagger_history = self.adjoint_solver.get_history()

        integrand = np.zeros_like(self.m, Nt)
        for j in range(Nt):
            u_t = u_history[:, j]
            dH_dm_t = self.forward_solver.evaluate_dH_dm(speed_field, u_t)  # matrix    
            integrand[:, j] = dH_dm_t.T @  u_dagger_history[:, j]   

        return simpson_integrator(integrand, Delta_t=dt)

        

        #INTEGRATE self.dH_dm().T @ u_dagger
       
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