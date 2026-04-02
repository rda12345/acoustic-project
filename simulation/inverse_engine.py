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
                 source: callable,
                 size: int,
                 L: float,
                 T0: float,
                 Nt: int = 2**8,
                 base_speed: float = 0.01,
                 learning_rate: float = 1e-4,
                 max_iters: int = 100,
                 tol: float = 1e-6
                 ) -> None:
            """
            Initializes the inverse engine with the observed data, source term, and parameters of the problem."""
        
            self.observed_data = observed_data  # the observed data, which is a dict mapping (time, position) points to pressure values
            self.source = source    # the source term, which is a callable that takes time as input and outputs the source term at that time, of shape (size,)
            self.L = L  # sets the length scale of the problem, which is used to define the grid of the acoustic model
            self.size = size    # number of grid points in the acoustic model
            self.Nt = Nt    # number of time points for the forward and adjoint solvers
            self.dt = T0 / (self.Nt-1) # time step size, which is used for the forward and adjoint solvers, and the detector
            self.speed_field = base_speed * np.ones(self.size)  # the speed field to be optimized, initialized to a constant initial guess, can be modified by set_speed_field_guess method
            self.learning_rate = learning_rate  # learning rate for the optimization, can be tuned.
            self.max_iters = max_iters  # maximum number of iterations for the optimization, can be tuned.
            self.tol = tol  # tolerance for convergence, can be tuned
            self.initial_state = np.zeros(2*self.size)    # set to coincide with the assumptions of the adjoint equation method

            self.model = AcousticModel(L=self.L, size=self.size)    # acoustic model
            self.forward_solver = ForwardSolver(model=self.model, T0=T0, Nt=Nt) # forward solver, used to propagate the initial state and compute the predicted measurements at each iteration
            self.adjoint_solver = AdjointSolver(model=self.model, T0=T0, Nt=Nt) # adjoint, used to solve the adjoint equation and compute the adjoint state dynamics
            
    def set_speed_field_guess(self, speed_field) -> None:
        if speed_field.shape != (self.size,):
            raise ValueError(f"speed field must have shape ({self.size},), got ({speed_field.shape})")
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
        for ii in range(self.max_iters):
            print(f'iteration: {ii}')
            self.forward_solver.run(self.speed_field, self.initial_state, self.source)  # propagate the initial state to get the predicted measurements
            u_history = self.forward_solver.get_history()     # pressure field at all time steps
            predicted_data = self.forward_solver.get_predicted_data()   # dict mapping (time, position) points to pressure values
            residual = self.get_residual_function(predicted_data, self.observed_data)  # compute the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
            gradient = self.get_gradient(self.speed_field, u_history, residual)  # compute the gradient of the loss function with respect to the speed field using the adjoint state method
            self.speed_field = self.speed_field - self.learning_rate * gradient  # update the speed field using gradient descent
            
            phi = 0.5 * self.dt * sum(abs(predicted_data[k] - self.observed_data[k])**2 for k in predicted_data)
            print(f'cost: {phi:.6e}')
            print(f'gradient norm: {np.linalg.norm(gradient)}')
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
        
        def residual(t: float) -> np.ndarray:
            """
            Return the residual at time t
            """
            r = np.zeros(self.model.size, dtype=complex)
            keys_at_t = [k for k in observed_data.keys() if np.isclose(k[0], t)]
            for k in keys_at_t:
                position = k[1]    # the keys of observed and predicted data are tuples, of the form (time, position)
                idx = np.where(self.model.grid == position)     # find the array element of x in the grid      
                r[idx] = predicted_data[k] - observed_data[k]
                
            return r 
   
        return  residual
    
    def get_gradient(self, speed_field: np.ndarray, pressure_history: np.ndarray, residual=None) -> np.ndarray:
        """
        Evaluates the gradiant of the cost function with respect to the speed field, which is the product of the adjoint of the derivative of the Hamiltonian with respect to the speed field and the adjoint state.
        
        Parameters
        ----------
        speed_field: np.ndarray (size,), the speed field of the acoustic model
        pressure_history: np.ndarray (size,), the whole pressure dynamics of the forward propagation
        residual: callable, residual function, utilized to evaluate the source of the adjoint equation

        Return
        ------
        np.ndarray (size,), the gradiant of the cost function with respect to the speed field, which is a vector representing the diagonal elements of the matrix grad_m phi(m) = (pd{H}{m})^* u^dagger, where * is the adjoint operatoration.
        """
        if residual is None:
            raise ValueError('get_gradiant requires a residual functions as an input.')
        
        # Backward propagate the adjoint equation to evaluate the adjoint state dynamics
        self.adjoint_solver.solve_adjoint_equation(speed_field, residual)  # solve the adjoint state equation to get u^dagger
        adjoint_history = self.adjoint_solver.get_history()   # the whole dynamics of the adjoint state, of shape (size, Nt)

        integrand = np.zeros((self.size, self.Nt), dtype=complex)
        for j in range(self.Nt):
            pressure_t = pressure_history[:, j]
            dH_dc_t = self.forward_solver.evaluate_dH_dc(pressure_t)  # evaluate (dH/dm @ state) at time t, which is a vector of size (size,)    
            integrand[:, j] =  dH_dc_t * adjoint_history[:, -(j+1)]   # adjoint_history is in reverse time: index 0 = t=T0, flip to align with forward time

        return - 2 * speed_field * simpson_integrator(integrand, Delta_t=self.dt)    # d(phi)/dc — Simpson quadrature
        

    
    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Sets the learning rate for the optimization.
        
        Parameters
        ----------
        learning_rate: float, the learning rate for the optimization.
        """
        self.learning_rate = learning_rate

    
    def set_max_iters(self, max_iters: int) -> None:
        """
        Sets the maximum number of iterations for the optimization.
        
        Parameters
        ----------
        max_iters: int, the maximum number of iterations for the optimization.
        """
        self.max_iters = max_iters
    

    def set_tol(self, tol: float) -> None:
        """
        Sets the tolerance for convergence.
        
        Parameters
        ----------
        tol: float, the tolerance for convergence.
        """
        self.tol = tol