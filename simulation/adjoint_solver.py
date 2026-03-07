"""
Adjoint Solver for acoustic wave equation inversion.
Implements the adjoint state method for computing the gradient of the loss
function with respect to the speed field in an acoustic isotropic wave
equation inversion problem.
"""


class AdjointSolver:
    """
    An adjoint solver for the acoustic wave equation, which uses the adjoint state method to compute the gradient of the loss function with respect to the speed field. 
    The loss function is defined as the L2 norm of the difference between the observed data and the predicted measurements at the defined space-time points. 
    The predicted measurements are obtained by propagating the initial state with the Chebyshev propagator.
    """
    def __init__(self, model: AcousticModel, T0: float) -> None:
        self.model = model
        self.T0 = T0

    def solve_adjoint_state_equation(self) -> np.ndarray:
        """
        Solves the adjoint state equation, with boundary conditions
        u^dagger(t=T) = 0, and  \pd{u^dagger(t=T)}{t} = 0, where T is the total simulation time.
        The solution is the adjoint state u^dagger. 

        
        Parameters
        ----------
        model: AcousticModel, the acoustic model, which defines the generator of the adjoint state equation.
        K: np.ndarray (p, 2*size), the measurement operator, which maps the state to the predicted measurements at the defined space-time points.
        r_func: np.ndarray (p,), the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
        
        Return
        ------
        np.ndarray (2*size,), the adjoint state u^dagger, which is the solution of the adjoint state equation.
        """
        vec = np.zeros(2*self.size)  # initial state for the adjoint state equation
        generator = self.model.generator()  # get the generator of the adjoint state equation, it is the same as the acoustic models. 
        source = self.adjoint_source()  # evaluate the adjoint source term
        u_dagger = propagate_with_source(vec, generator=generator, source=source, Nt=1)  # solve the adjoint state equation to get u^dagger
        return u_dagger





    def adjoint_source() -> np.ndarray:
        """
        Evaluates the adjoint source term, which is the right hand side of the adjoint state equation.
        
        Parameters
        ----------
        K: np.ndarray (p, 2*size), the measurement operator, which maps the state to the predicted measurements at the defined space-time points.
        r: np.ndarray (p,), the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
        
        Return
        ------
        np.ndarray (2*size,), the adjoint source term, which is -K^* r, where K^* is the adjoint of K.
        """
        K = self.get_measurement_operator()  # get the measurement operator K, which maps the state to the predicted measurements at the defined space-time points.
        r = self.get_residual_function(self.predicted_measurements, self.observed_data)  # get the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
        return self.speed_field**2 * (-K.T @ r)  # the adjoint of K is its transpose, since K is a real matrix. If K were complex, we would need to take the conjugate transpose.


    def dH_dm(self) -> np.ndarray:
        """ 
        Evaluates the derivative of the Hamiltonian with respect to the speed field, which is the matrix that maps the second time derivatives of the state to the gradiant of the cost function with respect to the speed field.
        
        Parameters
        ----------
        speed_field: np.ndarray (size,), the speed field of the acoustic model
        p: np.ndarray (size,), the pressure field of the acoustic model, which is the first half of the state vector.
        
        Return
        ------
        np.ndarray (size,), the derivative of the Hamiltonian with respect to the speed field,\
            which is a vector representing the diagonal elements of the matrix (\pd{H}{m}).

        """
        p = state[:self.size]
        return -2 * (1/self.speed_field) * self.deriv_n(p, self.grid, 2)  # the second time derivative of the pressure field, scaled by -2/m^3, where m is the speed field.

    def grad_m(self) -> np.ndarray:
        """
        Evaluates the gradiant of the cost function with respect to the speed field, which is the product of the adjoint of the derivative of the Hamiltonian with respect to the speed field and the adjoint state.
        
        Parameters
        ----------
        speed_field: np.ndarray (size,), the speed field of the acoustic model
        p: np.ndarray (size,), the pressure field of the acoustic model, which is the first half of the state vector.
        u_dagger: np.ndarray (2*size,), the adjoint state, which is the solution of the adjoint state equation.
        
        Return
        ------
        np.ndarray (size,), the gradiant of the cost function with respect to the speed field, which is a vector representing the diagonal elements of the matrix grad_m phi(m) = (pd{H}{m})^* u^dagger, where * is the adjoint operatoration.
        """
        u_dagger = self.solve_adjoint_state_equation()  # solve the adjoint state equation to get u^dagger
        return self.dH_dm().T @ u_dagger  # since dH_dm is a vector representing the diagonal elements of a matrix, its adjoint is simply its transpose. If dH_dm were a full matrix, we would need to take its conjugate transpose.


    def get_residual_function(self, predicted_measurements: dict, observed_data: dict) -> np.ndarray:
        """
        Evaluates the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
        
        Parameters
        ----------
        predicted_measurements: dict, a dictionary mapping tuples (time, position) to the predicted pressure field at that time and position, which is obtained from the measurement operator K applied to the state of the acoustic model.
        observed_data: dict, a dictionary mapping tuples (time, position) to the pressure field at that time and position, which is obtained from the detector.
        
        Return
        ------
        np.ndarray (p,), the residual function, which is a vector representing the difference between the observed data and the predicted measurements at the defined space-time points.
        """
        r_func = np.array([observed_data[tp] - predicted_measurements[tp] for tp in observed_data.keys()])  
        return r_func

    def update_speed_field(self, eta: float):
        """
        Updates the speed field using a gradient descent step, with a learning rate of eta.
        
        Parameters
        ----------
        eta: float, the learning rate for the gradient descent step.
        """

    # Evaluate the gradient grad_m phi(m) = (pd{H}{m})^* u^dagger, where * is the adjoint operatoration.
        