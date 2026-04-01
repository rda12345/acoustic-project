"""
Evaluation of the AcousticModel, ChebyshevPropagator, and ForwardSolver
Runs a single experiment and plots the initial and final states.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from simulation.acoustic_model import AcousticModel
from utilities.utility_functions import gaussian, simpson_integrator
from simulation.forward_solver import ForwardSolver
from simulation.inverse_engine import InverseEngine
from scipy.special import erf


def test_data(data: dict, model: AcousticModel, forward_solver: ForwardSolver, params: tuple) -> None:
    """
    Tests the observed data for an homogeneous speed field against the 
    semi-analytical solution
    """
    x_0, sig, T0, A_t, t_0, sig_t, Nt, base_speed = params
    # evaluating the Green's funciton integral
    def p_analytic(x: float, T_eval: float, base_speed: float) -> float:
        """
        Returns the pressure at position x and time T_eval using the Green's function.
        """
        nt = 100
        Delta_t = T_eval/nt
        def x_b(s):
            return (x + base_speed*(T_eval - s) - x_0)/(np.sqrt(2)*sig)
        def x_a(s):
            return (x - base_speed*(T_eval - s) - x_0)/(np.sqrt(2)*sig)
        def f(s):
            return (erf(x_b(s)) - erf(x_a(s))) * A_t * np.exp(-(s - t_0)**2/(2*sig_t**2))
        f_vec = np.array([f(Delta_t*j) for j in range(nt+1)]).reshape(1, -1)
        return float(((1/(4*base_speed)) * simpson_integrator(f_vec, Delta_t))[0].real)

    detector_positions = forward_solver.detector.positions
    detector_position = float(detector_positions[0])

    t_vec = np.linspace(0, T0, Nt)
    analytic_measurements = [p_analytic(detector_position, t, base_speed) for t in t_vec]
    pressure =  [data[(t, detector_position)] for t in t_vec]
    tol = 1e-3

    assert all(np.isclose(analytic_measurements[j], pressure[j], tol) for j in range(Nt))
   
def test_residual(residual: callable, predicted_data: dict, observed_data: dict, forward_solver: ForwardSolver, Nt) -> None:
    detector_positions = forward_solver.detector.positions    
    detector_position = float(detector_positions[0])
    detector_indices = forward_solver.detector.indices
    detector_index = detector_indices[0]

    t_vec = np.linspace(0, T0, Nt)
    predicted_pressure = np.array([predicted_data[(t, detector_position)] for t in t_vec])
    observed_pressure = np.array([observed_data[(t, detector_position)] for t in t_vec])
    residual_vec_check = predicted_pressure - observed_pressure
    residual_vec = np.array([residual(t)[detector_index] for t in t_vec])
    tol = 1e-3
    assert all(np.isclose(residual_vec, residual_vec_check, tol))

def test_trivial_convergence(ie: InverseEngine, observed_data: dict, speed_field: np.ndarray) -> None:
    """
    Tests trivial convergence condition, when the initial speed field is the true speed field, then the residual should be zero and the gradient should be zero.
    """
    ie.set_speed_field_guess(speed_field)  # set the initial guess for the speed field to the true speed field, so we expect the optimization loop to converge in one step.
    ie.forward_solver.run(ie.speed_field, ie.initial_state, ie.source)  # propagate the initial state to get the predicted measurements
    u_history = ie.forward_solver.get_history()     # pressure field at all time steps
    predicted_data = ie.forward_solver.get_predicted_data()   # dict mapping (time, position) points to pressure values
    residual = ie.get_residual_function(predicted_data, ie.observed_data)  # compute the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
    gradient = ie.get_gradient(ie.speed_field, u_history, residual)  # compute the gradient of the loss function with respect to the speed field using the adjoint state method
    assert np.linalg.norm(residual(0)) < 1e-6, "Residual not zero"
    assert np.linalg.norm(gradient) < 1e-6, "Gradient not zero."




def test_finite_difference_gradient(ie, speed_field, eps=1e-4):
    """Check gradient at the index of maximum gradient magnitude using finite differences."""
    # analytical gradient
    ie.forward_solver.run(speed_field, ie.initial_state, ie.source)
    u_history = ie.forward_solver.get_history()
    predicted = ie.forward_solver.get_predicted_data()
    res = ie.get_residual_function(predicted, ie.observed_data)
    grad = ie.get_gradient(speed_field, u_history, res)

    j = np.argmax(np.abs(grad.real))
    print(f"Testing gradient at index j={j} (max gradient magnitude)")

    # perturbed speed field — central difference O(eps^2)
    c_plus = speed_field.copy()
    c_plus[j] += eps
    ie.forward_solver.run(c_plus, ie.initial_state, ie.source)
    predicted_plus = ie.forward_solver.get_predicted_data()
    phi_plus = 0.5 * ie.dt * sum(abs(predicted_plus[k] - ie.observed_data[k])**2 for k in predicted_plus)

    c_minus = speed_field.copy()
    c_minus[j] -= eps
    ie.forward_solver.run(c_minus, ie.initial_state, ie.source)
    predicted_minus = ie.forward_solver.get_predicted_data()
    phi_minus = 0.5 * ie.dt * sum(abs(predicted_minus[k] - ie.observed_data[k])**2 for k in predicted_minus)

    fd_grad = (phi_plus - phi_minus) / (2 * eps)
    print(f"Analytical gradient[{j}]: {grad[j].real:.6e}")
    print(f"Finite difference  [{j}]: {fd_grad:.6e}")
    print(f"Relative error: {abs(grad[j].real - fd_grad) / abs(fd_grad):.3e}")


def test_adjoint(ie, speed_field):
    """
    Check the adjoint identity: <v, L·f> = <L†·v, f>
    where f is the source, L·f is the forward pressure field,
    and L†·v is the adjoint field driven by v as a residual.
    """
    np.random.seed(0)
    # pre-compute source values so the same array is used during propagation and in the inner product
    ie.forward_solver.run(speed_field, ie.initial_state, lambda t: np.zeros(ie.size))  # dummy run to get Nt, dt
    dt = ie.forward_solver.dt
    Nt = ie.forward_solver.Nt
    f_vals = np.random.randn(ie.size, Nt)  # fixed random source, shape (size, Nt)
    def rand_source(t):
        return f_vals[:, min(int(round(t / dt)), Nt - 1)]

    # forward pass with the fixed source
    ie.forward_solver.run(speed_field, ie.initial_state, rand_source)
    Lf = ie.forward_solver.get_history()  # pressure field L·f, shape (size, Nt)

    # random field v of same shape as the pressure field
    v = np.random.randn(*Lf.shape)
    def rand_residual(t):
        return v[:, min(int(round(t / dt)), Nt - 1)]

    # adjoint pass: L†·v
    ie.adjoint_solver.solve_adjoint_equation(speed_field, rand_residual)
    Ltv = ie.adjoint_solver.get_history()  # shape (size, Nt)

    lhs = np.sum(v * Lf.real)                    # <v, L·f>
    rhs = np.sum(Ltv[:, ::-1].real * f_vals)     # <L†·v, f>  — Ltv is in reverse time, flip it
    print(f"<v, L·f>   = {lhs:.6e}")
    print(f"<L†·v, f>  = {rhs:.6e}")
    print(f"Relative error: {abs(lhs - rhs)/abs(lhs):.3e}")


def test_gradient() -> None:
    pass
   


if __name__ == '__main__':
    
    ## Parameters
    size = 2**8
    L = 1.0  # sets the length scale of the problem
    T0 = 9
    Nt = 2**9
    base_speed = 0.05
        
    ## Evaluate observed data

    model = AcousticModel(size = size,L = L)    # create model 
    speed_field = base_speed * np.ones(size)
    initial_state = np.zeros(2*size)
    ### creating the source term
    x_0 = model.L/2
    sig = model.L/20
    A_t = 0.01
    t_0 = T0/2
    sig_t = T0/20    
    def gaussian_pulse(t: float) -> np.ndarray:
        g_t = A_t * np.exp(-(t-t_0)**2/(2*sig_t**2))
        return g_t * gaussian(model.grid, x_0, sig)

    forward_solver = ForwardSolver(model, T0=T0, Nt=Nt)
    forward_solver.run(speed_field=speed_field, initial_state=initial_state, source=gaussian_pulse)
    observed_data = forward_solver.get_predicted_data()

    
    # Verifuy observed data by comparison to an analytical solution
    params = (x_0, sig, T0, A_t, t_0, sig_t, Nt, base_speed)
    test_data(observed_data, model, forward_solver, params) 


    # Setting up the inverse engine
    ie = InverseEngine(
        observed_data=observed_data,
        source=gaussian_pulse,
        size=size,
        L=L,
        T0=T0,
        Nt=Nt)

    # input the real speed field and test if the gradiant vanishes
    test_trivial_convergence(ie, observed_data, speed_field)

    speed_field_guess = 0.5 * speed_field
    test_finite_difference_gradient(ie, speed_field_guess, eps=1e-5)
    test_finite_difference_gradient(ie, speed_field_guess, eps=1e-6)

    test_adjoint(ie, speed_field)
    
    ## Check a single optimization cycle
    ie.forward_solver.run(ie.speed_field, ie.initial_state, source=gaussian_pulse)  # propagate the initial state to get the predicted measurements
    predicted_data = ie.forward_solver.get_predicted_data()   # dict mapping (time, position) points to pressure values
    u_history = ie.forward_solver.get_history()[:size,:]     # pressure field at all time steps

    ie_base_speed = ie.speed_field[0]
    ie_params = (x_0, sig, T0, A_t, t_0, sig_t, Nt, ie_base_speed) 
    test_data(predicted_data, ie.forward_solver.model, ie.forward_solver, ie_params)
    
    
    residual = ie.get_residual_function(predicted_data, ie.observed_data)  # compute the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
    ie_Nt = ie.forward_solver.propagator.get_Nt()
    test_residual(residual, predicted_data, ie.observed_data, ie.forward_solver, ie_Nt)
    
    # TODO: verify gradient
    gradient = ie.get_gradient(ie.speed_field, u_history, residual)  # compute the gradient of the loss function with respect to the speed field using the adjoint state method
    test_gradient()
    # ie.speed_field = ie.speed_field - ie.learning_rate * gradient  # update the speed field using gradient descent
    # # print(f'gradiant norm: {np.linalg.norm(gradient)})





    # opt_speed_field = inverse_engine.optimize()
     

    

    



