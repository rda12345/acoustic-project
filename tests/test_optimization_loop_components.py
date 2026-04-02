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



## Parameters
size = 2**8
L = 1.0  # sets the length scale of the problem
T0 = 9
Nt = 2**9
base_speed = 0.05
    
## Evaluate observed data
speed_field = base_speed * np.ones(size)
initial_state = np.zeros(2*size)
# creating the source term
x_0 = L/2
sig = L/20
A_t = 0.01
t_0 = T0/2
sig_t = T0/20    
def gaussian_pulse(t: float, 
                    x_0: float = x_0, 
                    sig: float = sig, 
                    A_t: float = A_t, 
                    t_0: float = t_0, 
                    sig_t: float = sig_t
                    ) -> np.ndarray:
    g_t = A_t * np.exp(-(t-t_0)**2/(2*sig_t**2))
    return g_t * gaussian(np.linspace(0, L, size, endpoint=False), x_0, sig)

def get_observed_data():
    model = AcousticModel(size=size, L=L)   # Initialize model
    forward_solver = ForwardSolver(model, T0=T0, Nt=Nt)
    forward_solver.run(speed_field=speed_field, initial_state=initial_state, source=gaussian_pulse)
    return forward_solver.get_predicted_data(), model, forward_solver


# Setting up the inverse engine
def initialize_ie():
    observed_data, _, _ = get_observed_data()
    return InverseEngine(
        observed_data=observed_data,
        source=gaussian_pulse,
        size=size,
        L=L,
        T0=T0,
        Nt=Nt)

def test_data() -> None:
    """
    Tests the observed data for an homogeneous speed field against the 
    semi-analytical solution
    """   
    data, _, forward_solver = get_observed_data()
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
   
def test_residual() -> None:
    """Tests that the residual function, which is the difference between the observed data and the predicted measurements at the"""
    observed_data, _, forward_solver = get_observed_data()
    ie = initialize_ie()
    speed_field_guess = 0.5 * speed_field  # a guess that is different from the true speed field, so we get a non-zero gradient
    ie.set_speed_field_guess(speed_field_guess)  # set the initial guess for the speed field to the true speed field, so we expect the optimization loop to converge in one step.
    ie.forward_solver.run(ie.speed_field, ie.initial_state, source=gaussian_pulse)  # propagate the initial state to get the predicted measurements
    predicted_data = ie.forward_solver.get_predicted_data()   # dict mapping (time, position) points to pressure values
    residual = ie.get_residual_function(predicted_data, ie.observed_data)  # compute the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
    Nt = ie.forward_solver.propagator.get_Nt()
        
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

def test_trivial_convergence() -> None:
    """
    Tests trivial convergence condition, when the initial speed field is the true speed field, then the residual should be zero and the gradient should be zero.
    """
    ie = initialize_ie()
    ie.set_speed_field_guess(speed_field)  # set the initial guess for the speed field to the true speed field, so we expect the optimization loop to converge in one step.
    ie.forward_solver.run(ie.speed_field, ie.initial_state, ie.source)  # propagate the initial state to get the predicted measurements
    u_history = ie.forward_solver.get_history()     # pressure field at all time steps
    predicted_data = ie.forward_solver.get_predicted_data()   # dict mapping (time, position) points to pressure values
    residual = ie.get_residual_function(predicted_data, ie.observed_data)  # compute the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
    gradient = ie.get_gradient(ie.speed_field, u_history, residual)  # compute the gradient of the loss function with respect to the speed field using the adjoint state method
    assert np.linalg.norm(residual(0)) < 1e-6, "Residual not zero"
    assert np.linalg.norm(gradient) < 1e-6, "Gradient not zero."


def test_finite_difference_gradient(eps=1e-5):
    """Check gradient at the index of maximum gradient magnitude using finite differences."""
    # analytical gradient
    ie = initialize_ie()
    speed_field_guess = 0.5 * speed_field  # a guess that is different from the true speed field, so we get a non-zero gradient
    ie.forward_solver.run(speed_field, ie.initial_state, ie.source)
    u_history = ie.forward_solver.get_history()
    predicted = ie.forward_solver.get_predicted_data()
    res = ie.get_residual_function(predicted, ie.observed_data)
    grad = ie.get_gradient(speed_field_guess, u_history, res)

    j = np.argmax(np.abs(grad.real))

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
    assert np.isclose(grad[j].real, fd_grad, rtol=1e-2), f"Gradient check failed at index {j}: \
        relative error {abs(grad[j].real - fd_grad) / abs(fd_grad):.3e} exceeds tolerance"
   

if __name__ == '__main__':
    test_data()
    test_trivial_convergence()
    test_finite_difference_gradient()
    test_residual()  
    print("All tests passed.")