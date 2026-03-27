"""
Evaluation of the AcousticModel, ChebyshevPropagator, and ForwardSolver
Runs a single experiment and plots the initial and final states.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from simulation.acoustic_model import AcousticModel
from simulation.detector import Detector
from simulation.chebyshev_propagator import ChebyshevPropagator
from utilities.utility_functions import gaussian, simpson_integrator
from simulation.forward_solver import ForwardSolver
from simulation.inverse_engine import InverseEngine
from scipy.special import erf
import matplotlib.pyplot as plt


def test_data(data: dict, model: AcousticModel, forward_solver: ForwardSolver, params: tuple) -> None:
    """
    Tests the observed data for an homogeneous speed field against the 
    semi-analytical solution
    """
    x_0, sig, T0, A_t, sig_t, Nt, base_speed = params
    # evaluating the Green's funciton integral
    def p_analytic(x: float, T_eval: float, base_speed: float) -> float:
        """
        Returns the pressure at position x and time T_eval using the Green's function.
        """
        nt = 100
        Delta_t = T_eval/nt
        x_b = lambda s: (x + base_speed*(T_eval - s) - x_0)/(np.sqrt(2)*sig)
        x_a = lambda s: (x - base_speed*(T_eval - s) - x_0)/(np.sqrt(2)*sig)
        f = lambda s: (erf(x_b(s)) - erf(x_a(s))) * A_t * np.exp(-(s - t_0)**2/(2*sig_t**2))
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

def test_gradient() -> None:
    pass
   


if __name__ == '__main__':
    
    ## Parameters
    size = 2**8
    L = 1.0  # sets the length scale of the problem
    T0 = 20
    base_speed = 0.05
    
    # amp_speed = 0.05
    
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

    forward_solver = ForwardSolver(model, T0=T0)
    forward_solver.run(speed_field=speed_field, initial_state=initial_state, source=gaussian_pulse)
    observed_data = forward_solver.get_predicted_data()

    
    # Verifuy observed data by comparison to an analytical solution
    Nt = forward_solver.propagator.get_Nt()
    params = (x_0, sig, T0, A_t, sig_t, Nt, base_speed)
    test_data(observed_data, model, forward_solver, params) 


    # Setting up the inverse engine
    ie = InverseEngine(
        observed_data=observed_data,
        source=gaussian_pulse,
        size=size,
        L=L,
        T0=T0)

    # TODO: verify gradient


    ## Check a single optimization cycle
    ie.forward_solver.run(ie.speed_field, ie.initial_state, source=gaussian_pulse)  # propagate the initial state to get the predicted measurements
    predicted_data = ie.forward_solver.get_predicted_data()   # dict mapping (time, position) points to pressure values
    u_history = ie.forward_solver.get_history()[:size,:]     # pressure field at all time steps

    ie_base_speed = ie.speed_field[0]
    ie_params = (x_0, sig, T0, A_t, sig_t, Nt, ie_base_speed) 
    test_data(predicted_data, ie.forward_solver.model, ie.forward_solver, ie_params)
    
    
    residual = ie.get_residual_function(predicted_data, ie.observed_data)  # compute the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
    ie_Nt = ie.forward_solver.propagator.get_Nt()
    test_residual(residual, predicted_data, ie.observed_data, ie.forward_solver, ie_Nt)
    
    gradient = ie.get_gradient(ie.speed_field, u_history, residual)  # compute the gradient of the loss function with respect to the speed field using the adjoint state method
    test_gradient()
    # ie.speed_field = ie.speed_field - ie.learning_rate * gradient  # update the speed field using gradient descent
    # # print(f'gradiant norm: {np.linalg.norm(gradient)})





    # opt_speed_field = inverse_engine.optimize()
     

    

    



