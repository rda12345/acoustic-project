"""
Evaluation of the AcousticModel, ChebyshevPropagator, and ForwardSolver
Runs a single experiment and plots the initial and final states.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from simulation.acoustic_model import AcousticModel
from simulation.detector import Detector
from simulation.chebyshev_propagator import ChebyshevPropagator
from utilities.utility_functions import gaussian, simpson_integrator
from simulation.forward_solver import ForwardSolver
from scipy.special import erf
import matplotlib.pyplot as plt

# global parameters
SIZE = 2**8
L = 1.0  # sets the length scale of the problem
T0 = 20
Nt = 2**8
dt = T0/(Nt-1)
BASE_SPEED = 0.01   # speed field with a Gaussian initial state
AMP_SPEED = 0.05
SIG = L/25  # Gaussian initial state standard deviation
AMP = 1.0   # Gaussian initial state amplitude
PLOT = False
measure_times = [dt*idx for idx in range(Nt)]
grid = np.linspace(0, L, SIZE)
positions = [grid[(SIZE*3)//4]]  # default to measuring at the 3/4 point of the grid, can be modified by setup_specific method]

def test_no_source_gaussian_initial_state(
        size:int = SIZE,
        L:float = L,
        T0:float = T0, 
        Nt:int = Nt,
        base_speed:float = BASE_SPEED,
        amp_speed:float = AMP_SPEED,
        sig:float = SIG, 
        AMP:float = AMP
        ) -> None:
    
    """Testing the forward solver against an analytical solution for a Gaussian initial state and no source"""
    
    model = AcousticModel(size = size,L = L)
    speed_field, initial_state = model.gaussian_initial_state(AMP, sig, base_speed, amp_speed)  # creating the speed field and initial state
    model.initialize(speed_field, initial_state)    # initialization of an initial gaussian pressure distribution and no source
    detector = Detector(model.grid, measure_times=None, positions=None)  # initialize detector
    propagator = ChebyshevPropagator(model, detector, T0=T0, Nt=Nt)  # initialize propagator
    propagator.propagate()
    observed_data = detector.get_data()
    final_state = model.get_state()

    if PLOT:    
        initial_pressure = initial_state[:size]
        final_pressure = final_state[:size]
        plt.figure(figsize=(10,6))
        plt.plot(model.grid, initial_pressure, x_label='position', y_label='pressure', label='Initial Pressure')
        plt.plot(model.grid, final_pressure, x_label='position', y_label='pressure', label='Final Pressure')
        plt.legend()
        plt.title('Initial and Final Pressure Distributions')
        plt.show()
        
    ## Testing the forward solver against the propagator
    forward_solver = ForwardSolver(model, T0=T0, Nt=Nt)
    forward_solver.run(speed_field, initial_state)
    predicted_measurements = forward_solver.get_predicted_data()
    final_state_2 = forward_solver.get_state()

    assert np.allclose(final_state, final_state_2), "The final states from the propagator and forward solver do not match!"
    assert observed_data.keys() == predicted_measurements.keys(), "Keys differ"
    assert all(np.isclose(observed_data[k],predicted_measurements[k]) for k in observed_data), "Measurements differ" 


def test_time_dependent_source(
        size=SIZE,
        L=L, T0=T0,
        Nt=Nt, 
        base_speed=BASE_SPEED, 
        amp_speed=AMP_SPEED, 
        sig=SIG, 
        AMP=AMP
        ) -> None:
    """Testing the forward solver against an analytical solution for a time-dependent source"""
    
    model = AcousticModel(size = size, L = L)    # create model 
    
    x_0 = model.L/2
    x = model.grid
    sig = model.L/20
    A_t = 0.01
    t_0 = T0/2
    sig_t = T0/20
    def gaussian_pulse(t: float) -> np.ndarray:
        g_t = A_t * np.exp(-(t-t_0)**2/(2*sig_t**2))
        return g_t * gaussian(model.grid, x_0, sig)


    # evaluating forward solver
    speed_field, initial_state = base_speed*np.ones(size), np.zeros(2*size)
    forward_solver = ForwardSolver(model, T0=T0, Nt=Nt)
    forward_solver.run(speed_field, initial_state, source=gaussian_pulse)
    final_state = forward_solver.get_state()
    final_pressure = final_state[:model.size]

    # evaluating the Green's funciton integral
    def x_b(t):
        return (x + base_speed*(T0-t)-x_0)/(np.sqrt(2)*sig)
    def x_a(t):
        return (x - base_speed*(T0-t)-x_0)/(np.sqrt(2)*sig)
    def f(t):
        return (erf(x_b(t))-erf(x_a(t))) * A_t * np.exp(-(t-t_0)**2/(2*sig_t**2))
    Nt = 100
    Delta_t = T0/(Nt-1)
    f_vec = np.array([f(Delta_t*j) for j in range(Nt)])
    f_vec = f_vec.transpose()
    analytic_result = (1/(4*base_speed)) * simpson_integrator(f_vec, Delta_t)

    assert np.all(np.isclose(final_pressure, analytic_result)), "Difference between the numeric and anlytical result"


if __name__ == "__main__":
    test_no_source_gaussian_initial_state()
    test_time_dependent_source()
    print("All tests passed")