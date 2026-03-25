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
from utilities.utility_functions import Plot, gaussian, simpson_integrator
from simulation.forward_solver import ForwardSolver
from scipy.special import erf

import matplotlib.pyplot as plt



if __name__ == '__main__':
    
    # parameters
    size = 2**8
    L = 1.0  # sets the length scale of the problem
    dt = 0.1
    T0 = 20
    
    # create model 
    model = AcousticModel(size = size,L = L)
    
    # speed field with a Gaussian initial state
    base_speed = 0.01
    amp_speed = 0.05
    
    # initial state
    sig = model.L/25
    amp = 1.0
    
    # creating the speed field and initial state
    speed_field, initial_state = model.gaussian_initial_state(amp, sig, base_speed, amp_speed)

    # initialization of an initial gaussian pressure distribution and no source
    model.initialize(speed_field, initial_state)
    # initialize detector
    detector = Detector(model)
    
    
    # initialize propagator
    propagator = ChebyshevPropagator(model, detector, T0=T0)
    Nt, dt = propagator.get_Nt(), propagator.get_dt()
    

    # setup detector 
    detector.setup_default({"dt": dt, "Nt": Nt})
    propagator.propagate()
    observed_data = detector.get_data()
    
     
    ## Plotting the dynamics
    initial_pressure = initial_state[:size]
    final_state = model.get_state()
    final_pressure = final_state[:size]
    final_velocity = final_state[size:]
    #Plot(model.grid, initial_pressure,  x_axis_label = 'position',y_axis_label = 'pressure',label = 'initial state')
    #Plot(model.grid, final_pressure.real, x_axis_label = 'position',y_axis_label = 'pressure',label = 'final state')
    
    ## Testing the forward solver against the propagator
    forward_solver = ForwardSolver(model, T0=T0)
    forward_solver.run(speed_field, initial_state)
    predicted_measurements = forward_solver.get_predicted_data()
    final_state_2 = forward_solver.get_state()

    assert np.allclose(final_state, final_state_2), "The final states from the propagator and forward solver do not match!"
    assert observed_data.keys() == predicted_measurements.keys(), "Keys differ"
    assert all(np.isclose(observed_data[k],predicted_measurements[k]) for k in observed_data), "Measurements differ" 



    ## Testing the forward solver against an analytical solution for a time-dependent source
    
    # create model 
    model = AcousticModel(size = size,L = L)
    x_0 = model.L/2
    x = model.grid
    sig = model.L/20

    A_t = 0.01
    t_0 = T0/2
    sig_t = T0/20
    def gaussian_pulse(t: float) -> np.ndarray:
        g_t = A_t * np.exp(-(t-t_0)**2/(2*sig_t**2))
        return g_t * gaussian(model.grid, x_0, sig)




    
    
    # speed field with a Gaussian initial state
    c = 0.02
    amp_speed = 0.05
    
    # evaluating forward solver
    speed_field, initial_state = c*np.ones(size), np.zeros(2*size)
    forward_solver = ForwardSolver(model, T0=T0)
    forward_solver.run(speed_field, initial_state, source=gaussian_pulse)
    predicted_measurements = forward_solver.get_predicted_data()
    final_state = forward_solver.get_state()
    final_pressure = final_state[:model.size]
    
    # evaluating the Green's funciton integral
    x_b = lambda t: (x + c*(T0-t)-x_0)/(np.sqrt(2)*sig)
    x_a = lambda t: (x - c*(T0-t)-x_0)/(np.sqrt(2)*sig)
    f = lambda t: (erf(x_b(t))-erf(x_a(t))) * A_t * np.exp(-(t-t_0)**2/(2*sig_t**2))
    Nt = 100
    Delta_t = T0/Nt
    f_vec = np.array([f(Delta_t*j) for j in range(Nt+1)])
    f_vec = f_vec.transpose()
    analytic_result = (1/(4*c)) * simpson_integrator(f_vec, Delta_t)

    # plt.figure()
    # plt.plot(model.grid, final_pressure.real, label='numerical')
    # plt.plot(model.grid, analytic_result.real, '.',label='analytical')
    # plt.xlabel("position")
    # plt.ylabel("pressure")
    # plt.legend()
    # plt.show()
    assert np.all(np.isclose(final_pressure, analytic_result)), "Difference between the numeric and anlytical result"