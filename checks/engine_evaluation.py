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
from utilities.utility_functions import Plot
from simulation.forward_solver import ForwardSolver
from simulation.inverse_engine import InverseEngine



if __name__ == '__main__':
    
    ## Parameters
    size = 2**8
    L = 1.0  # sets the length scale of the problem
    dt = 0.1
    T0 = 20
    base_speed = 0.02
    
    # amp_speed = 0.05
    
    ## Evaluate observed data
    model = AcousticModel(size = size,L = L)    # create model 
    speed_field = base_speed * np.ones(size)
    initial_state = np.zeros(2*size)
    #TODO modify source to gaussian pulse
    source = lambda x: np.zeros(size).astype(complex)
    forward_solver = ForwardSolver(model, T0=T0)
    forward_solver.run(speed_field=speed_field, initial_state=initial_state, source=source)
    observed_data = forward_solver.get_predicted_data()

    # TODO: verifuy observed data


    # Setting up the inverse engine
    ie = InverseEngine(
        observed_data=observed_data,
        source=source,
        size=size,
        L=L,
        T0=T0)
    
    #opt_speed_field = inverse_engine.optimize()

    # TODO: verify predicted measurements
    # TODO: verify residual function
    # TODO: verify gradient
    

    ## Check a single optimization cycle
    ie.forward_solver.run(ie.speed_field, ie.initial_state)  # propagate the initial state to get the predicted measurements
    predicted_measurements = ie.forward_solver.get_predicted_data()   # dict mapping (time, position) points to pressure values
    residual = ie.get_residual_function(predicted_measurements, ie.observed_data)  # compute the residual function, which is the difference between the observed data and the predicted measurements at the defined space-time points.
    gradient = ie.get_gradient(ie.speed_field, residual)  # compute the gradient of the loss function with respect to the speed field using the adjoint state method
    ie.speed_field = ie.speed_field - ie.learning_rate * gradient  # update the speed field using gradient descent
    print(f'gradiant norm: {np.linalg.norm(gradient)})
     

    

    



