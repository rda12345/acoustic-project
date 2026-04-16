"""
Example run, demonstrating the inverse engine's performance
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from simulation.acoustic_model import AcousticModel
from utilities.utility_functions import gaussian
from simulation.forward_solver import ForwardSolver
from simulation.inverse_engine import InverseEngine
import pickle


## Evaluate observed data

# Parameters
size = 2**8
L = 1.0  # sets the length scale of the problem
T0 = 18
Nt = 2**9
base_speed = 0.05
learning_rate = 1e-2
max_iters = 200
regularization_constant = 0.0
times = np.linspace(0, T0, Nt)


grid = np.linspace(0, L, size, endpoint=False)

# Set up the measurement positions for the detector
def indices_to_positions(indices, grid):
    return [grid[idx] for idx in indices]

indices_right = list(np.arange((size*3)//4, (size*3)//4 + size//5)) 
indices_left = list(np.arange((size*1)//4 - size//5, (size*1)//4)) 
indices = indices_left + indices_right

measurement_positions = indices_to_positions(indices, grid)  # default to measuring at the 3/4 point of the grid, can be modified by setup_specific method
pulse_parameters = {
    'x_0': L/2,
    'sig': L/20,
    'A_t': 0.01,
    't_0': T0/2,
    'sig_t': T0/20
}

def gaussian_pulse(
        t: float, 
        L: float = L, 
        x_0: float = pulse_parameters['x_0'],
        sig: float = pulse_parameters['sig'], 
        A_t: float = pulse_parameters['A_t'], 
        t_0: float = pulse_parameters['t_0'], 
        sig_t: float = pulse_parameters['sig_t']
    ) -> np.ndarray:
    grid = np.linspace(0, L, size, endpoint=False)
    g_t = A_t * np.exp(-(t-t_0)**2/(2*sig_t**2))
    return g_t * gaussian(grid, x_0, sig)

def get_real_data(
        size: int, 
        L:float, 
        T0: float, 
        base_speed: float, 
        source: callable
    ) -> dict:
    model = AcousticModel(size = size,L = L)    # create model 
    speed_field = base_speed * (np.ones(size) + gaussian(np.linspace(0, L, size, endpoint=False), L/2, L/20) * 0.05)  # create speed field with a Gaussian perturbation
    initial_state = np.zeros(2*size)
    forward_solver = ForwardSolver(model, T0=T0, Nt=Nt, measurement_positions=measurement_positions)  # initialize forward solver with the source term, and measure at the center of the grid at time T0/2, which is when the Gaussian pulse is centered
    forward_solver.run(
        speed_field, 
        initial_state,
        source, 
    )  # run forward solver with the source term, and measure at the center of the grid at time T0/2, which is when the Gaussian pulse is centered
    
    return model.speed_field, forward_solver.get_predicted_data(), forward_solver.get_history()


real_speed_field, observed_data, history = get_real_data(size, L, T0, base_speed, source=gaussian_pulse)

run_params = {
    'size': size,
    'L': L,
    'T0': T0,
    'Nt': Nt,
    'base_speed': base_speed,
    'learning_rate': learning_rate,
    'max_iters': max_iters,
}


## Save simulation results
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
save_path = os.path.join(results_dir, 'simulation_single_pulse.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({
        'real_speed_field': real_speed_field,
        'observed_data': observed_data,
        'history': history,
        'grid': grid,
        'times': times,
        'run_params': run_params,
    }, f)


## Set up the inverse engine
ie = InverseEngine(
    observed_data=observed_data,
    source=gaussian_pulse,
    size=size,
    L=L,
    T0=T0,
    Nt=Nt,
    base_speed=base_speed,
    learning_rate=learning_rate,
    max_iters=max_iters,
    regularization_constant=regularization_constant,
    verbose=True,
    )


## Optimize the speed field
ie.optimize()  # run the optimization

# Save results
results = ie.get_results()
optimized_speed_field = results['optimized_speed_field']
cost_history = results['cost']
gradient_norm_history = results['gradient_norm']
iterations = np.arange(1, max_iters + 1, 1)

save_path = os.path.join(results_dir, 'info_single_pulse.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({
        'real_speed_field': real_speed_field,
        'optimized_speed_field': optimized_speed_field,
        'cost_history': cost_history,
        'gradient_norm_history': gradient_norm_history,
        'iterations': iterations,
        'pulse_parameters': pulse_parameters,
        'run_params': run_params,
    }, f)



