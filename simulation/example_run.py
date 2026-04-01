"""
Example run, demonstrating the inverse engine's performance
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from simulation.acoustic_model import AcousticModel
from utilities.utility_functions import gaussian
from simulation.forward_solver import ForwardSolver
from simulation.inverse_engine import InverseEngine


## Evaluate observed data

# Parameters
size = 2**8
L = 1.0  # sets the length scale of the problem
T0 = 10
Nt = 2**8
base_speed = 0.05
learning_rate = 5e-3
max_iters = 100

def gaussian_pulse(t: float, L: float = L) -> np.ndarray:
    grid = np.linspace(0, L, size, endpoint=False)
    x_0, sig, A_t, t_0, sig_t = L/2, L/20, 0.01, T0/2, T0/20
    g_t = A_t * np.exp(-(t-t_0)**2/(2*sig_t**2))
    return g_t * gaussian(grid, x_0, sig)

def get_real_data(size: int, L:float, T0: float, base_speed: float, source: callable) -> dict:
    model = AcousticModel(size = size,L = L)    # create model 
    speed_field = base_speed * np.ones(size)
    initial_state = np.zeros(2*size)
    forward_solver = ForwardSolver(model, T0=T0, Nt=Nt)
    forward_solver.run(speed_field, initial_state, source)
    
    return model.speed_field, forward_solver.get_predicted_data()


real_speed_field, observed_data = get_real_data(size, L, T0, base_speed, source=gaussian_pulse)


## Set up the inverse engine
ie = InverseEngine(
    observed_data=observed_data,
    source=gaussian_pulse,
    size=size,
    L=L,
    T0=T0,
    Nt=Nt,
    learning_rate=learning_rate,
    max_iters=max_iters
    )

## Optimize teh speed field
optimized_speed_field = ie.optimize()

print(real_speed_field)
print(optimized_speed_field.real)
#TODO: check that the optized speed field agrees with the real speed field