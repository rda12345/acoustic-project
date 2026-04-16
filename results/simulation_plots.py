"""
A file containing code for plotting the results of the simulation,
including the speed field, the predicted measurements,
and the history of the state during propagation.
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'simulation_single_pulse.pkl'), 'rb') as f:
    data = pickle.load(f)
real_speed_field = data['real_speed_field']
observed_data = data['observed_data']
history = data['history']
grid = data['grid']
times = data['times']
run_params = data['run_params']
Nt = run_params['Nt']
results_dir = os.path.dirname(__file__)



plt.figure(figsize=(10,6))
plt.plot(grid, history[:,0], label='t=0')
plt.plot(grid, history[:,Nt//4], label=f't={times[Nt//4]:.2f}')
plt.plot(grid, history[:,Nt//2], label=f't={times[Nt//2]:.2f}')
plt.plot(grid, history[:,3*Nt//4], label=f't={times[3*Nt//4]:.2f}')
plt.plot(grid, history[:,-1], label=f't={times[-1]:.2f}')
plt.xlabel('Position', fontsize=14)
plt.ylabel('Pressure', fontsize=14)
plt.title('Pressure Field Evolution', fontsize=16)
plt.legend(fontsize=14)
plt.savefig(os.path.join(results_dir, 'pressure_evolution.png'), bbox_inches='tight')
plt.show()