"""
This file contains code for plotting the results of the optimization process,
including the real and optimized speed fields,
the cost function value over iterations, and the gradient
norm over iterations.
"""
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'info_single_pulse.pkl'), 'rb') as f:
    data = pickle.load(f)
real_speed_field = data['real_speed_field']
optimized_speed_field = data['optimized_speed_field']
cost_history = data['cost_history']
gradient_norm_history = data['gradient_norm_history']
iterations = data['iterations']
run_params = data['run_params']
L = run_params['L']
size = run_params['size']
grid = np.linspace(0, L, size, endpoint=False)
results_dir = os.path.dirname(__file__)

plt.figure(figsize=(10,6))
plt.plot(grid, real_speed_field, label='Real Speed Field')
plt.plot(grid, optimized_speed_field, label='Optimized Speed Field')
plt.xlabel('Position', fontsize=14)
plt.ylabel('Speed', fontsize=14)
plt.title('Real vs Optimized Speed Field', fontsize=16)
plt.legend(fontsize=14)
plt.savefig(os.path.join(results_dir, 'speed_field.png'), bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(iterations, np.log(cost_history), label='Cost Function Value')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('log(Cost Function Value)', fontsize=14)
plt.title('Cost Function vs Iteration', fontsize=16)
plt.legend(fontsize=14)
plt.savefig(os.path.join(results_dir, 'cost_history.png'), bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(iterations, np.log(gradient_norm_history), label='Gradient Norm')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('log(Gradient Norm)', fontsize=14)
plt.title('Gradient Norm vs Iteration', fontsize=16)
plt.legend(fontsize=14)
plt.savefig(os.path.join(results_dir, 'gradient_norm.png'), bbox_inches='tight')
plt.show()