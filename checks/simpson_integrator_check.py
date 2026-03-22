"""
Checks the Simpson integrator function
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utilities.utility_functions import simpson_integrator
import numpy as np

if __name__ == "__main__":
    N = 2**9
    x = np.arange(0, 2*np.pi, 2*np.pi/N)
    f = np.sin(x)
    f = f.reshape(1, -1)
    
    g = np.sin(0.5 * x)
    g = g.reshape(1, -1)
    dx = 2*np.pi/N
    error1 = np.abs(simpson_integrator(f, dx)[0])
    error2 = np.abs(simpson_integrator(g, dx)[0]) - 4.0
    print(f'Error sin(x): {error1}')
    print(f'Error sin(x/2): {error2}')
    