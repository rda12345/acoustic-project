"""
Checks the Simpson integrator function
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utilities.utility_functions import simpson_integrator
import numpy as np

def test_simpson_integrator():
    N = 2**10
    x = np.arange(0, 2*np.pi, 2*np.pi/N)
    f = np.sin(x)
    f = f.reshape(1, -1)
    g = np.sin(0.5 * x)
    g = g.reshape(1, -1)
    dx = 2*np.pi/N
    assert np.isclose(simpson_integrator(f, dx)[0], 0.0, atol=1e-3), "Error in simpson integrator for sin(x)"
    assert np.isclose(simpson_integrator(g, dx)[0], 4.0, atol=1e-3), "Error in simpson integrator for sin(x/2)"

if __name__ == "__main__":
    test_simpson_integrator()
    