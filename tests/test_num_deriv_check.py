"""
Checking the performance of deriv_n_gen
"""
import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from utilities.derivative_with_fft import deriv_n_gen

def gaussian(x,mu,sig):
    '''Returns a gaussian with mean mu and standard deviation sig'''
    return (1/(np.sqrt(2*np.pi*sig**2))) * np.exp(-np.power(x-mu,2)/(2*sig**2))

## Parameters
L = 1
x = np.linspace(0,L,2**8)

def test_deriv_n_gen(L=L, x=x, PLOT=False):
    """
    Tests the performance of the deriv_n_gen function for a product of a sinusoid and a gaussian,
    which is not an eigenvector of the Fourier derivative operator.
    The results are compared to the analytical solution.
    """
    k = 50*np.pi/L
    y1 = np.sin(k*x)
    dy1_analyt = k*np.cos(k*x)


    mu = L/2
    sig = L/10
    y2 = gaussian(x,mu,sig)
    dy2_analyt= (-(x-mu)/(sig**2))*y2
    ddy2_analyt = (-1/(sig**2) + ((-(x-mu)/(sig**2))**2))*y2

    y = y1*y2
    dy_analyt = dy1_analyt * y2 + y1 * dy2_analyt 

    dy = deriv_n_gen(y,x,1)
    dy2 = deriv_n_gen(y2,x,1)
    ddy2 = deriv_n_gen(y2,x,2)

    if PLOT:
        plt.figure(figsize=(10,6))
        plt.plot(x,dy_analyt,label = "analytic")
        plt.plot(x,dy,label = "numeric", linestyle='dashed')
        plt.legend()
        plt.title("First derivative of a product of a sinusoid and a gaussian")
        plt.show()

        plt.figure(figsize=(10,6))
        plt.plot(x,ddy2_analyt,label = "analytic")
        plt.plot(x,ddy2,label = "numeric", linestyle='dashed')
        plt.legend()
        plt.title("Second derivative of a gaussian")
        plt.show()
    assert np.all(np.isclose(dy, dy_analyt, atol=1e-1)), f"Difference between the numeric and anlytical result for the first derivative {np.max(np.abs(dy - dy_analyt))}"
    assert np.all(np.isclose(dy2, dy2_analyt, atol=1e-4)), f"Difference between the numeric and anlytical result for the first derivative of the gaussian {np.max(np.abs(dy2 - dy2_analyt))}"
    assert np.all(np.isclose(ddy2, ddy2_analyt, atol=1e-4)), f"Difference between the numeric and anlytical result for the second derivative of the gaussian {np.max(np.abs(ddy2 - ddy2_analyt))}"


if __name__ == "__main__":
    PLOT = False
    test_deriv_n_gen(PLOT=PLOT)
    print("All tests passed")


