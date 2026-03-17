"""
The file contains the utility functions for the acoustic model.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy


def gaussian(x, mu, sig):
    """Returns a gaussian with mean mu and standard deviation sig"""
    return (1/(np.sqrt(2*np.pi*sig**2))) * np.exp(-np.power(x-mu,2)/(2*sig**2))


def gaussian_dot(x, mu, sig, c):
    """Returns a gaussian with mean mu and standard deviation sig"""
    return (c*(x-mu)/sig**2)*gaussian(x,mu,sig)

def besseli(v, z):
    """
    Modified Bessel function of the first kind of real order.
    Parameters
    ----------  
    """
    return scipy.special.iv(v,z)

    
def Plot(x,y,x_axis_label = None,y_axis_label = None,label = None):
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.plot(x,y,label = label)
    plt.legend()
    plt.show()
    
def cv(value_list):
    """Takes a list of numbers and returns a column vector:  n x 1"""
    return rv(value_list).T

def rv(value_list):
    """Takes a list of numbers and returns a row vector: 1 x n"""    
    return np.array([value_list])

def simpson_integrator(integrand: np.ndarray, Delta_t: float):
    """
    Integrates a 2D numpy array, employing the Simpson method.

     Parameters
    ----------
        integrand: np.ndarray (dimension, N), where dimension is the size of the vector at each time-step
                                            and N is the number of time points.
    """
    dimension, N = integrand.shape[0], integrand.shape[1]
    if N%2 != 0:
        raise ValueError("integrate_source_term: Nt must be even for Simpson's rule.")
            
    s = np.zeros(dimension, dtype=complex)
    t = N * Delta_t
    for j in range(N): 
        if j == 0 or j == N:
            s += integrand[:, j]
        elif j % 2 == 1:
            s += 4*integrand[:, j]
        else:
            s += 2*integrand[:, j]
    s *= Delta_t/3
    return s
