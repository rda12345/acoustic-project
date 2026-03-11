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