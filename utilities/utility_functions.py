"""
The file contains the utility functions for the acoustic model.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy


def gaussian(x: np.array, mu: float, sig: float) -> np.ndarray:
    """
    Returns an area-normalized Gaussian (peak = 1/(sqrt(2*pi)*sig)).

    Parameters
    ----------
    x: np.ndarray, evaluation points
    mu: float, mean
    sig: float, standard deviation
    """
    return (1/(np.sqrt(2*np.pi*sig**2))) * np.exp(-np.power(x-mu,2)/(2*sig**2))


def gaussian_dot(x: np.ndarray, mu: float, sig: float, c: float):
    """
    Returns the spatial derivative of a Gaussian, scaled by c.
    Equivalent to c * d/dx [gaussian(x, mu, sig)].

    Parameters
    ----------
    x: np.ndarray, evaluation points
    mu: float, mean of the Gaussian
    sig: float, standard deviation of the Gaussian
    c: float or np.ndarray, scaling factor (e.g. speed field)
    """
    return (c*(x-mu)/sig**2)*gaussian(x,mu,sig)

def besseli(v, z):
    """
    Modified Bessel function of the first kind of real order.

    Parameters
    ----------
    v: int, order of the Bessel function
    z: float or complex, argument
    """
    return scipy.special.iv(v,z)


def Plot(x, y, x_axis_label=None, y_axis_label=None, label=None):
    """
    Plots y vs x with optional axis labels and legend entry.

    Parameters
    ----------
    x: np.ndarray, x-axis values
    y: np.ndarray, y-axis values
    x_axis_label: str, label for the x axis
    y_axis_label: str, label for the y axis
    label: str, legend label for the plotted line
    """
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.plot(x,y,label = label)
    plt.legend()
    plt.show()

def cv(value_list):
    """
    Takes a list of numbers and returns a column vector of shape (n, 1).

    Parameters
    ----------
    value_list: list or np.ndarray, input values
    """
    return rv(value_list).T

def rv(value_list):
    """
    Takes a list of numbers and returns a row vector of shape (1, n).

    Parameters
    ----------
    value_list: list or np.ndarray, input values
    """
    return np.array([value_list])

def simpson_integrator(integrand: np.ndarray, Delta_t: float):
    """
    Integrates a 2D numpy array along the time axis using the Simpson method.
    For an even number of points N, uses the 3/8 Simpson rule for the last three
    intervals and the 1/3 Simpson rule for the rest.
    For an odd number of points, uses the 1/3 Simpson rule throughout.

    Parameters
    ----------
    integrand: np.ndarray, shape (dimension, N)
    dimension: size of the state vector at each time step
    N: number of time points
    Delta_t: float, time step size

    Returns
    -------
    np.ndarray, shape (dimension,), the time-integrated result
    """
    dimension, N = integrand.shape[0], integrand.shape[1]
    s = np.zeros(dimension, dtype=complex)
    t = N * Delta_t
    n = N
    if N%2 == 0:
        s += (3*Delta_t/8) * (integrand[:, N-4] + 3*(integrand[:, N-3]+integrand[:, N-2]) + integrand[:, N-1])
        n = N - 3

    for j in range(n):
        if j == 0 or j == n-1:
            s += integrand[:, j]
        elif j % 2 == 1:
            s += 4*integrand[:, j]
        else:
            s += 2*integrand[:, j]
    s *= Delta_t/3
    return s
