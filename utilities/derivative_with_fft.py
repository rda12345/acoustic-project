"""
The file contains two functions which evaluate the derivative of a 1D function.
They differ by the conditions on the boundaries.

deriv_n: works if the funciton has finite support, meaning f(min(x)) = f(max(x)) = 0
deriv_n_gen: works for general boundary conditions. 
"""
import numpy as np
import math

def deriv_n(func, x, n):
    """
    Evaluates the n'th derivative of f applying fast foureir transform.
    f must vanish at the domain boundaries: f(min(x)) = f(max(x)) = 0
    
    Parameters
    ----------
    
        func: np.ndarray (N,), the function which derivative is evaluated
        x: np.ndarray (N,), the domain of the function
        n: int, the derivative order
        N: int, grid size
    
    Returns
    -------
        df, np.ndarray (N,)
    """
    N = len(x)
    max_x = max(x)
    min_x= min(x)
    fftx = np.fft.fft(func)
    vec = np.array(list(range(0,N//2))+[0]+list(range(-N//2+1,0)))
    pi = math.pi
    k = 2*pi/(max_x-min_x)*vec  #[0:N/2-1, 0, -N/2+1:-1])
    dffft = ((1j*k)**n)*fftx 
    df = np.fft.ifft(dffft)
    return df


def deriv_n_gen(func, x,n):
    """
    Evaluates the n'th derivative of f applying fast foureir transform.
    The function generalizes deriv_n function by substracting a linear function
    from f in the begining and then correcting the derivative in the end.
    
    Evaluates the n'th derivative of f applying fast foureir transform.
    f must vanish at the domain boundaries: f(min(x)) = f(max(x)) = 0
    
    Parameters
    ----------
    
        func: np.ndarray (N,), the function which derivative is evaluated
        x: np.ndarray (N,), the domain of the function
        n: int, the derivative order
    
    Returns
    -------
        df, np.ndarray (N,)
    """
    y1 = func[0]
    y2 = func[-1]
    delta_x = x[-1]-x[0]
    if abs(delta_x) < 1e-13:
        slope = 0
    else:
        slope = (y2-y1)/delta_x
    linear_correction = slope if n == 1 else 0
    lin_func = slope*(x-x[0])+y1
    func2 = func - lin_func
    dfn = deriv_n(func2,x,n)
    df = dfn + linear_correction * np.ones(len(x))
    return df



