"""
Chebychev propagator with a source term. This is used for the forward simulation.
"""

import numpy as np
from utilities.utility_functions import besseli

def compute_cheby_coefficients(lam_min: float, lam_max: float, dt: float, Nmax: int) -> np.ndarray:
    """
    Evaluates the Chebychev coefficients for the propagator exp(O*dt),
    where O is a linear differential operator with units of 1/time.
    Uses modified Bessel functions as in Kosloff (1994) style formation.
        
    Note: lam_min and lam_max do not have to be exact. Although it is important 
            keep the normalized range of eigenvalues within [-1,1].
    
    Parameters
    ----------
    lam_min: float, the minimum eigenvalue of the operator O
    lam_max: float, the maximum eigenvalue of the operator O
    dt: float, the time interval of propagation.
    Nmax: int, maximum number of coefficients to compute

            
    Returns
    -------
    ndarray, contains the Chebychev coefficients, which can be used to
            propagate the system.      
    """
    dE = lam_max-lam_min
    R = dE * dt/2  # factor to adjust the eigenvalues to be between [-1,1]
    c = np.zeros(Nmax,dtype = complex)
    c[0] = besseli(0,R)           # zero coefficient.
    # generate until either Nmax or coefficients become negligible  
    n_final = Nmax - 1  # default to full range
    for n in range(1, Nmax):
        c[n] = 2.0*besseli(n,R)
        if abs(c[n])<1e-17 and n>R:
            n_final = n
            break
    # use only the significant coefficients
    actual_Nmax = n_final + 1
    c = c[:actual_Nmax]
    # multiply by global exponential factor which normalizes the coefficients
    d = np.exp((lam_min+lam_max)*dt/2.0)*c
    return d
    

        
def propagate_with_source(vec: np.ndarray, generator: callable, source: callable, info: tuple, Nt: int) -> np.ndarray:
    """
    Evaluates exp(O*t)*vec, with t=Nt*dt, utilizing a Chebychev expansion of the dynamical propagator.
    Updates the system state.  
    
    Parameters
    ----------
    vec: np.ndarray (2*size,), containing the initial state
    generator: function, the dynamical generator (e.g, a differential operator)
    source: callable, the source function
    info: tuple, containing the dynamical generator additional parameters
    Nt, int, the number of time-steps
        
    Returns: 
    -------
    np.ndarray (2*size,), field's state at final time
    """
    lam_min, lam_max, dt, Nmax = info
    d = compute_cheby_coefficients(lam_min=lam_min, lam_max=lam_max, dt=dt, Nmax=Nmax)
    fi = np.zeros((vec.shape[0],3),dtype = complex)
    for j in range(Nt):     # running over the time steps
        vec = propagation_step(vec,fi,generator,d)
    source_term = integrate_source_term(generator, source, info, Nt, dt) 
    return vec + source_term
    

def integrate_source_term(generator: callable, source: np.ndarray, info: tuple, Nt: int, dt: float) -> np.ndarray:
    """
    Integrates the source term over time using Simpson's rule, while propagating the system state.
    """
    lam_min, lam_max, dt, Nmax = info
    d = compute_cheby_coefficients(lam_min=lam_min, lam_max=lam_max, dt=dt, Nmax=Nmax)
    dimension = source(0).shape[0]
    fi = np.zeros((dimension,3),dtype = complex)    
    if Nt%2 != 0:
        raise ValueError("integrate_source_term: Nt must be even for Simpson's rule.")
    
    s = np.zeros(dimension, dtype=complex)
    t = Nt*dt
    for Ntau in range(Nt+1):
        tau = Ntau*dt
        vec = source(t-tau)
        for _ in range(Ntau):     # Running over the time steps
            vec = propagation_step(vec, fi, generator, d) 
        if Ntau == 0 or Ntau == Nt:
            s += vec
        elif Ntau % 2 == 1:
            s += 4*vec
        else:
            s += 2*vec
    s *= dt/3
    return s

def propagation_step(vec: np.ndarray, fi: np.ndarray, generator: callable, d: np.ndarray) -> np.ndarray:
    """
    Performs a propagation step of size dt
        
    Parameters
    ----------
    vec: np.ndarray (model.size,), the current system state
    fi: np.ndarray (3,(2*model.size,), introduced to prevent reallocation of memory each propagation step
    generator: callable, the dynamical generator (e.g, a differential operator), assumed to be normalized
                        that its eigenvalues are in the range [-1, 1]
    d: np.ndarray, the Chebychev coefficients for the propagator exp(O*dt)

    Returns
    -------
    np.ndarray: propagated system state
    """
    fi[:,0] = vec
    # The normalized differential operator O
    fi[:,1] = generator(vec)              
    actual_Nmax = len(d)  # Use actual number of coefficients
    cheb_sum = d[0]*fi[:,0]+ d[1]*fi[:,1]
    for i in range(1,actual_Nmax-1):
        fi[:,2] = 2*generator(fi[:,1])-fi[:,0]
        fi[:,0], fi[:,1] =fi[:,1], fi[:,2]
        cheb_sum = cheb_sum + d[i+1]*fi[:,2]
    return cheb_sum.copy()

    


    




