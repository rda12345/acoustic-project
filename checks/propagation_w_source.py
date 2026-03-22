"""
Tests the solution in chebychev_propagator_w_source
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utilities.chebychev_propagator_w_source import *
from simulation.acoustic_model import AcousticModel
from simulation.chebyshev_propagator import ChebyshevPropagator
import matplotlib.pyplot as plt
from utilities.utility_functions import gaussian, simpson_integrator
from scipy.special import erf

if __name__ == "__main__":
    from scipy.linalg import expm
    print('-------------- TESTS --------------')

    # Dynamical parameters
    T0 = 1.0
    Nt = 2**8
    dt = T0/(Nt+1)


    
    ## Test 1
    generator = lambda x: -x  # Example generator (e.g., a simple decay)
    
    a = 2.0
    # Source term
    def source(t, a = a):
        return np.array([a])

    # Generator parameters
    lam_min = -1.0
    lam_max = 1.0
    Nmax = 50
    info = (lam_min, lam_max, dt, Nmax)   

    initial_state = np.array([1.0])  # Example initial state
    result = propagate_with_source(initial_state, generator, source, info, Nt = Nt)
    analytic_result = (initial_state - a) * np.exp(-T0) + a  # Analytical solution for the example generator and source
    print(f'Scalar function, error: {np.linalg.norm(analytic_result - result[0])}')


    ## Test 2: Scalar function with a time-dependent source term
    # Source term
    b = 0.5
    omega = 0.1
    def source(t, b=b, omega=omega):
        return np.array([b * np.cos(omega * t)])
    lam_min = -2.0
    lam_max = 2.0
    Nmax = 50
    info = (lam_min, lam_max, dt, Nmax)   

    initial_state = np.array([1.0])  # Example initial state
    result = propagate_with_source(initial_state, generator, source, info, Nt = Nt)
    temp = (omega * np.sin(omega * T0) + a * np.cos(omega * T0))
    c_1 = initial_state - (a*b/(a**2 + omega**2))  
    analytic_result = np.exp(-a * T0) * c_1 +  (b/(a**2 + omega**2))*temp # Analytical solution for the example generator and source
    print(f'Scalar function with a time-dependent source, error: {np.linalg.norm(analytic_result - result[0])}')


    ## Test 3: Rabi Oscillations
    
    # Generator parameters
    omega = 0.1
    epsilon = 0.3
    sigma_x = np.array([[0, 1],[1, 0]])
    sigma_z = np.array([[1, 0],[0, -1]])
    H = omega * sigma_z + epsilon * sigma_x

    Omega = np.sqrt(omega**2 + epsilon**2)
    generator = lambda vec:  - 1j * H @ vec/Omega
    lam_min, lam_max = - Omega, Omega 
    Nmax = 50
    info = (lam_min, lam_max, dt, Nmax)

    # Vanishing source term
    def source(t: float, a: float = 0):
        return np.array([0, a])
       
    initial_state = np.array([1.0, 0.0])  # Example initial state
    result = propagate_with_source(initial_state, generator, source, info, Nt = Nt)
    analytic_result = expm(-1j * H * T0) @ initial_state    

    print(f'Vector propagation with vanishing source, error: {np.linalg.norm(analytic_result - result)}')
    
    ## Test 4: Vector differential equation with a constant source term
    
    # Generator parameters
    lam = 0.3
    H = lam * sigma_x

    generator = lambda vec:  - 1j * H @ vec/lam
    lam_min, lam_max = - lam, lam 
    Nmax = 50
    info = (lam_min, lam_max, dt, Nmax)

    a = 0.0
    def source(t: float, a: float = a):
        return np.array([a, 0])
       
    initial_state = np.array([1.0, 1.0])  # Example initial state
    result = propagate_with_source(initial_state, generator, source, info, Nt = Nt)
    integral = (a/lam) * np.array([np.cos(lam * T0), -1j * np.sin(lam * T0)])  # Integral of the source term over time
    analytic_result = expm(-1j * H * T0) @ initial_state + integral   
    print(f'Vector propagation with a constant source, error: {np.linalg.norm(analytic_result - result)}')




    ## Test 5: Acoustic wave equation with a vanishing source term
    
    size = 2**8
    L = 1.0  # sets the length scale of the problem
    
    dt = 0.05
    T0 = 20

    model = AcousticModel(size = size,L = L)
    base_speed = 0.01
    amp_speed = 0.6
    
    sig = model.L/20
    amp = 1.0
    model = AcousticModel(size = 2**8, L = 1.0)
    speed_field, initial_state = model.gaussian_initial_state(amp, sig, base_speed, amp_speed)
    model.initialize(speed_field, initial_state)
    propagator = ChebyshevPropagator(model=model, T0=T0)
    propagator.propagate()
    
     
    initial_pressure = initial_state[:size]
    final_state = model.get_state()
    final_pressure = final_state[:size]
    


    Nt = 400
    dt = T0/(Nt+1)
    model_2 = AcousticModel(size = size, L = L)
    model_2.initialize(speed_field, initial_state)
    propagator_2 = ChebyshevPropagator(model=model_2, dt=dt, T0=T0)
    propagator_2.propagate_with_source(source = lambda x: np.zeros(2*size))
    final_state_2 = model_2.get_state()
    final_pressure_2 = final_state_2[:size]
    print(f'Wave equation, error: {np.linalg.norm(final_state - final_state_2)}  ')
    


    ## Test 6: Acoustic wave equation with a monochromatic source term
    
    size = 2**8
    L = 1.0  # sets the length scale of the problem
    T0 = 1.0
    c = 0.01    # wave speed
    speed_field = c * np.ones(size)
    initial_state = np.zeros(2*size)
    initial_pressure = initial_state[:size]

    # Initialize model
    model = AcousticModel(size=size, L=L)
    model.initialize(speed_field, initial_state)

    Nt = 400
    dt = T0/(Nt+1)
    # Defining the source term
    A = 0.1

    grid = model.grid
    n = 1
    k = 2*np.pi*n/L
    omega_k = c*k
    omega = 0.2
    def monochromatic_source(t: float) -> np.ndarray:
        s = A * np.sin(k*grid) * np.cos(omega*t)
        return np.concatenate([np.zeros(size), s])

    def non_resonant_analytic_result(t):
        coeff = (A/(omega_k**2 - omega**2))
        pressure = (coeff * (np.cos(omega*t)-np.cos(omega_k*t))) * np.sin(grid*k)
        return pressure


    # propagation
    propagator = ChebyshevPropagator(model=model, dt=dt, T0=T0)
    propagator.propagate_with_source(source=monochromatic_source)
    final_state = model.get_state()
    final_pressure = final_state[:size]

    analytical_result = non_resonant_analytic_result(T0)

    plt.figure()
    plt.plot(grid, final_pressure.real, '.-', label="numerical")
    plt.plot(grid, analytical_result.real, label="analytical")
    plt.xlabel("position")
    plt.ylabel("pressure")
    plt.legend()
    plt.show()

    # print(f"Wave equation with a monochromatic source, error: {np.sum(np.abs(final_pressure-analytical_result))}")
    


    ## Test 7: Acoustic wave equation with a delta function in time source term
    
    size = 2**8
    L = 1.0  # sets the length scale of the problem
    T0 = 1.0
    c = 0.01    # wave speed
    speed_field = c * np.ones(size)
    initial_state = np.zeros(2*size)
    
    Nt = 400
    dt = T0/(Nt+1)
    # Defining the source term
    A = 0.1

    
    mu = L/4
    sig = L/20
    # Initialize model
    model = AcousticModel(size = size, L = L)
    model.initialize(speed_field, initial_state)
    grid = model.grid

    f = (2/dt)*gaussian(x=grid, mu = mu, sig=sig)   # normalized so to have the correct kick amplitude (dt/2)*f
    
    def delta_source(t: float) -> np.ndarray:
        if np.isclose(t,0):
            s = f 
        else:
            s = np.zeros(size)
        return np.concatenate([np.zeros(size), s])

    def delta_analytic_result(t):
        b = (grid + c*t - mu)/(sig * np.sqrt(2))
        a = (grid - c*t - mu)/(sig * np.sqrt(2))
        u =  (1/(4*c))*(erf(b)-erf(a))
        return u


    # propagation
    propagator = ChebyshevPropagator(model=model, dt=dt, T0=T0)
    propagator.propagate_with_source(source=delta_source)
    final_state = model.get_state()
    final_pressure = final_state[:size]

    analytical_result = delta_analytic_result(T0)
    
    # plt.figure()
    # plt.plot(grid, final_pressure.real, '.-', label="numerical")
    # plt.plot(grid, analytical_result.real, label="analytical")
    # plt.xlabel("position")
    # plt.ylabel("pressure")
    # plt.legend()
    # plt.show()

    print(f"Wave equation with a delta source, error: {np.sum(np.abs(final_pressure-analytical_result))}")
    

   

    












