#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple wave eq solver
Solves a standard wave equation with periodic boundary conditions, using finite
difference and fourier spectral based solvers
Assuming an eigenvector as an initial condition the result is benchmarked w.r.t
the analytical solution.

"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utilities.derivative_with_fft import deriv_n_gen
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Global Parameterss
L = 1.0         # Length of the domain
Nx = 2**10        # Number of spatial points
c = 0.01         # Wave speed
x = np.linspace(0, L, Nx, endpoint=False)
dx = x[1] - x[0]


## Initial conditions

def initial_displacement(x):
    return np.sin((2*np.pi * x))  # Example: eigenvector

def initial_gaussian(x,mu,sig):
    N = 1/np.sqrt(2*np.pi*sig**2)
    return N*np.exp(-(x-mu)**2/(2*sig**2))

def initial_gaussian_velocity(x,mu,sig):
    return np.zeros(x.shape)

def gauss_t(x,mu,sig,c,t):
    N = 1/np.sqrt(2*np.pi*sig**2)
    center = (mu + c*t) % L
    return N*np.exp(-(x-center)**2/(2*sig**2))
    
def gauss_analyt_sol(x,mu,sig,c,t):
    return (gauss_t(x,mu,sig,c,t) + gauss_t(x,mu,sig,c,-t))/2
        
def initial_velocity(x):
    return -2*np.pi*c*np.cos((2*np.pi * x))

# Second spatial derivative using central differences
def laplacian(u, dx):
    '''Finite difference with periodic boundary conditions '''
    d2u = np.zeros_like(u)
    d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    d2u[0] = (u[1] - 2*u[0] + u[-1]) / dx**2
    d2u[-1] = (u[0] - 2*u[-1] + u[-2]) / dx**2
    return d2u

# ODE system: y = [u, v]
def wave_eq(t, y):
    u = y[:Nx]
    v = y[Nx:]
    du_dt = v
    dv_dt = c**2 * laplacian(u, dx)
    return np.concatenate([du_dt, dv_dt])

def laplacian_fft(u, x):
    '''Finite difference with periodic boundary conditions '''
    return deriv_n_gen(u,x,2).real

def wave_eq_fft(t, y):
    u = y[:Nx]
    v = y[Nx:]
    du_dt = v
    dv_dt = c**2 * laplacian_fft(u, x)
    return np.concatenate([du_dt, dv_dt])


def test_wave_eq_solver(L=L, Nx=Nx, c=c, x=x, PLOT=False):
    """Tests the wave equation solver against an analytical solution for a Gaussian initial state and zero initial velocity."""
    # Initial state
    mu = 1/2
    sig = L/10
    u0 = initial_gaussian(x, mu, sig)
    v0 = initial_gaussian_velocity(x, mu, sig)
    y0 = np.concatenate([u0, v0])
    
    # Time span
    k0 = np.pi/L
    omega0 = c*k0
    T0 = np.pi/omega0
    t_span = (0, T0)
    t_eval = np.linspace(0,t_span[1],30)

    # Solve
    sol = solve_ivp(wave_eq, t_span, y0, t_eval=t_eval, method='RK45')  # standard finite difference solver
    sol_fft = solve_ivp(wave_eq_fft, t_span, y0, t_eval=t_eval, method='RK45')  # fourier spectral solver

    # Plot solution at a few time steps
    i = 4
    t = t_eval[i]

    
    if PLOT:
        plt.figure(figsize=(10, 6))
        plt.plot(x, sol.y[:Nx, i], label=f't = {t_eval[i]:.2f}')
        plt.plot(x, gauss_analyt_sol(x, mu, sig, c, t))  # analytical solution for a guassian with initial zero velocity    
        plt.plot(x, sol_fft.y[:Nx, i], label=f'FFT t = {t_eval[i]:.2f}', linestyle='dashed')
        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("Wave Equation Solution over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    assert np.allclose(sol.y[:Nx, i], gauss_analyt_sol(x, mu, sig, c, t), atol=1e-2), "Difference between the numeric and anlytical result"
    assert np.allclose(sol_fft.y[:Nx, i], gauss_analyt_sol(x, mu, sig, c, t), atol=1e-2), "Difference between the numeric and anlytical result"

if __name__ =='__main__':
    test_wave_eq_solver()
    