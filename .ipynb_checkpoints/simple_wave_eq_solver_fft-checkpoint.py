#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple wave eq solver
Solves a standard wave equation with periodic boundary conditions, using finite
difference solver.
Assuming an eigenvector as an initial condition the result is benchmarked w.r.t
the analytical solution.

Analytics: appears as method I in wave_eq_solver.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from derivative_with_fft import deriv_n_gen

# Parameters
L = 1.0         # Length of the domain
Nx = 2**10        # Number of spatial points
c = 0.01         # Wave speed
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]

# Initial conditions


def initial_gaussian(x,mu,sig):
    N = 1/np.sqrt(2*np.pi*sig**2)
    return N*np.exp(-(x-mu)**2/(2*sig**2))

def initial_gaussian_velocity(x,mu,sig):
    return np.zeros(x.shape)

def gauss_t(x,mu,sig,c,t):
    N = 1/np.sqrt(2*np.pi*sig**2)
    return N*np.exp(-(x-mu-c*t)**2/(2*sig**2))
    
def gauss_analyt_sol(x,mu,sig,c,t):
    return (gauss_t(x,mu,sig,c,t) + gauss_t(x,mu,sig,c,-t))/2
    

# Second spatial derivative using fft
def laplacian(u, x):
    '''Finite difference with periodic boundary conditions '''
    return deriv_n_gen(u,x,2)

# ODE system: y = [u, v]
def wave_eq(t, y):
    u = y[:Nx]
    v = y[Nx:]
    du_dt = v
    dv_dt = c**2 * laplacian(u, x)
    return np.concatenate([du_dt, dv_dt])

if __name__ =='__main__':
    # Initial state
    #u0 = initial_displacement(x)
    #v0 = initial_velocity(x)
    
    mu = 1/2
    sig = L/10
    u0 = initial_gaussian(x, mu, sig)
    v0 = initial_gaussian_velocity(x, mu, sig)
    y0 = np.concatenate([u0, v0])
    
    # Time span
    k0 = np.pi/L
    omega0 = c*k0;
    T0 = 2*np.pi/omega0
    t_span = (0, T0)
    t_eval = np.linspace(0,t_span[1],30)

    # Solve
    sol = solve_ivp(wave_eq, t_span, y0, t_eval=t_eval, method='RK45')

    # Plot solution at a few time steps
    plt.figure(figsize=(10, 6))
    i = 2
    plt.plot(x, sol.y[:Nx, i], label=f't = {t_eval[i]:.2f}')
    t = t_eval[i]
    k = 2*np.pi/L
    omega = c*k
    
    ## analytical solution for an eigenvector
    #plt.plot(x,np.sin(k*x-omega*t),label = 'analytical')
    
    ## analytical solution for a guassian with initial zero velocity
    plt.plot(x,gauss_analyt_sol(x, mu, sig, c, t))
    
    plt.title("Wave Equation Solution over Time")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.legend()
    plt.grid(True)
    plt.show()
    

    
    