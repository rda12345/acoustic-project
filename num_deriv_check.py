#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checking the performance of deriv_n_gen
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from derivative_with_fft import deriv_n_gen


def gaussian(x,mu,sig):
    '''Returns a gaussian with mean mu and standard deviation sig'''
    return (1/(np.sqrt(2*np.pi*sig**2))) * np.exp(-np.power(x-mu,2)/(2*sig**2))

L = 1
x = np.linspace(0,L,2**8)
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


plt.figure()
## first derivative
#plt.plot(x,dy_analyt,label = "analytic")
#plt.plot(x,dy,label = "numeric")

## second derivative

plt.plot(x,dy2)
plt.plot(x,ddy2_analyt,label = "analytic")
plt.plot(x,ddy2,label = "numeric")

plt.legend()
plt.show()






