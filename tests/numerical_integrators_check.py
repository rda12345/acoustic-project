#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical integrators
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.integrate import quad


def g(x,mu,sig,A):
    '''Returns a gaussian with mean mu and standard deviation sig'''
    return A*(1/(np.sqrt(2*np.pi*sig**2))) * np.exp(-np.power(x-mu,2)/(2*sig**2))

A = 2
sig = 1
t0 = 0
tf = 2
t = np.linspace(-tf,tf,2**8)
f = A*g(t,t0,sig,A)
def F_analyt(tf):
    return (A/2)*(special.erf((tf-t0)/(np.sqrt(2)*sig))+special.erf(t0/np.sqrt(2)*sig))

check = A/2

F_numeric = quad(g, 0, tf, args=(t0,sig,A))
print(f'analytic: {F_analyt(tf)}')
print(f'check: {check}')

print(f'numeric: {F_numeric}')

#def inv_deriv_source(t,x,mu,sig):
 #   '''Returns the anti-derivative of a gaussian function, the erf function at
  #      time t.'''
        

check1 = special.erf(10)
print(f'check 1: {check1}')
