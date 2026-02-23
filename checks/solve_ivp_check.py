#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checking the solve_ivp numpy function
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


t_span = (0,10)
def  exponent_decay(t,y):
    return -0.5*y

y0 = [2]
t_vec = np.linspace(0,10,2**8)
sol = solve_ivp(exponent_decay,t_span,y0,method = 'RK45',t_eval = t_vec)
fig = plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.legend("line")
plt.title('y vs x')
plt.plot(sol.t,sol.y[0,:])
plt.show()