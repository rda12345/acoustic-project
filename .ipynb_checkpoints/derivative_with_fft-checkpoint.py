#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file contains two functions which evaluate the derivative of a 1D function.
They differ by the conditions on the boundaries.

deriv_n: works if the funciton has finite support, meaning f(min(x)) = f(max(x)) = 0
deriv_n_gen: works for general boundary conditions. 
"""
import numpy as np
import math
#import pylab

def deriv_n(func,x,n):
    '''
    Evaluates the n'th derivative of f applying fast foureir transform.
    f must vanish at the domain boundaries: f(min(x)) = f(max(x)) = 0
    
    func: array, the function which derivative is evaluated
    x: array, the domain of the function
    n: int, the derivative order
    '''
    N=len(x)
    max_x = max(x)
    min_x= min(x)
    fftx = np.fft.fft(func)
    vec = np.array(list(range(0,N//2))+[0]+list(range(-N//2+1,0)))
    pi = math.pi
    k = 2*pi/(max_x-min_x)*vec  #[0:N/2-1, 0, -N/2+1:-1])
    dffft = ((1j*k)**n)*fftx 
    df = np.fft.ifft(dffft)
    return df


def deriv_n_gen(func,x,n):
    '''
    Evaluates the n'th derivative of f applying fast foureir transform.
    The function generalizes deriv_n function by substracting a linear function
    from f in the begining and then correcting the derivative in the end.
    '''
    y1 = func[0]
    y2 = func[-1]
    slope = (y2-y1)/(x[-1]-x[0])
    lin_func = slope*(x-x[0])+y1
    func2 = func - lin_func
    df2 = deriv_n(func2,x,n)
    df = df2 +slope*np.ones(len(x));
    return df


# # Test
# sig = 1
# dx = 0.01
# x = np.arange(-10,10+dx,dx)
# y = np.exp(-x**2/(2*sig**2))
# diffy = (-x/(sig**2))*y
# dy = deriv_n(y,x,1)

# pylab.figure(1)
# pylab.plot(x,diffy,'b')
# pylab.plot(x,dy,'r--')
# pylab.show()
