#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isotropic acoustic wave equation. 
Following the theory in adjoint_state_method.lyx (simple model section)
The isotropic acoustic wave equation is expressed in terms of two first order
coupled partial differential equations. 
These are solved employing thy scipy function solve_ivp 

TASKS:
    1. Compare the solution to an analytical solution for a limiting case.
    2. Obtain the anti-derivative of the source by using a numeric method.
    3. Compare the analytic and numeric results and have the propagator only
        get the source function, the anti-derivative will than be evaluated and
        utilized in the initialization method.
        i.e. solve with the analytic solution and check the results
        solve with the numeric one
        compare the two results.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from derivative_with_fft import deriv_n_gen
import matplotlib.pyplot as plt


class realModel(object):
    
    def __init__(self,size):
        '''Initialization of the model
            
            Input
                size: int, the number of points in the 1D space grid
            
            Parameters
            L: float, length of the grid    
            grid: array, the space grid of the proplem in the range [0,:L]
                        with a total of size equally distributed points
                K_0: array, the bulk modulus of compressability
        '''
        self.size = size
        self.L = 1.0   # sets the aribitrary length scale of the problem. 
        self.grid =  np.linspace(0,self.L,self.size)     # The grid is [0,L] with an inteval of length step
        self.K_0 = np.ones(self.grid.shape)  # the bulk modulus of compressability
    
    
    def initialization(self,source = lambda x: 0,amp = 1, mu = None, sig = None, base_speed = 0.01, base_pressure = 2):
        '''Intialization of the model with a constant wave-velocity field and 
            a either a constant pressure field or a gaussian if the mean (mu)
            and standard deviation (sig) are given.
            
            Input
                source: function, depends only on time. The time-dependence of the laser pulse
                        assumed to be a delta function in the position starting a
                amp: float, amplitude (the gaussian is normalized and multiplied by amp)
                mu: float, the center of the gaussian
                sig: float, the standard deviation of the gaussian
                
                
            Parameters
                speed: array, the wave velocity vector 
                velocity: array, the particle velocity vector
                pressure: array, the pressure vector (corresponding to the space (grid) vector)
                state: array, the total state vector concatenating the pressure and velocity vectors
                rho_0: array, background (stationary) desity

        '''
        self.speed = base_speed * np.ones(self.grid.shape) #+ gaussian(self.grid,0,sig)     # the true model
        lam = 2*self.L    # lam should be between smaller than L and bigger than the grid spacing
        k0 =2*np.pi/lam;
        if mu:
            #self.pressure = amp*gaussian(self.grid,mu,sig)*np.exp(1j*k0*self.grid)+base_pressure
            self.pressure = np.exp(1j*2*(k0*self.grid))
            #self.velocity = amp*gaussian_dot(self.grid,mu,sig,base_speed)*np.exp(1j*k0*self.grid)\
             #   -1j*k0*base_speed*amp*gaussian(self.grid,mu,sig)*np.exp(1j*k0*self.grid)
            #self.velocity = amp*gaussian(self.grid,mu,sig)*np.exp(1j*k0*self.grid)*((self.grid-mu)/sig**2 + 1j*k0)*base_speed
            self.velocity = -base_speed*2*k0*self.pressure 
            #self.velocity = np.zeros_like(self.grid)
        else:
         #   self.pressure = base_pressure*np.ones(self.grid.shape)
             self.velocity = np.zeros_like(self.grid)
        self.state =  np.concatenate((self.pressure,self.velocity))      # initial state (p(x,0),u(x,0))
        self.rho_0 = np.power(self.K_0/self.speed,2)  # setting the background density
        self.source = source
      
        
   
        
        
    def propagator(self, t_eval = None, t_span = None):
        '''Propagates the wave equation'''
        if t_span is None:
            t_span = (0,t_eval[-1])          # note that the initial time is always t=0
        parameters = (self.grid,self.K_0,self.rho_0,self.source)
        sol = solve_ivp(dynamics,t_span,self.state,args = parameters,t_eval = t_eval)
        return sol
    
    
    def sample(self,points):
        '''Sample function recieves a list of tuples, each representing a space-time point
            and samples the pressure field of at these points. The points correspond to 
            measurements of the pressure.
            
            Input
                points: list of tuples, each tuple contains a space time points (x,t),
                        where x and t are floats.
                        
            Returns
                measurements: dictionary, mapping space-time points (x,t) to pressure values
        ''' 
        # For each point  
        measurements = {}
        t_eval_list = []
        # Creating t_eval
        for point in points:    
            if point[1] not in t_eval_list:
                t_eval_list.append(point[1])
        t_eval = np.array(t_eval_list).T
        
        # Solving the dynamics and displaying the solution at the time points t_eval
        sol = self.propagator(t_eval)
        
        # For each space-time point construct a dictionary mapping the space-time points to the 
        # real (measured) pressure.
        for point in points:
            pos, t = point
            pos_index = np.where(self.grid == pos)[0][0]
            measurements[point] = sol.y[pos_index,t_eval_list.index(t)]          
        return measurements
   

def gaussian(x,mu,sig):
    '''Returns a gaussian with mean mu and standard deviation sig'''
    return (1/(np.sqrt(2*np.pi*sig**2))) * np.exp(-np.power(x-mu,2)/(2*sig**2))


def gaussian_dot(x,mu,sig,c):
    '''Returns a gaussian with mean mu and standard deviation sig'''
    return (c*(x-mu)/sig**2)*gaussian(x,mu,sig)

def source(t):
    mu = 0.4
    sig = 0.05
    amp = 0.1
    return amp*gaussian(t,mu,sig)
        
       

        
def dynamics(t,state,grid,K_0,rho_0,source):
    '''Returns the derivative of np.array([p,u]).T , where
        p and u are the pressure and velocity vectors
        
        Input
            state: tuple, (pressure vector, velocity vector), each an array of the grid size
            grid: array, the position (1D) grid
            K_0: float, is the bulk modolus of compressability
            rho_0: float, the background (stationary) density
            S_func: function, the anti-partial derivative w.r.t to time of the source
    '''
    n = grid.shape[0]
    p = state[0:n]
    u = state[n:2*n]
    c = np.sqrt(K_0/rho_0)
    #S_func = source(t)*delta(grid.shape)
    #dp_dt = - K_0 * deriv_n_gen(u,grid,1) #+ S_func
    #du_dt = - (1/rho_0) * deriv_n_gen(p,grid,1)       
    dp_dt = u
    #du_dt = c**2 * laplacian(u, dx)
    du_dt = c**2 * deriv_n_gen(deriv_n_gen(p,grid,1),grid,1)
    dstate = np.concatenate((dp_dt,du_dt))

    return dstate
     
    

def delta(grid_shape):
    delta = np.zeros(grid_shape)
    delta[100] = 1
    return delta
    
def Plot(x,y,x_axis_label = None,y_axis_label = None,label = None):
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.plot(x,y,label = label)
    plt.legend()
    plt.show()

def simple_no_source_solution(grid,c,t):
    # COMPLETE CODE OF AN ANALYTICAL SOLUTION
    pass

if __name__ == '__main__':
    
    ## Parameters
    size = 2**8
   
    
    ## Numerical solution
    model = realModel(size)
    
    mu = model.L/2
    sig = model.L/20
    amp = 0.1
    
    # Initialization of an initial gaussian pressure distribution and no source
    model.initialization(amp = amp, mu = mu,sig = sig)
    
    # Dynamics parameters
    c = model.speed[0]
    k0 = np.pi/model.L
    omega = c*k0
    T0 = 2*np.pi/omega
    t_end = T0
    t_span = [0,t_end]
    
    # Initialization of a constant initial pressure distribution with a
    # gaussian source
    # NOTE: This solution led to a instable numerical solution, i.e., wierd increase and 
    # decrease in the pressure in the sides of the grid. Since a guassian pulse, just creates
    # a gaussian pressure distribution, initializing a moving pulse in one of the sides is equivalent.
    # model.initialization(amp = amp, source = source)
    #t_vec = np.linspace(0,t_end,4)
    t_vec = T0*np.linspace(0,0.1,10)
    sol = model.propagator(t_eval = t_vec)
    

    ## Plotting the dynamics at various time steps
    
    # analyt_sol = simple_no_source_solution(model.grid, c, sol.t[t_index])
    
    fig = plt.figure(1)
    n = model.grid.shape[0]
    for t_index in range(0,sol.t.shape[0]):
    #t_index = 1
        plt.plot(model.grid,sol.y[0:n,t_index],label = round(sol.t[t_index],2))
    plt.xlabel('x')
    plt.ylabel('pressure')
    #plt.plot(model.grid,model.pressure,label = "initial")
    #plt.plot(model.grid,analyt_sol, label = "analytic")
    plt.legend()
    plt.show()
    
    '''
    ## Sampling the solution, assuming two detectors at x = 0.1 and x = L - 0.1
    
    # positions of the sampling points
    x1 = model.grid[size//6]
    x2 = model.grid[size*5//6]
    
    # The sample times (times the measurements where performed)
    sample_t_vec = np.linspace(0.7,0.9,10)
    #t_vec = np.array([0.7])
    points = []
    for t in sample_t_vec:
        points.append((x1,t))
        points.append((x2,t))
          
    measurements = model.sample(points)
    '''
    #print(measurements)
    '''
    plt.figure(2)
    Plot(model.grid,np.real(model.velocity),x_axis_label = 'x',y_axis_label = 'velocity')
    plt.figure(3)
    Plot(model.grid,np.imag(model.velocity),x_axis_label = 'x',y_axis_label = 'velocity')
    plt.figure(4)
    Plot(model.grid,model.pressure,x_axis_label = 'x',y_axis_label = 'pressure')

    
    '''
    plt.figure(2)
    Plot(model.grid,sol.y[0:n,0],x_axis_label = 'x',y_axis_label = 'pressure')
