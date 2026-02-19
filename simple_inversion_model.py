#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full waveform inversion. 
Overview: 
    The following program solves the acoustic inverse problem of a 1D system, 
    The medium dynamics are govened by the dissipationless acoustic equation.



Description: 
    The program is structured in the following way. A "model" class incorporates
    The following procedures.
    1. Initialization procedure, setting up the details of the model,
    the initial velocity and pressure fields (vectors), and the 
    initial guess of the wave speed field (vector). 
    2. Propatation procedure, which solves the system's equation of motion, 
        i.e., the isotropic acoustic wave equation, to obtain the pressure
        field (vector) at space-time point (x,t).
    3. Gradient descent (GD) and update speed field procedures,
        which apply the adjoint state method to 
        to evaluate the gradient of the cost (objective) function, and 
        utilize the gradient to update the speed field.
    4. Evaluate procedure, which compares the optimized speed field to the real
        result.
        
    Additional classes are:
    1. realModel, which sets a predefined (the real)
        speed field, solves the equation of motion to obtain the "real" pressure field.
        The pressure field is then sampled to obtain the ultrasound measurements.
    2. CostFunction, which gives the value and the analytic part of the derivative
        for a given speed field.
    
Coding road map
3. code and check evalution relative to the initialized model and the real model
4. code the gradient descent without the using the adjoint state vector,
    just update the model, knowing the real model parameters.
5. code the adjoint state method
    subtasks:
        - solve for the adjoint state
        - evaluate the gradiant


General Pseudo code for my use: 
    
    - Run a real model and get the true measurements
    
    - evaluate the error, the error function should propagate and update the predictions 
        then compare the them with the true measurements.
    Go in this loop until error < error_threshold
    - evaluate the gradiant: requires first solving the adjoint equation etc.
    - update the state
    - evaluate the error, the error function should propagate and update the predictions 
        then compare the them with the true measurements.
        
        
NOTE: MAYBE TURN THE MEASUREMENTS TO A DICTIONARY SINCE THESE POINTS ARE GETTING ACESSED A LOT...
ANd maybe have an input points to experiment and output observed data


Tasks: 
    1. Finish summarizing the theory lyx file of the adjoint state method.
    2. Solve for the adjoint state.
    3. Update the gradient
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy
from derivative_with_fft import deriv_n_gen
#from scipy.integrate import solve_ivp


class model(object):
    
    def __init__(self,size,T0):
        """
        Initialization of the model
            
        Input:
            size: int, the number of points in the 1D space grid
            T0: float, approximate duration of the experiment 
        Parameters:
            L: float, length of the grid    
            grid: array, the space grid of the proplem in the range [0,L]
                        with a total of size equally distributed points
            dt: float, time step size of the Chebychev expansion method
            Nt: int, number of iterations of the Chebychev method
            total_time: float, accumulated simulation time
            meas_positions: list, containing the positions of the array
            observed_data: dictionary, which maps space-time points to the associated
                        measurement result
                    
            Nmax: int, maximum number of Chebychev coefficients
        """
        self.size = size
        self.T0 = T0
        self.L = 1.0   # sets the aribitrary length scale of the problem. 
        self.grid =  np.linspace(0,self.L,self.size)     
        self.dt = 0.5 
        self.Nt = int(self.T0//self.dt) 
        self.total_time = self.dt*self.Nt
        self.meas_positions = []
        self.observed_data = {}
        self.Nmax = 1000  
        

        
    
    def initialization(self,speed_field,initial_state):
        """
        Intialization of the model with an initial gaussian wave packet
            
        Input
            amp: float, amplitude (the gaussian is normalized and multiplied by amp)
            mu: float, the center of the gaussian
            sig: float, the standard deviation of the gaussian
                
                
        Parameters
            speed_field: array, the wave velocity vector
            velocity: array, the particle velocity vector
            pressure: array, the pressure vector (corresponding to the space (grid) vector)
            state: array, the total state vector concatenating the pressure and velocity vectors
            
        """
        self.speed_field = speed_field   # the true model
        self.state =  initial_state
        dx =self.L/self.size  # space interval
        kmax = np.pi/dx          # maximum wavenumber
        self.lam_max = np.sqrt((np.max(self.speed_field)**2)*kmax**2+1) 
        self.lam_min = - self.lam_max
        self.dE =self.lam_max-self.lam_min  # range of the 'energy' spectrum, with units of 1/time
        
        
     
        
    def propagator(self):
        """
        Evaluates exp(O*t)*vec, with t=Nt*dt, utilizing a Chebychev expansion of the dynamical propagator.
        Updates the system state.

        Input: 
            vec: numpy array, containing the initial state
            tup: tuple, containing the dynamical generator additional parameters
            
        Parameters: 
            generator: function, the dynamical generator (e.g, a differential operator)
            d_j: numpy array, the Chebychev coefficients
            
        Returns: tuples containing:
            final_pressure_field: array, pressure field after time total_time
            final_velocity_field: array, velocity field after time total_time
        """
        vec = self.state
        d_j = self.cheby_coeff()
        fi = np.zeros((vec.shape[0],3),dtype = complex)
        #x,max_x,min_x,lx,mass,k,lam_min,lam_max,dk = tup
        for j in range(self.Nt):     # Running over the time steps
            fi[:,0] = vec
            # The normalized differential operator O
            fi[:,1] = self.generator(vec)              
            G1 = d_j[0]*fi[:,0]
            G2 = d_j[1]*fi[:,1]
            G_3 = G1+G2
            for i in range(1,self.Nmax-1):
                fi[:,2] = 2*self.generator(fi[:,1])-fi[:,0]
                fi[:,0]=fi[:,1]
                fi[:,1]=fi[:,2]
                G_3 = G_3+d_j[i+1]*fi[:,2]
            vec = G_3
        # updating the system state
        self.state = vec   
        final_pressure_field = self.state[:self.size]
        final_velocity_field = self.state[self.size:]
        return final_pressure_field, final_velocity_field
    
    
    
    def set_points(self,points = None):
        """
        Sets the space-time points (time,position) which correspond to the conducted
        measurement.
        
        Input:
            meas_positions: list, containing the positions of the detectors
            meas_times: list, containing the associated detection times
        
        Returns:
            points: list, containing space-time points (time,space) of the measurements
            
        """
        if points:
            self.points = points
        else:
            time_interv = self.Nt//5
            time_ind = np.arange(time_interv,self.Nt+1,time_interv)
            self.meas_times = [self.dt*ind for ind in time_ind]       # measurement times
            pos_ind_1 = self.size//4   
            pos_ind_2 = self.size*3//4 
            pos_1 = self.grid[pos_ind_1] # position of detector 1
            pos_2 = self.grid[pos_ind_2] # position of detector 2
            self.meas_positions = [pos_1,pos_2]
            self.points = [(time,pos) for time in self.meas_times for pos in self.meas_positions]
        return self.points
    
    def experiment(self):
        """
        Propagates the system state and returns measurements conducted space time points

        Input: 
            vec: numpy array, containing the initial state
            dj: numpy array, the Chebychev coefficients
            O: function, the dynamical generator (e.g, a differential operator)
            tup: tuple, containing the dynamical generator additional parameters
            meas_tup: list, containing tuples of space time points (time,[position idices]) 
        
        
        Returns: tuples containing:
            observed_data: dictionary, mapping space-time points (time,position) 
                            to the associated pressure field
        """
        vec = self.state
        self.total_time = self.dt*self.Nt
        d_j = self.cheby_coeff()
        meas_inds = [int(self.points[i][0]/self.dt) for i in range(len(self.points))]
        fi = np.zeros((vec.shape[0],3),dtype = complex)
        for j in range(self.Nt):     # Running over the time steps
            fi[:,0] = vec
            # The normalized differential operator O
            fi[:,1] = self.generator(vec)              
            G1 = d_j[0]*fi[:,0]
            G2 = d_j[1]*fi[:,1]
            G_3 = G1+G2
            for i in range(1,self.Nmax-1):
                fi[:,2] = 2*self.generator(fi[:,1])-fi[:,0]
                fi[:,0]=fi[:,1]
                fi[:,1]=fi[:,2]
                G_3 = G_3+d_j[i+1]*fi[:,2]
            vec = G_3
            # storing the measurement results in measurements
            if (j+1) in meas_inds:
                time = (j+1)*self.dt
                # storing the measements at the detector positions at a certain time
                for pos in self.meas_positions:
                    pos_ind = np.where(self.grid == pos)
                    pressure = vec[:self.size]
                    self.observed_data[time,pos] = float(pressure[pos_ind][0])
        
        # updating the system state
        self.state = vec   
        return self.observed_data    
 

    
    def cheby_coeff(self):
        """
        Evaluates the Chebychev coefficients for the propagator exp(O*dt),
        where O is a linear differential operator with units of 1/time.
         
        Based on the paper: Propagation Methods for Quantum Molecular Dynamics - Ronnie Kosloff 1994

        Input:
            dt: float, the time interval of propagation.
            lam_min: float, expected minimum eigenvalue of O
            lam_max: float, expected max eigenvalue of O
         
        Note: lam_min and lam_max do not have to be exact. Although it is important 
              to keep the normalized range of eigenvalues within [-1,1].
         
        Retun: array, contains the Chebychev coefficients, which can be used to propagate the system.
         
        """
        # Factor to ajust the eigenvalues to be between [-1,1]
        R = self.dE*self.dt/2
        c_j = np.zeros(self.Nmax,dtype = complex)
        c_j[0] = besseli(0,R)           #Zero coefficient.
        # The vector a contans c_n coefficients from 1 to Nmax, further multipication by exp(lam_min+ *dt) is needed  
        for n in range(1,self.Nmax):
            c_j[n] = 2*besseli(n,R)
            if abs(c_j[n])<1e-17 and n>R: 
                break
            
        self.Nmax = n+1
        d_j = np.exp((self.lam_min+self.lam_max)*self.dt/2)*c_j #normalizing factor times the cooefficeints
        return d_j
    

    def generator(self,y):
        """
        Operation of the normalized wave equation dynamical generator on the 
        state O
            Input:
                y: array, the initial state
                
            Returns:
                Oy: array, mapped state, operated on by the normalized version
                    of the dynamical generator
        """
        u = y[:self.size]
        v = y[self.size:]
        du_dt = v
        dv_dt = self.speed_field**2 * deriv_n_gen(u,self.grid,2)
        df = np.concatenate([du_dt, dv_dt])
        # The normalized operator, in the appropriat form for the Chebychev propagator.
        Oy = (2/self.dE)*(df-self.lam_min*y)-y
        return Oy
    
    
    
class optModel(model):
    
    def __init__(self,size,total_time,points,observed_data):
        super().__init__(size,total_time)
        self.observed_data = observed_data
        self.points = points        
        

    

    def GD(self,eta,max_iter,error_thresh):
        """
        Gradient Descent. The function applies the GD algorithm
            employing the adjoint state method.
            It modifies the speed field (model) parameters iterative until conversion
            or the number of iterations surpasses max_inter.
        
        
        Input
            eta: float, the step size
            max_inter: int, the maximum number of iterations
            error_thresh: float, the error threshold of the optimization procedure
        """
        # While the costfunction is above the error threshold or the number of 
        # iteration is below max_iter update the model parameters, employing 
        # gradient descent.
        
        # COMPLETE CODE
        
        i = 0
        error = model.evaluate()
        
        while error > error_thresh and i <= max_inter:
            i += 1
            #t1 = time.time()
            self.update_speed_field(eta)
            #t2 = time.time()
            #time_interval = t2 - t1
            error = model.evaluate()
        
            
    
    def update_speed_field(self,eta):
        """Updates the speed field using the adjoint state method"""
        # COMPLETE CODE
        
        # Set up the gradiant arrays
        Delta_speed_field = self.grad(x,y)
        self.speed_field = self.speed_field - eta*Delta_speed_field
        #self.speed_field = [w - (eta/m)*Delta_w
         #               for w, Delta_w in zip(self.speed_field,dw)]
        
                    
    def grad(self):
        """
        Propagates the equation of motion to find the pressure field, and then 
            evaluates the gradiant of the cost function with respect to the speed
            field, employing the adjoint state method.
        """
        # COMPLETE CODE
        
        # Feedforward
        
        #CALCULATE u
        
        
        # Gradiant of the cost function w.r.t the speed field 
        #return Delta_dm

    def L2_cost_function(self,r_func):        
         ''' Evaluates the L2 cost function, given the residual function (array).
           
             Input
                 r_func: array, difference between the observed data (real model) and
                 optimized model at a defined space-time points.
             
             Return
                 cost: float, <r_func,r_func>/2
         '''
         return (np.dot(r_func.T,r_func)/2)[0][0]
     
        
    def evaluate(self):
         '''Evaluates error of the prediction relative to error.
         
             Input: 
                 true_model: array
             
             Returns:
                 relative_error: float, the average relative error
         '''
         # sample the optimized model in the appropriate data points.
         opt_data = self.experiment()
         r_func = []  # a list containing the residual function values, matching
                      # the order of points
                      
         for point in self.points:
             r_func.append(opt_data[point]-self.observed_data[point])            
         
         # converting the list to an array and evaluating the L2 cost function
         return self.L2_cost_function(cv(r_func))
     
        
###############################################################################
################################# AUXILARY FUNCTIONS ##########################
###############################################################################

def gaussian(x,mu,sig):
    '''Returns a gaussian with mean mu and standard deviation sig'''
    return (1/(np.sqrt(2*np.pi*sig**2))) * np.exp(-np.power(x-mu,2)/(2*sig**2))


def gaussian_dot(x,mu,sig,c):
    '''Returns a gaussian with mean mu and standard deviation sig'''
    return (c*(x-mu)/sig**2)*gaussian(x,mu,sig)

def besseli(v,z):
    '''
    Modified Bessel function of the first kind of real order.
    Input:
        v: float real, order of the function
        z: float or complex, argument   
    '''
    return scipy.special.iv(v,z)

    
def Plot(x,y,x_axis_label = None,y_axis_label = None,label = None):
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.plot(x,y,label = label)
    plt.legend()
    plt.show()

def cv(value_list):
    '''Takes a list of numbers and returns a column vector:  n x 1'''
    return rv(value_list).T

def rv(value_list):
    '''Takes a list of numbers and returns a row vector: 1 x n'''    
    return np.array([value_list])




###############################################################################
################################# EVALUATION ##################################
###############################################################################


if __name__ == '__main__':
    
    
    ################################# REAL MODEL ##############################
    
    # setting up the real model, and sampling the solution
    #                          at the defined space-time points
    
    ## Initialization
    size = 2**8
    T0 = 60
    model = model(size,T0)
    base_speed = 0.01
    amp = 0.6
    speed_field = base_speed*(np.ones(model.grid.shape) + amp*gaussian(model.grid, model.L/2, model.L/10))
    sig = model.L/20
    amp = 0.1
    initial_pressure = gaussian(model.grid, 0, sig) + gaussian(model.grid, model.L, sig)
    initial_velocity = np.zeros(model.grid.shape)
    initial_state = np.concatenate((initial_pressure,initial_velocity))
    model.initialization(speed_field,initial_state)
    points = model.set_points()  # initializing the detection scheme
    # propagation and measurement
    observed_data = model.experiment()
    total_time = model.total_time
    final_pressure = model.state[:size]
    final_velocity = model.state[size:]
    
    print(f'observed_data: {observed_data}')
    
    ## Plotting the initial and final state
    Plot(model.grid,initial_pressure,x_axis_label = 'position',y_axis_label = 'pressure',label = 'initial state')
    Plot(model.grid,final_pressure,x_axis_label = 'position',y_axis_label = 'pressure',label = 'final state')

    ###########################################################################
    
      
    ################################# REAL MODEL ##############################
      
                                
    ## Setting up the optimization procedure
    optModel = optModel(size,total_time,points,observed_data)
    optModel.initialization(speed_field,initial_state)
    optModel.set_points(points)
    print(f'opt_data: {optModel.observed_data}')

    cost = optModel.evaluate()
    print(cost)


      
    

###############################################################################
################################# OLD CODE ####################################
###############################################################################
        
                
    
        
    


        
        
            


    

'''
class L2CostFunction(object):
    
 #   @staticmethod
  #  def value(pred_data,obs_data):
        Returns the value of the L2 cost function
            
            Input
                pred_data: array, the predicted data sampled from the pressure field
                                    which is a solution of the equations of motion, at 
                                    positions and time corresponding to the observed data.
                obs_data: array, observed (simulated) data
            
            Return
                value of the L2 cost function (float)
'''
        
   #     r = pred_data - obs_data;       # The residual function
    #    return (1/2)*np.dot(np.transpose(r),r) 
    
    #@staticmethod
    #def delta(a,y,z):
     #   return 
    
    
class GaussianSource(object):
    '''Returns a gaussian pulse, centered at a frequency omega_0, 
        standard diviation of sigma0 and amplitude amp.
        '''
    def __init__(self,omega0,sigma0,amp,t):
        '''Initializes a the source
            
            Input
                omega0: float, the central frequency of the Gaussian pulse
                sigma0: float, the standard deviation of the Gaussian pulse
                amp: float, the amplitude of the pulse
                t: float, the current time
        '''
        self.omega0 = omega0
        self.sigm0 = sigma0
        self.amp = amp
        self.t = t
        
        
    def getValue(self):    
        '''Returns the  pulse's value at time t'''
        # COMEPLET CODE
        pass
        #return pulse

class MonochromaticSource(object):
    '''Returns a monochromatic source at frequency omega_0 and a standard diviation of sigma0'''
        
    def __init__(self,omega0,amp,t):
        self.omega0 = omega0
        self.amp = amp
        self.t = t
        
        
    def getValue(self):    
        source = self.amp * np.exp(-1j * self.omega0 * self.t)
        return source
    
    







    
    
    



        
