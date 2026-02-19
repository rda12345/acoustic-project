#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isotropic acoustic wave equation, based on the analytics in wave_eq_solver.lyx 
The isotropic acoustic wave equation is expressed in terms of two first order
coupled partial differential equations and solved utilizing a Chebychev expansion
of the dynamical propagator.


Code review suggestions: 

    1. Consider @dataclass for small classes like Detector if it simplifies code.
    2. Add __repr__ for AcousticModel for easier debugging.
    3. Replace generator to an allocation-free generator_apply(state, out).
    4. Use pyfftw for fft caching plan
 
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy



class AcousticModel(object):
    """
    An acoustic model, contains a grid, speed field and state, and applies the 
    dynamical generator to evolve the state.
    """
    def __init__(self,size: int, L: float = 1.0):
        """
        Initialization of the acoustic model
        
        Parameters
        ----------
        size: int, the number of points in the 1D space grid
        L: float, length of the grid
        grid: array, the space grid of the proplem in the range [0,L]
                    with a total of size equally distributed points
        speed_field: np.ndarray (size,), the sound speed at every grid point
        state: np.ndarray (2*size,), system state (pressure, velocity)
        lam_min: float, minimum eigenvalue
        lam_max: float, maximum eigenvalue
        dE: float, spectral range
            
        """
        self.size = int(size)
        self.L = float(L)   # sets the aribitrary length scale of the problem. 
        self.grid = np.linspace(0.0,self.L,self.size) 
        
        # physical fields (populated by initalize)
        self.speed_field: np.ndarray = None
        self.state: np.ndarray = None
        
        # spaectal range used by Chebyshev (set in initialize)
        self.lam_min: float = None
        self.lam_max: float = None
        self.dE: float = None
        vec = np.array(list(range(0,size//2))+[0]+list(range(-size//2+1,0)))
        self.k = 2*np.pi/(max(self.grid)-min(self.grid))*vec  #[0:N/2-1, 0, -N/2+1:-1])
        
            
        

        
    
    def initialize(self,speed_field: np.ndarray,initial_state: np.ndarray) -> None:
        """
        Sets speed field and initial state. Precompute spectral range for propagator.
          
        Parameters
        ----------
        speed_field: ndarray (size,), wave speed at each grid point.
        initial_state: ndarray (2*size,), concatenation of pressure and velocity initial fields.            
        """
        #assert speed_field.shape == (self.size,)
        #assert initial_state.shape == (2*self.size,)
        
        self.speed_field = speed_field.astype(float).copy()   # the true model
        self.state =  initial_state.astype(complex).copy()
        dx =self.L/self.size  # space interval
        
        # maximum resolvable wavenumber for the grid (Nyquist)
        kmax = np.pi/dx          # maximum wavenumber
        
        # gemerator eigenvalues roughly scale like c*k; add small offset
        cmax = np.max(self.speed_field)
        self.lam_max = np.sqrt((cmax**2)*kmax**2+1.0) 
        self.lam_min = - self.lam_max
        self.dE = self.lam_max-self.lam_min  # range of the 'energy' spectrum, with units of 1/time
        
    def get_state(self):
        """Returns the field's state"""
        return self.state
    
    def set_state(self,vec):
        """Updates the field's state"""
        self.state = vec
        
        
    def generator(self, state: np.ndarray) -> np.ndarray:
        """
        Apply the normalized wave equation dynamical generator O on the 
        state vector.
        
        Parameters
        ----------
        state: ndarray, initial state (2*size,)
                
        Returns
        -------
        np.ndarray (2*size,), action of the normalized operator used by
        Chebychev propagator on the state.
        """
        #assert self.state is not None "Model not initialized"
        #assert self.speed_field is not None
        p = state[:self.size]
        v = state[self.size:]
        
        dp_dt = v
        dv_dt = (self.speed_field**2) * self.deriv_n_gen(p,self.grid,2)
        
        df = np.concatenate([dp_dt, dv_dt])   # Efficiency can be improved if df is first allocated and then 
                                                # inputed to the function, instead of reallocating each time
        # The normalized operator, in the appropriat form for the Chebychev propagator.
        return (2/self.dE)*(df-self.lam_min*state)-state
    
    
    def deriv_n(self,func,x,n):
        """
        Evaluates the n'th derivative of f applying fast fourier transform.
        f must vanish at the domain boundaries: f(min(x)) = f(max(x)) = 0
        
        Parameters
        ----------
        func: np.ndarray (N,), the function which derivative is evaluated
        x: np.ndarray (N,), the domain of the function (not used, kept for API consistency)
        n: int, the derivative order
        N: int, grid size
        
        Returns
        -------
        df, np.ndarray (N,)
        """
        # Note: x parameter is kept for API consistency but self.k is used instead
        dffft = ((1j*self.k)**n)*np.fft.fft(func) 
        df = np.fft.ifft(dffft)
        return df

    def deriv_n_gen(self,func,x,n):
        """
        Evaluates the n'th derivative of f applying fast foureir transform.
        The function generalizes deriv_n function by substracting a linear function
        from f in the begining and then correcting the derivative in the end.
        
        Evaluates the n'th derivative of f applying fast foureir transform.
        f must vanish at the domain boundaries: f(min(x)) = f(max(x)) = 0
        
        Parameters
        ----------        
        func: np.ndarray (N,), the function which derivative is evaluated
        x: np.ndarray (N,), the domain of the function
        n: int, the derivative order
            
        Returns
        -------
        df, np.ndarray (N,)
        """
        y1 = func[0]
        y2 = func[-1]
        slope = (y2-y1)/(x[-1]-x[0])
        lin_func = slope*(x-x[0])+y1
        func2 = func - lin_func
        df2 = self.deriv_n(func2,x,n)
        df = df2 +slope*np.ones(len(x));
        return df
    
    def default_initial_state(self, amp, sig, base_speed, amp_speed):
        initial_pressure = amp*(gaussian(self.grid, 0, sig) + gaussian(self.grid, self.L, sig))
        initial_velocity = np.zeros(self.grid.shape)
        speed_field = base_speed*(np.ones(self.grid.shape) + amp_speed*gaussian(self.grid, self.L/2,  self.L/10))
        initial_state = np.concatenate((initial_pressure,initial_velocity))
        return speed_field, initial_state
    
class Detector(object):

    
    def __init__(self,model: AcousticModel):    
        """
        Initialize the detector 
        
        Parameters
        ----------
        model: AcousticModel
        indices: list, containing the indices of the grid which will be measured
        """
        self.model = model 
        self.positions = []
        self.indices = []
        self.measure_times = []
        self.recording = False
        self.observed_data = {}
        
    
    def setup_default(self,propagator):
        """
        Default detector setup, sets detectors positions and measurement times
        
        Parameters
        ----------
        measure_times: list, measurement times
        positions: list, positions
        """
        dt = propagator.dt 
        Nt = propagator.Nt
        interval = max(1,Nt//5)
        time_inds = np.arange(interval,Nt+1,interval)
        self.measure_times = [dt*ind for ind in time_inds]       # measurement times        
        pos_idx = [self.model.size//4, self.model.size*3//4] 
        self.indices = pos_idx
        self.positions = [self.model.grid[i] for i in pos_idx]
    
    def setup_specific(self, measure_times: list, positions: list):
        """
        Setup specific detector positions and times
            
        Parameters
        ----------
        positions: list
        measure_times: list
        """
        self.positions = positions 
        self.measure_times = measure_times
        # Convert positions to indices
        self.indices = [np.argmin(np.abs(self.model.grid - pos)) for pos in positions] 
        
        
    def record(self):   
        """Signifies to the propagator to record measurements"""
        self.recording = True 
        
    #def get_data(self):
     #   return self.observed_data
    
    def get_data(self,propagator):
        # record measurements
        self.record()
        # propagate
        propagator.propagate()
        return self.observed_data


class ChebyshevPropagator:
    """
    Implements Chebyshev expansion of the propagator exp(O*dt).
    
    This class computes the Chebyshev coefficients once and performs time stepping
    by repeatedly applying the generator in the three-term recurrence.
    """

    def __init__(self,model: AcousticModel, detector: Detector, dt: float = 0.1, Nmax: int = 1000,T0: float = 10, cfl_check = False):
        """
        Parameters
        ----------
        model: object, the propagated model
        dt: float, the time interval of propagation. 
        Nmax: int, maximum number of coefficients
        Nt: int, number of time-steps
        total_time: float, total simulation duration
        d: np.ndarray, storring the Chebychev coefficients
        """
        self.model = model
        self.detector = detector
        self.dt = float(dt)
        self.Nmax = int(Nmax)
        self.Nt = int(T0//dt)
        self.total_time = self.Nt*dt
        self.d = None
        self._actual_Nmax = None  # Will be set by compute_cheby_coefficients
        # Note: CFL check requires model to be initialized (speed_field set)
        # It's safer to call _cfl_check() after model.initialize() if needed
        if cfl_check and model.speed_field is not None:
            self._cfl_check()

        


    def compute_cheby_coefficients(self) -> np.ndarray:
        """
        Evaluates the Chebychev coefficients for the propagator exp(O*dt),
        where O is a linear differential operator with units of 1/time.
        Uses modified Bessel functions as in Kosloff (1994) style formation.
         
        Note: lam_min and lam_max do not have to be exact. Although it is important 
              keep the normalized range of eigenvalues within [-1,1].
         
        Returns
        -------
        ndarray, contains the Chebychev coefficients, which can be used to
                propagate the system.      
        """
        lam_min = self.model.lam_min
        lam_max = self.model.lam_max
        dE = lam_max-lam_min
        assert lam_min is not None and dE is not None
        
        R = self.model.dE*self.dt/2  # Factor to adjust the eigenvalues to be between [-1,1]
        original_Nmax = self.Nmax  # Store original to avoid modifying instance variable
        c = np.zeros(original_Nmax,dtype = complex)
        c[0] = besseli(0,R)           #Zero coefficient.
        # generate until either Nmax or coefficients become negligible  
        n_final = original_Nmax - 1  # Default to full range
        for n in range(1,original_Nmax):
            c[n] = 2.0*besseli(n,R)
            if abs(c[n])<1e-17 and n>R:
                n_final = n
                break
        # Use only the significant coefficients
        actual_Nmax = n_final + 1
        c = c[:actual_Nmax]
        # multiply by global exponential factor which normalizes the coefficients
        self.d = np.exp((lam_min+lam_max)*self.dt/2.0)*c
        # Store actual number of coefficients for use in propagation_step
        self._actual_Nmax = actual_Nmax 

    def _cfl_check(self) -> None:
        """
        Performs a conservative CFL-like check and warn/raise if dt is large.

        This is heuristic: the scheme is spectral + Chebyshev and stability depends on
        spectral range; we check a common explicit-scheme guideline dx/c_max.
        Explaination: one would like to have omega*dt < 1, therefore, since 
        omega_max = c_max * k_max = c_max * dx /pi, we get dt_limit = dx / (cmax * pi)
        """
        dx = self.model.L / self.model.size
        cmax = np.max(self.model.speed_field)
        # conservative CFL limit for explicit finite-difference would be dt <= dx/(cmax*pi)
        # divide by pi to be conservative for spectral k values
        dt_limit = dx / (cmax * np.pi + 1e-16)
        if self.dt > dt_limit:
            # Do not silently continue: warn the user strongly but allow override
            raise ValueError(
                f"Time step dt={self.dt:.3e} exceeds conservative stability limit {dt_limit:.3e}. "
                "Reduce dt or refine spatial grid."
                )  
            
    def get_Nt(self):
        return self.Nt
        
    def propagate(self):
        """
        Evaluates exp(O*t)*vec, with t=Nt*dt, utilizing a Chebychev expansion of the dynamical propagator.
        Updates the system state.

        Input: 
        vec: numpy array, containing the initial state
        tup: tuple, containing the dynamical generator additional parameters
        
        Parameters
        ----------
        generator: function, the dynamical generator (e.g, a differential operator)
            
        Returns: 
        -------
        state: ndarray (2*size,), field's state at final time
        """
        vec = self.model.get_state()
        self.model.total_time = self.dt*self.Nt
        self.compute_cheby_coefficients()
        #x,max_x,min_x,lx,mass,k,lam_min,lam_max,dk = tup
        fi = np.zeros((vec.shape[0],3),dtype = complex)
        for j in range(self.Nt):     # Running over the time steps
            vec = self.propagation_step(vec,fi)
            if self.detector.recording:
                time = (j+1)*self.dt
                # Check if time matches any measurement time (with tolerance for floating point)
                if any(abs(time - mt) < 1e-10 for mt in self.detector.measure_times):
                    # storing the measements at the detector positions at a certain time
                    for pos_ind in self.detector.indices:
                        pos = self.model.grid[pos_ind]
                        pressure = vec[:self.model.size]
                        self.detector.observed_data[time,pos] = float(pressure[pos_ind])
        # updating the system state
        self.model.set_state(vec) 
        return vec
    
  
    
    def propagation_step(self,vec: np.ndarray, fi: np.ndarray) -> np.ndarray:
        """
        Performs a propagation step of size dt
           
        Parameters
        ----------
        vec: np.ndarray (model.size,), the current system state
        fi: np.ndarray (3,(2*model.size,), introduced to prevent reallocation of memory each propagation step
        
        Returns
        -------
        np.ndarray: propagated system state
        """
        
        fi[:,0] = vec
        # The normalized differential operator O
        fi[:,1] = self.model.generator(vec)              
        actual_Nmax = getattr(self, '_actual_Nmax', len(self.d))  # Use actual number of coefficients
        cheb_sum = self.d[0]*fi[:,0]+ self.d[1]*fi[:,1]
        for i in range(1,actual_Nmax-1):
            fi[:,2] = 2*self.model.generator(fi[:,1])-fi[:,0]
            fi[:,0], fi[:,1] =fi[:,1], fi[:,2]
            cheb_sum = cheb_sum + self.d[i+1]*fi[:,2]
        return cheb_sum.copy()
    

        
    
        
    
        
   

    
    


    
    

     
###############################################################################
################################# UTILITY FUNCTIONS ##########################
###############################################################################

def gaussian(x,mu,sig):
    """Returns a gaussian with mean mu and standard deviation sig"""
    return (1/(np.sqrt(2*np.pi*sig**2))) * np.exp(-np.power(x-mu,2)/(2*sig**2))


def gaussian_dot(x,mu,sig,c):
    """Returns a gaussian with mean mu and standard deviation sig"""
    return (c*(x-mu)/sig**2)*gaussian(x,mu,sig)

def besseli(v,z):
    """
    Modified Bessel function of the first kind of real order.
    Parameters
    ----------  
    """
    return scipy.special.iv(v,z)

    
def Plot(x,y,x_axis_label = None,y_axis_label = None,label = None):
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.plot(x,y,label = label)
    plt.legend()
    plt.show()
    
def cv(value_list):
    """Takes a list of numbers and returns a column vector:  n x 1"""
    return rv(value_list).T

def rv(value_list):
    """Takes a list of numbers and returns a row vector: 1 x n"""    
    return np.array([value_list])



###############################################################################
################################# EVALUATION ##################################
###############################################################################


if __name__ == '__main__':
    
    # parameters
    size = 2**8
    L = 1.0  # sets the length scale of the problem
    dt = 0.1
    T0 = 10 
    
    # create model 
    model = AcousticModel(size = size,L = L)
    
    # speed field with a Gaussian initial state
    base_speed = 0.01
    amp_speed = 0.6
    #  cycle_time = model.L/base_speed     # the time it takes the wave packet to
                                        # complete a complete circle around the grid
    
    # initial state
    sig = model.L/20
    amp = 1.0
    
    #initial_pressure = amp*(gaussian(model.grid, 0, sig) + gaussian(model.grid, model.L, sig))
    #initial_velocity = np.zeros(model.grid.shape)
    #initial_state = np.concatenate((initial_pressure,initial_velocity))
    speed_field, initial_state = model.default_initial_state(amp,sig,base_speed,amp_speed)
    

    # Initialization of an initial gaussian pressure distribution and no source
    model.initialize(speed_field,initial_state)
    
    # initialize detector
    detector = Detector(model)
    
    
    # initialize propagator
    propagator = ChebyshevPropagator(model, detector,T0 = T0)
    Nt = propagator.get_Nt()
    # setup detector 
    detector.setup_default(propagator)
    observed_data = detector.get_data(propagator)
    
    print(observed_data)
     
    ## Plotting the dynamics
    initial_pressure = initial_state[:size]
    final_state = model.get_state()
    final_pressure = final_state[:size]
    final_velocity = final_state[size:]
    
    Plot(model.grid,initial_pressure,x_axis_label = 'position',y_axis_label = 'pressure',label = 'initial state')
    Plot(model.grid,final_pressure,x_axis_label = 'position',y_axis_label = 'pressure',label = 'final state')
