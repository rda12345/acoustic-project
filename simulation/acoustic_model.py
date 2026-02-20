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
# TODO fix the code review suggestions

import numpy as np
from utilities.utility_functions import gaussian


class AcousticModel(object):
    """
    An acoustic model, contains a grid, speed field and state, and applies the 
    dynamical generator to evolve the state.
    """
    def __init__(self,size: int, L: float = 1.0) -> None:
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
        if speed_field.shape != (self.size,):
            raise ValueError( "Speed field must be of shape (size,)")
        if initial_state.shape != (2*self.size,):
            raise ValueError( "Initial state must be of shape (2*size,)")
        
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
        
    def get_state(self) -> np.ndarray:
        """Returns the field's state"""
        return self.state
    
    def set_state(self,vec: np.ndarray) -> None:
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
        if self.state is None:
            raise ValueError( "Model not initialized")
        p = state[:self.size]
        v = state[self.size:]
        
        dp_dt = v
        dv_dt = (self.speed_field**2) * self.deriv_n_gen(p,self.grid,2)
        
        df = np.concatenate([dp_dt, dv_dt])   # Efficiency can be improved if df is first allocated and then 
                                                # inputed to the function, instead of reallocating each time
        # The normalized operator, in the appropriat form for the Chebychev propagator.
        return (2/self.dE)*(df-self.lam_min*state)-state
    
    
    def deriv_n(self,func: np.ndarray,x: np.ndarray,n: int) -> np.ndarray:
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

    def deriv_n_gen(self,func: np.ndarray,x: np.ndarray,n: int) -> np.ndarray:
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
        delta_x = x[-1]-x[0]
        if abs(delta_x) < 1e-13:
            slope = 0
        else:
            slope = (y2-y1)/delta_x
        linear_correction = slope if n == 1 else 0
        lin_func = slope*(x-x[0])+y1
        func2 = func - lin_func
        dfn = self.deriv_n(func2,x,n)
        df = dfn + linear_correction * np.ones(len(x))
        return df
    
    def default_initial_state(self, amp: float,
                                sig: float, 
                                base_speed: float,
                                amp_speed: float
                                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the default initial state for the acoustic model.
        
        Parameters
        ----------
        amp: float, amplitude of the initial pressure
        sig: float, standard deviation of the initial pressure
        base_speed: float, base speed of the initial speed field
        amp_speed: float, amplitude of the initial speed field
        
        Returns
        -------
        speed_field: np.ndarray (size,), speed field
        initial_state: np.ndarray (2*size,), initial state
        """
        initial_pressure = amp*(gaussian(self.grid, 0, sig) + gaussian(self.grid, self.L, sig))
        initial_velocity = np.zeros(self.grid.shape)
        speed_field = base_speed*(np.ones(self.grid.shape) + amp_speed*gaussian(self.grid, self.L/2,  self.L/10))
        initial_state = np.concatenate((initial_pressure,initial_velocity))
        return speed_field, initial_state
    



    

        
    
        
    
        
   

    
    


    
    

     


