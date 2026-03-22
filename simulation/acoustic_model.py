"""
Isotropic acoustic wave equation, based on the analytics in wave_eq_solver.lyx 
The isotropic acoustic wave equation is expressed in terms of two first order
coupled partial differential equations and solved utilizing a Chebychev expansion
of the dynamical propagator. 
"""
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
        self.grid = np.linspace(0.0, self.L, self.size, endpoint=False)     # final point is set so it doesn't coincide with the first point

        # physical fields (populated by initalize)
        self.speed_field: np.ndarray = None
        self.state: np.ndarray = None

        # spaectal range used by Chebyshev (set in initialize)
        self.lam_min: float = None
        self.lam_max: float = None
        self.dE: float = None
        vec = np.array(list(range(0,size//2))+[0]+list(range(-size//2+1,0)))
        self.k = 2*np.pi/self.L*vec  #[0:N/2-1, 0, -N/2+1:-1])
        
            
        

        
    
    def initialize(self, speed_field: np.ndarray, initial_state: np.ndarray) -> None:
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
    
    def set_state(self, vec: np.ndarray) -> None:
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
        dv_dt = (self.speed_field**2) * self.deriv_n_gen(p, 2)
        
        df = np.concatenate([dp_dt, dv_dt])   # Efficiency can be improved if df is first allocated and then 
                                                # inputed to the function, instead of reallocating each time
        # The normalized operator, in the appropriat form for the Chebychev propagator.
        return (2/self.dE)*(df-self.lam_min*state)-state
    
    
    def deriv_n(self, func: np.ndarray, n: int) -> np.ndarray:
        """
        Evaluates the n'th derivative of f applying fast fourier transform.
        f must vanish at the domain boundaries: f(min(x)) = f(max(x)) = 0
        
        Parameters
        ----------
        func: np.ndarray (N,), the function which derivative is evaluated
        n: int, the derivative order
        
        Returns
        -------
        np.ndarray (N,), the n'th derivative of func.
        """
        df_fft = ((1j*self.k)**n)*np.fft.fft(func) 
        return np.fft.ifft(df_fft)
         

    def deriv_n_gen(self,func: np.ndarray, n: int) -> np.ndarray:
        """
        Evaluates the n'th derivative of f applying fast foureir transform.
        The function generalizes deriv_n function by substracting a linear function
        from f in the begining and then correcting the derivative in the end.
        
        Evaluates the n'th derivative of f applying fast foureir transform.
        f must vanish at the domain boundaries: f(min(x)) = f(max(x)) = 0
        
        Parameters
        ----------        
        func: np.ndarray (N,), the function which derivative is evaluated
        n: int, the derivative order
            
        Returns
        -------
        np.ndarray (N,)
        """
        y1 = func[0]
        y2 = func[-1]
        delta_x = self.grid[-1]-self.grid[0]
        if abs(delta_x) < 1e-13:
            slope = 0
        else:
            slope = (y2-y1)/delta_x
        linear_correction = slope if n == 1 else 0
        lin_func = slope*(self.grid-self.grid[0])+y1
        func2 = func - lin_func
        dfn = self.deriv_n(func2, n)
        return dfn + linear_correction * np.ones_like(self.grid)
        
    
    def gaussian_initial_state(
            self,
            amp: float,
            sig: float, 
            base_speed: float,
            amp_speed: float
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        A Gaussian initial pressure distribution moving to the right
        and a speed field with a Gaussian perturbation on top of a constant background.
        
        Parameters
        ----------
        amp: float, amplitude of the initial pressure
        sig: float, standard deviation of the initial pressure
        base_speed: float, base speed of the initial speed field
        amp_speed: float, amplitude of the added Gaussian perturbation to the speed field
        
        Returns
        -------
        speed_field: np.ndarray (size,), speed field
        initial_state: np.ndarray (2*size,), initial state
        """
        speed_field = base_speed*(np.ones(self.size) + amp_speed*gaussian(self.grid, mu=self.L/2, sig=self.L/10))
        initial_pressure = amp * gaussian(self.grid, mu=0.3*self.L, sig=sig)
        dp_dx = self.deriv_n_gen(initial_pressure, n=1).real
        initial_velocity = - speed_field * dp_dx
        initial_state = np.concatenate((initial_pressure, initial_velocity))
        return speed_field, initial_state
    

    def defult_initial_state(
            self, 
            base_speed: float,
            amp_speed: float,
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the default initial state for the acoustic model.
        Vanishing pressure and velocity fields, matching the conditions of the adjoint method.
        A speed field with a Gaussian perturbation on top of a constant background.
        
        Parameters
        ----------
        amp: float, amplitude of the initial pressure
        sig: float, standard deviation of the initial pressure
        base_speed: float, base speed of the initial speed field
        amp_speed: float, amplitude of the added Gaussian perturbation to the speed field
        
        Returns
        -------
        speed_field: np.ndarray (size,), speed field
        initial_state: np.ndarray (2*size,), initial state
        """
        speed_field = base_speed*(np.ones(self.size) + amp_speed*gaussian(self.grid, mu=self.L/2, sig=self.L/10))
        initial_pressure = np.zeros(self.size)
        initial_velocity = np.zeros(self.size)
        initial_state = np.concatenate((initial_pressure, initial_velocity))
        return speed_field, initial_state
    

    def dH_dm(self, speed_field: np.ndarray, pressure: np.ndarray, source_point: np.ndarray) -> np.ndarray:
        """ 
        Evaluates the derivative of the Hamiltonian with respect to the speed field, which is the matrix that maps the second time derivatives of the state to the gradiant of the cost function with respect to the speed field.
        
        Parameters
        ----------
        speed_field: np.ndarray (size,), the speed field of the acoustic model
        pressure: np.ndarray (size,), the pressure of the acoustic model, which is the first half of the state vector.
        source_point: np.ndarray (siee,) the value of the source at the evaluated time step                              
        
        Returns
        ------
        np.ndarray (size, size), the derivative of the Hamiltonian with respect to the speed field, a diagonal matrix
                                for the present model.
        """  
        return np.diag(-2 * (1/speed_field) * self.deriv_n_gen(pressure, 2) + source_point) # the second time derivative of the pressure field, scaled by -2/m^3, where m is the speed field.

 


    

        
    
        
    
        
   

    
    


    
    

     


