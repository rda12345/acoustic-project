"""
The file contains the ChebyshevPropagator class, which implements the Chebyshev expansion of the propagator exp(O*dt).
"""
import numpy as np
from .acoustic_model import AcousticModel
from .detector import Detector
from utilities.utility_functions import besseli


class ChebyshevPropagator:
    """
    Implements Chebyshev expansion of the propagator exp(O*dt).
    
    This class computes the Chebyshev coefficients once and performs time stepping
    by repeatedly applying the generator in the three-term recurrence.
    """

    def __init__(
            self, 
            model: AcousticModel, 
            detector: Detector = None, 
            dt: float = 0.1, 
            Nmax: int = 1000, 
            T0: float = 10, 
            cfl_check = False
            ):
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
        if T0 < dt:
            raise('Total run time should be more than the inteval time dt')

        


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

        This is a heuristic: the scheme is spectral + Chebyshev and stability depends on
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
    
    def get_dt(self):
        return self.dt
        
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
        self.compute_cheby_coefficients()
        fi = np.zeros((vec.shape[0],3),dtype = complex)
        for j in range(self.Nt):     # Running over the time steps
            vec = self.propagation_step(vec,fi)
            if self.detector and self.detector.recording:
                time = (j+1)*self.dt
                # Check if time matches any measurement time (with tolerance for floating point)
                if any(abs(time - mt) < 1e-10 for mt in self.detector.measure_times):
                    # storing the measements at the detector positions at a certain time
                    for pos_idx in self.detector.indices:
                        pos = self.model.grid[pos_idx]
                        pressure = vec[:self.model.size]
                        self.detector.observed_data[time,pos] = float(pressure[pos_idx])
        # updating the system state
        self.model.set_state(vec) 
        return vec
    
  
    
    def propagation_step(self, vec: np.ndarray, fi: np.ndarray) -> np.ndarray:
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



    def propagate_with_sources(
            self,
            source: callable
            ) -> np.ndarray:
        """
        Evaluates exp(O*t)*vec, with t=Nt*dt, utilizing a Chebychev expansion of the dynamical propagator.
        Evaluates the integral of the source term using Simpson's rule, while propagating the system state.
        Updates the system state.
        
        Parameters
        ----------
        source: callable, the source function
            
        Returns: 
        -------
        np.ndarray (2*size,), field's state at final time
        """
        vec = self.model.get_state()
        self.compute_cheby_coefficients()
        fi = np.zeros((vec.shape[0],3),dtype = complex)
        for j in range(self.Nt):     # running over the time steps
            vec = self.propagation_step(vec, fi)
        source_term = self.integrate_source_term(source) 
        vec += source_term
        self.model.set_state(vec) 
        return vec
        
    def integrate_source_term(
            self,
            source: np.ndarray
            ) -> np.ndarray:
        """
        Integrates the source term over time using Simpson's rule, while propagating the system state.
        """
        dimension = source(0).shape[0]
        fi = np.zeros((dimension,3),dtype = complex) 
        if self.Nt%2 != 0:
            raise ValueError("integrate_source_term: Nt must be even for Simpson's rule.")
        
        s = np.zeros(dimension, dtype=complex)
        t = self.Nt*self.dt
        for Ntau in range(self.Nt+1):
            tau = Ntau*self.dt
            vec = source(t-tau)
            for _ in range(Ntau):     # Running over the time steps
                vec = self.propagation_step(vec, fi) 
            if Ntau == 0 or Ntau == self.Nt:
                s += vec
            elif Ntau % 2 == 1:
                s += 4*vec
            else:
                s += 2*vec
        s *= self.dt/3
        return s
