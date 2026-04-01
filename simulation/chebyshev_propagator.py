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
            Nt: int = 2**8, 
            Nmax: int = 1000, 
            T0: float = 10, 
            cfl_check = False
            ):
        """
        Parameters
        ----------
        model: object, the propagated model
        Nmax: int, maximum number of coefficients
        Nt: int, number of time points including zero
        total_time: float, total simulation duration
        d: np.ndarray, storring the Chebychev coefficients
        """
        self.model = model
        self.detector = detector
        self.Nmax = int(Nmax)
        self.Nt = Nt
        self.dt = T0/(self.Nt-1)


        self.cheby_coeff = None
        self._actual_Nmax = None  # Will be set by compute_cheby_coefficients
        # note: CFL check requires model to be initialized (speed_field set)
        # it's safer to call _cfl_check() after model.initialize() if needed
        if cfl_check and model.speed_field is not None:
            self._cfl_check()
        if T0 < self.dt:
            raise ValueError('Total run time should be more than the inteval time dt')

        


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
        if lam_min is None or lam_max is None:
            raise RuntimeError("Model must be initialized before computing Chebyshev coefficients") 
        R = self.model.dE*self.dt/2  # Factor to adjust the eigenvalues to be between [-1,1]
        original_Nmax = self.Nmax  # Store original to avoid modifying instance variable
        cheby_coeff = np.zeros(original_Nmax, dtype=complex)
        cheby_coeff[0] = besseli(0, R)           #Zero coefficient.
        # generate until either Nmax or coefficients become negligible  
        n_final = original_Nmax - 1  # Default to full range
        for n in range(1, original_Nmax):
            cheby_coeff[n] = 2.0 * besseli(n, R)
            if abs(cheby_coeff[n]) < 1e-17 and n>R:
                n_final = n
                break
        # Use only the significant coefficients
        actual_Nmax = n_final + 1
        cheby_coeff = cheby_coeff[:actual_Nmax]
        # multiply by global exponential factor which normalizes the coefficients
        self.cheby_coeff = np.exp((lam_min+lam_max) * self.dt/2.0) * cheby_coeff
        # store actual number of coefficients for use in propagation_step
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
        Propagation of an homogeneous linear differential equation (without a source term).
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
        if self.is_recording(time=0.0):
            self.record(time=0.0, vec=vec)
        self.compute_cheby_coefficients()
        for j in range(self.Nt-1):     # Running over the time steps
            vec = self.propagation_step(vec)
            if self.detector and self.detector.recording:
                time = (j+1)*self.dt
                # check if time matches any measurement time (with tolerance for floating point)
                if any(abs(time - mt) < 1e-10 for mt in self.detector.measure_times):
                    # storing the measements at the detector positions at a certain time
                    for pos_idx in self.detector.indices:
                        pos = self.model.grid[pos_idx]
                        pressure = vec[:self.model.size]
                        self.detector.observed_data[time,pos] = pressure[pos_idx]
        # updating the system state
        self.model.set_state(vec) 
        return vec
    
  
    
    def propagation_step(self, vec: np.ndarray) -> np.ndarray:
        """
        Performs a propagation step of size dt
           
        Parameters
        ----------
        vec: np.ndarray (model.size,), the current system state
        
        Returns
        -------
        np.ndarray: propagated system state
        """
        fi = np.zeros((vec.shape[0],3),dtype = complex)
        fi[:,0] = vec
        # The normalized differential operator O
        fi[:,1] = self.model.generator(vec)              
        actual_Nmax = getattr(self, '_actual_Nmax', len(self.cheby_coeff))  # Use actual number of coefficients
        cheb_sum = self.cheby_coeff[0]*fi[:,0]+ self.cheby_coeff[1]*fi[:,1]
        for i in range(1,actual_Nmax-1):
            fi[:,2] = 2*self.model.generator(fi[:,1])-fi[:,0]
            fi[:,0], fi[:,1] =fi[:,1], fi[:,2]
            cheb_sum = cheb_sum + self.cheby_coeff[i+1]*fi[:,2]
        return cheb_sum.copy()


        

    def propagate_with_source(
            self,
            source: callable
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates exp(O*t)*vec, with t=(Nt-1)*dt, utilizing a Chebychev expansion of the dynamical propagator.
        Evaluates the integral of the source term using the trapezoidal rule, while propagating the system state.
        Updates the system state.
        
        Parameters
        ----------
        source: callable, the source function
            
        Returns: 
        -------
        np.ndarray (2*size,), state at final time
        np.ndarray (2*size, Nt), state at all time points
        """
        vec_hom = self.model.get_state()        # homogeneous solution
        vec_private = np.zeros_like(source(0)).astype(complex)        # private solution
        # check if the dimensions of the homogeneous and private solutions are the same
        if vec_hom.shape[0] != vec_private.shape[0]:
            raise ValueError(f"Dimensions of homogeneous and private solutions must be the same.\
                                                            The dimensions of the homeneous and private solutions are {vec_hom.shape[0]}\
                                                            and {vec_private.shape[0]}, respectively.")            
        history = np.zeros((vec_hom.shape[0], self.Nt), dtype=complex)
        vec = vec_hom       # initial state
        if self.is_recording(time=0.0):             # record the data if requested
            self.record(time=0.0, vec=vec)
        history[:,0] = vec
        self.compute_cheby_coefficients()
        for j in range(self.Nt-1):     # running over the time steps, in total there are Nt times
            prev_time = j*self.dt
            vec = self.propagation_step(vec + (self.dt/2)*source(prev_time))
            vec = vec + (self.dt/2) * source(prev_time+self.dt)
            history[:,j+1] = vec
            time = (j+1)*self.dt
            if self.is_recording(time):     
                self.record(time, vec)

        self.model.set_state(vec) 
        return vec, history

    def is_recording(self, time: float) -> bool:
        """
        Check if to record the pressure at time 
        """
        if self.detector and self.detector.recording:
            # check if time matches any measurement time (with tolerance for floating point)
            if any(abs(time - mt) < 1e-10 for mt in self.detector.measure_times):
                return True
        return False
    
    def record(self, time: float, vec: np.ndarray) -> None: 
        """
        Stores the measements at the detector positions at a certain time
        """
        for pos_idx in self.detector.indices:
            pos = self.model.grid[pos_idx]
            pressure = vec[:self.model.size]
            self.detector.observed_data[time, pos] = pressure[pos_idx]
