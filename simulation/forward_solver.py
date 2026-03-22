"""
Forward solver for the acoustic wave equation using Chebyshev expansion of the propagator.
"""
import numpy as np
from simulation.acoustic_model import AcousticModel
from simulation.detector import Detector
from simulation.chebyshev_propagator import ChebyshevPropagator



class ForwardSolver:
    """
    A forward solver for the acoustic wave equation, which uses the Chebyshev propagator to propagate the initial
    state and compute the predicted measurements at the defined space-time points. 
    """
    def __init__(self, model: AcousticModel, T0: float) -> None:
        self.model = model
        self.T0 = T0
        self.predicted_data = None
        self.state_history = None
        self.dt = None


    def run(
            self,
            speed_field: np.ndarray,
            initial_state: np.ndarray=None,
            source: callable = None,
            ) -> None:
        """
        Propagates the initial state of the acoustic model, with a source term using the Chebyshev propagator,
        and computes the predicted measurements at the defined space-time points using the detector.
        
        Parameters
        ----------
        speed_field: np.ndarray, The speed field of the acoustic model, which defines the spatially varying speed of sound.
        initial_state: np.ndarray, The initial state of the acoustic model. It should be a 1D array of length 2*size,
                         where the first size elements correspond to the initial pressure distribution, and the next 
                         size elements correspond to the initial velocity distribution. Initialized by defult to None and set
                         later to a zero array for no input.
        source: callable, given the time, t, and outputs and np.ndarray of size (p,) of the source term at time t.
                        set by default to None and set later to a zero array for no input.
        """
        if initial_state is None:
            initial_state = np.zeros(2*self.model.size)
        if source is None:              # if source is not given define it as a zero array of the appropriate dimension
            extended_source = lambda t: np.zeros(2*self.model.size)
        else:
            extended_source = lambda x: np.concatenate([np.zeros(self.model.size), source(x)])
        
        self.model.initialize(speed_field, initial_state)   # initialize acoustic model
        self.detector = Detector(self.model)        # define detector
        self.propagator = ChebyshevPropagator(self.model, self.detector, T0=self.T0)   # define propagator
        Nt, self.dt = self.propagator.get_Nt(), self.propagator.get_dt()     
        self.detector.setup_default({"dt": self.dt, "Nt": Nt})      # setup detector
        _, self.state_history = self.propagator.propagate_with_source(extended_source)    # propagate the model
        self.predicted_data = self.detector.get_data()    # extract the predicted data from the detector  

    

    def get_history(self) -> np.ndarray:
        """
        Returns the pressure field dynamics
        """
        assert self.state_history is not None, "Did not evaluate the state history"
        return self.state_history[:self.model.size,:]
    
    
    
    def get_dt(self) -> float:
        assert self.dt is not None, "Time-step not defined, first run the forward solver"
        return self.dt
            

    def get_predicted_data(self) -> dict:
        """
        Returns the predicted measurements at the defined space-time points.
        """
        if self.predicted_data is None:
            raise ValueError("The forward solver has not been run yet. Please call the run() method first.")
        return self.predicted_data
    
    def get_state(self) -> np.ndarray:
        """
        Returns the state of the acoustic model at the final time after propagation.
        """
        if self.model.state is None:
            raise ValueError("The forward solver has not been run yet. Please call the run() method first.")
        return self.model.state
    
    def evaluate_dH_dm(self, speed_field: np.ndarray, pressure: np.ndarray, source: callable) -> np.ndarray:
        """
        Returns dH/dm as a function of time.
        """
        return self.model.dH_dm(speed_field, pressure, source)
