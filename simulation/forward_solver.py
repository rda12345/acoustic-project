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
    def __init__(self, model: AcousticModel, T0: float, Nt: int, measurement_times: list=None, measurement_positions: list=None) -> None:
        """
        Initializes the forward solver with the acoustic model, time parameters, and detector configuration.
        Parameters
        ----------
        model: AcousticModel, the acoustic model to be propagated
        T0: float, total time for propagation
        Nt: int, number of time points for the forward solver
        measurement_times: list, times at which to measure the acoustic field, used to define the detector. If not given, defaults to measuring at all time points.
        measurement_positions: list, positions at which to measure the acoustic field, given in the same units as the grid. 
                    The positions are converted to indices internally, and can be modified by setup_specific method. If not given, defaults to measuring at the 3/4 point of the grid.
        """
        self.model = model
        self.T0 = T0
        self.Nt = Nt
        self.predicted_data = None
        self.state_history = None
        self.dt = T0 / (self.Nt-1)
        if measurement_times is None:
            measurement_times = [self.dt*idx for idx in range(self.Nt)] 
        if measurement_positions is None:
            measurement_positions = [self.model.grid[(self.model.size*3)//4]]  # default to measuring at the 3/4 point of the grid, can be modified by setup_specific method
        self.detector = Detector(self.model.grid, measurement_times=measurement_times, measurement_positions=measurement_positions)        # define detector
        self.propagator = ChebyshevPropagator(self.model, self.detector, T0=self.T0, Nt=self.Nt)   # define propagator


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
                         size elements correspond to the initial pressure time-derivative distribution. Initialized by defult to None and set
                         later to a zero array for no input.
        source: callable, given the time, t, and outputs and np.ndarray of size (p,) of the source term at time t.
                        set by default to None and set later to a zero array for no input.
        """
        if initial_state is None:
            initial_state = np.zeros(2*self.model.size)
        if source is None:              # if source is not given define it as a zero array of the appropriate dimension
            def extended_source(t):
                return np.zeros(2*self.model.size)
        else:
            def extended_source(x):
                return np.concatenate([np.zeros(self.model.size), source(x)])
            
        self.model.initialize(speed_field, initial_state)   # initialize acoustic model
        _, self.state_history = self.propagator.propagate_with_source(extended_source)    # propagate the model
        self.predicted_data = self.detector.get_data()    # extract the predicted data from the detector  

    

    def get_history(self) -> np.ndarray:
        """
        Returns the pressure field dynamics

        Returns
        -------
        np.ndarray (size, Nt), pressure at all time points
        """
        if self.state_history is None:
            raise ValueError("Did not evaluate the state history")
        return self.state_history[:self.model.size,:]
    
    

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

        Returns
        -------
        np.ndarray (2*size,)
        """
        if self.model.state is None:
            raise ValueError("The forward solver has not been run yet. Please call the run() method first.")
        return self.model.state
    
    def evaluate_dH_dc(self, pressure: np.ndarray) -> np.ndarray:
        """
        Returns dH/dm as a function of time.
        """
        return self.model.dH_dc(pressure)
