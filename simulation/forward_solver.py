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


    def run(self, speed_field, initial_state) -> dict:
        """
        Propagates the initial state of the acoustic model using the Chebyshev propagator,
        and computes the predicted measurements at the defined space-time points using the detector.
        
        Parameters
        ----------
        speed_field: np.ndarray, The speed field of the acoustic model, which defines the spatially varying speed of sound.
        initial_state: np.ndarray, The initial state of the acoustic model. It should be a 1D array of length 2*size,
                         where the first size elements correspond to the initial pressure distribution, and the next 
                         size elements correspond to the initial velocity distribution.
        """
        self.model.initialize(speed_field, initial_state)
    
        # initialize detector
        self.detector = Detector(self.model)
        
        # initialize propagator
        self.propagator = ChebyshevPropagator(self.model, self.detector, T0=self.T0)
        Nt, dt = self.propagator.get_Nt(), self.propagator.get_dt()
        
        # setup detector 
        self.detector.setup_default({"dt": dt, "Nt": Nt})
        self.propagator.propagate()
        self.predicted_data = self.detector.get_data()
    
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