"""
Detector class, which is used to record measurements from the acoustic model.
"""
import numpy as np
from .acoustic_model import AcousticModel


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
        self.observed_data = {} # dictionary mapping (time, position) -> pressure field(time, position)
        
    
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
        interval = max(1,Nt//10)
        time_idxs = np.arange(0,Nt,interval)
        self.measure_times = [dt*idx for idx in time_idxs]       # measurement times        
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
        
    
    def get_data(self,propagator):
        """
        Returns the observed in a dictionary format

        Returns
        -------
        dict[int, int], a dictionary mapping tuples (time, position) 
                        to the pressure field at that time and position.
        """
        # record measurements
        self.record()
        # propagate
        propagator.propagate()
        return self.observed_data