"""
Detector class, which is used to record measurements from the acoustic model.
"""
import numpy as np
from .acoustic_model import AcousticModel


class Detector(object):

    
    def __init__(self, model: AcousticModel) -> None:    
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
        self.recording = True
        self.observed_data = {} # dictionary mapping (time, position) -> pressure field(time, position)
        
    
    def setup_default(self, info: dict) -> None:
        """
        Default detector setup, sets detector position and measurement times
        
        Parameters
        ----------
        info: dict, containing the parameters of the problem, including:
            dt: float, time step size
            Nt: int, number of time steps
        measure_times: list, measurement times
        positions: list, positions
        """
        dt = info["dt"] 
        Nt = info["Nt"]
        self.measure_times = [dt*idx for idx in range(Nt)]       # measurement times        
        self.indices = [self.model.size*3//4]                    # detector position index 
        self.positions = [self.model.grid[i] for i in self.indices]
    
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
        self.indices = [np.argmin(np.abs(self.model.grid - pos)) for pos in positions]  # convert positions to indices

        
        
    def do_not_record(self) -> None:   
        """Signifies to the propagator to record measurements"""
        self.recording = False 
        
    
    def get_data(self) -> dict:
        """
        Returns the observed data in a dictionary format.

        Returns
        -------
        dict[int, int], a dictionary mapping tuples (time, position) 
                        to the pressure field at that time and position.
        """
        if not self.observed_data or self.recording == False:
            raise("Warning: No data recorded. Check if the detector is set up correctly and if the propagator is propagating the system.")
        return self.observed_data
    

