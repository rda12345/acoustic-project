"""
Detector class, which is used to record measurements from the acoustic model.
"""
import numpy as np
from .acoustic_model import AcousticModel


class Detector(object):

    
    def __init__(self, grid: np.ndarray, measurement_times: list, measurement_positions: list) -> None:    
        """
        Initialize the detector 
        
        Parameters
        ----------
        grid: np.ndarray, grid points of the acoustic model, used to define the positions of the detector
        measurement_times: list, times at which to measure the acoustic field
        measurement_positions: list, positions at which to measure the acoustic field, given in the same units as the grid. 
                    The positions are converted to indices internally, and can
                    be modified by setup_specific method. If not given, defaults to measuring at the 3/4 point of the grid.
        """
        self.grid = grid
        self.size = len(grid)
        if measurement_times:
            self.measurement_times = measurement_times
        else:
            self.measurement_times = [0.0]  # default to measuring at time 0, can be modified by setup_specific method
        
        if measurement_positions:
            self.measurement_positions = measurement_positions
            self.indices = [np.argmin(np.abs(self.grid - pos)) for pos in measurement_positions]  # convert positions to the closest corresponding indices
        else:
            self.indices = [(self.size*3)//4]     # default to measuring at the 3/4 point of the grid, can be modified by setup_specific method
            self.measurement_positions = [self.grid[i] for i in self.indices]
            
        self.recording = True
        self.observed_data = {} # dictionary mapping (time, position) -> pressure field(time, position)
        
        
        
        
        
    def get_time_indices(self, dt: float) -> None:
        """Build the integer-index set for O(1) time lookup in is_recording."""
        return set(round(t / dt) for t in self.measurement_times)

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
        if not self.observed_data:
            raise RuntimeError("Warning: No data recorded. Check if the detector is set up correctly and if the propagator is propagating the system.")
        return self.observed_data
    

