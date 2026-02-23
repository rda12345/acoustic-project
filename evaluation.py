"""
Evaluation of the acoustic model.
Runs a single experiment and plots the initial and final states.
"""
from simulation.acoustic_model import AcousticModel
from simulation.detector import Detector
from simulation.chebyshev_propagator import ChebyshevPropagator
from utilities.utility_functions import Plot

if __name__ == '__main__':
    
    # parameters
    size = 2**8
    L = 1.0  # sets the length scale of the problem
    dt = 0.1
    T0 = 20
    
    # create model 
    model = AcousticModel(size = size,L = L)
    
    # speed field with a Gaussian initial state
    base_speed = 0.01
    amp_speed = 0.6
    #  cycle_time = model.L/base_speed     # the time it takes the wave packet to
                                        # complete a complete circle around the grid
    
    # initial state
    sig = model.L/20
    amp = 1.0
    # creating the speed field and initial state
    speed_field, initial_state = model.default_initial_state(amp,sig,base_speed,amp_speed)
    

    # initialization of an initial gaussian pressure distribution and no source
    model.initialize(speed_field,initial_state)
    
    # initialize detector
    detector = Detector(model)
    
    
    # initialize propagator
    propagator = ChebyshevPropagator(model, detector,T0 = T0)
    Nt = propagator.get_Nt()
    # setup detector 
    detector.setup_default(propagator)
    observed_data = detector.get_data(propagator)
    
     
    ## Plotting the dynamics
    initial_pressure = initial_state[:size]
    final_state = model.get_state()
    final_pressure = final_state[:size]
    final_velocity = final_state[size:]
    print(f'observed_data = {observed_data}')
    Plot(model.grid,initial_pressure,x_axis_label = 'position',y_axis_label = 'pressure',label = 'initial state')
    Plot(model.grid,final_pressure,x_axis_label = 'position',y_axis_label = 'pressure',label = 'final state')
