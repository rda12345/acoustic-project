#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full waveform inversion. 
Overview: 
    The following program solves the acoustic inverse problem of a 1D system, 
    The medium dynamics are govened by the dissipationless acoustic equation.



Description: 
    
General Pseudo code for my use: 
    
    - Run a real model and get the true measurements
    - evaluate the error, the error function should propagate and update the predictions 
        then compare the them with the true measurements.
    Go in this loop until error < error_threshold
    - evaluate the gradiant: requires first solving the adjoint equation etc.
    - update the state
    - evaluate the error, the error function should propagate and update the predictions 
        then compare the them with the true measurements.


Description: 
    
General Pseudo code: 
    
    - Run a real model and get the true measurements
    - evaluate the error, the error function should propagate and update the predictions 
        then compare the them with the true measurements.
    Go in this loop until error < error_threshold
    - evaluate the gradiant: requires first solving the adjoint equation etc.
    - update the state
    - evaluate the error, the error function should propagate and update the predictions 
        then compare the them with the true measurements.



Full waveform inversion. 
Overview: 
    The following program solves the acoustic inverse problem of a 1D system, 
    The medium dynamics are govened by the dissipationless acoustic equation.



Description: 
    The program is structured in the following way. A "model" class incorporates
    The following procedures.
    1. Initialization procedure, setting up the details of the model,
    the initial velocity and pressure fields (vectors), and the 
    initial guess of the wave speed field (vector). 
    2. Propatation procedure, which solves the system's equation of motion, 
        i.e., the isotropic acoustic wave equation, to obtain the pressure
        field (vector) at space-time point (x,t).
    3. Gradient descent (GD) and update speed field procedures,
        which apply the adjoint state method to 
        to evaluate the gradient of the cost (objective) function, and 
        utilize the gradient to update the speed field.
    4. Evaluate procedure, which compares the optimized speed field to the real
        result.
        
    Additional classes are:
    1. realModel, which sets a predefined (the real)
        speed field, solves the equation of motion to obtain the "real" pressure field.
        The pressure field is then sampled to obtain the ultrasound measurements.
    2. CostFunction, which gives the value and the analytic part of the derivative
        for a given speed field.
    
Coding road map
3. code and check evalution relative to the initialized model and the real model
4. code the gradient descent without the using the adjoint state vector,
    just update the model, knowing the real model parameters.
5. code the adjoint state method
    subtasks:
        - solve for the adjoint state
        - evaluate the gradiant


General Pseudo code for my use: 
    
    - Run a real model and get the true measurements
    
    - evaluate the error, the error function should propagate and update the predictions 
        then compare the them with the true measurements.
    Go in this loop until error < error_threshold
    - evaluate the gradiant: requires first solving the adjoint equation etc.
    - update the state
    - evaluate the error, the error function should propagate and update the predictions 
        then compare the them with the true measurements.
        
        
NOTE: MAYBE TURN THE MEASUREMENTS TO A DICTIONARY SINCE THESE POINTS ARE GETTING ACESSED A LOT...
ANd maybe have an input points to experiment and output observed data


Tasks: 
    1. Finish summarizing the theory lyx file of the adjoint state method.
    2. Solve for the adjoint state.
    3. Update the gradient            
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from simulation.acoustic_model import AcousticModel
from simulation.chebyshev_propagator import ChebyshevPropagator
from simulation.detector import Detector



        
   
class OptModel(AcousticModel):
    
    def __init__(self,size: int, L: float = 1.0, observed_data = None, initial_state: np.ndarray = None):
        super().__init__(size, L)
        self.observed_data = observed_data
        self.initial_state = initial_state
        self.points = observed_data.keys()
        self.propagator = None
        self.detector = None
        base_speed = 0.1
        self.opt_speed_field = base_speed*np.ones(self.size)  # initial guess for the speed field
        

    

    def GD(self,eta,max_iter,error_thresh):
        """
        Gradient Descent. The function applies the GD algorithm
        employing the adjoint state method.
        It modifies the speed field (model) parameters iterative until conversion
        or the number of iterations surpasses max_inter.
        
        
        Parameters
        ----------
            eta: float, the step size
            max_inter: int, the maximum number of iterations
            error_thresh: float, the error threshold of the optimization procedure
        """
        # While the costfunction is above the error threshold or the number of 
        # iteration is below max_iter update the model parameters, employing 
        # gradient descent.
        
        # TODO
        i = 0
        optModel.run()
        error = optModel.evaluate()
        while error > error_thresh and i <= max_inter:
            i += 1
            #t1 = time.time()
            self.update_speed_field(eta)
            #t2 = time.time()
            #time_interval = t2 - t1
            error = model.evaluate()
        
            
    
    def update_speed_field(self,eta):
        """Updates the speed field using the adjoint state method"""
        # COMPLETE CODE
        
        # Set up the gradiant arrays
        Delta_speed_field = self.grad(x,y)
        self.speed_field = self.speed_field - eta*Delta_speed_field
        #self.speed_field = [w - (eta/m)*Delta_w
         #               for w, Delta_w in zip(self.speed_field,dw)]
        
                    
    def grad(self):
        """
        Propagates the equation of motion to find the pressure field, and then 
        evaluates the gradiant of the cost function with respect to the speed
        field, employing the adjoint state method.
        """
        # COMPLETE CODE
        
        # Feedforward
        
        #CALCULATE u
        
        
        # Gradiant of the cost function w.r.t the speed field 
        #return Delta_dm

    def L2_cost_function(self,r_func):        
         """
         Evaluates the L2 cost function, given the residual function (array).
           
         Parameters
         ----------
                 r_func: array, difference between the observed data (real model) and
                 optimized model at a defined space-time points.
             
        Return
        ------
                cost: float, <r_func,r_func>/2
        """
         return (np.dot(r_func.T,r_func)/2)[0][0]
     
    def get_times_positions(self):
        measure_times = sorted({t for (t,p) in self.observed_data.keys()})
        positions = sorted({p for (t,p) in self.observed_data.keys()})
        return measure_times, positions
    
    def run(self):
        """
        Evaluates the predicted measured pressures associated with opt_speed_field 

        Parameters
        ----------
        optModel: AcousticModel, the optimization model
        initial_state: np.ndarray (2*size,), the initial state 
            

        Returns
        -------
        opt_observed_data: dict, maps tuples of (time,position) to the predicted 
                                pressures of the optimization model
        """
        self.initialize(self.opt_speed_field,self.initial_state)
        self.detector = Detector(self)
        self.propagator = ChebyshevPropagator(self,self.detector,T0 = T0) 
        self.detector.setup_default(self.propagator)
        self.opt_data = self.detector.get_data(self.propagator)
    
    def get_data(self):
        return self.opt_data
        
    def evaluate(self):
         """
         Evaluates error of the prediction relative to error.
          
         Parameters
         ----------
                 true_model: array
             
         Returns
         -------
             relative_error: float, the average relative error
         """
         # sample the optimized model in the appropriate data points.
         r_func = []  # a list containing the residual function values, matching
                      # the order of points
                      
         for point in self.points:
             r_func.append(self.opt_data[point]-self.observed_data[point])            
         
         # converting the list to an array and evaluating the L2 cost function
         return self.L2_cost_function(cv(r_func))

        



    
    #opt_observed_data = detector.get_data(propagator)


###############################################################################
################################# EVALUATION ##################################
###############################################################################


if __name__ == '__main__':
    
    
 
    ## Setup real model
    # parameters
    size, L, dt = 2**8, 1.0, 0.1 
    T0, base_speed, amp_speed = 1, 0.01, 0.6
    sig, amp = 1/20, 1.0
    # setup model, detector and propagator
    model = AcousticModel(size = size,L = L)
    real_speed_field, initial_state = model.default_initial_state(amp,sig,base_speed,amp_speed)
    model.initialize(real_speed_field,initial_state)
    detector = Detector(model)
    propagator = ChebyshevPropagator(model, detector,T0 = T0) 
    detector.setup_default(propagator)
    observed_data = detector.get_data(propagator)
    #print(f'observed_data: {observed_data}')


    ## Setup optimization model
    optModel = OptModel(size, L, observed_data,initial_state)
    # Run a single theoretical experiment and evaluate the results
    optModel.run()
    error = optModel.evaluate()
    print(error)
    
    #opt_data = optModel.get_data()
    #print(f'opt_data: {opt_data}')
    
    

    #THERE IS NO APPERENT DIFFERENCE BETWEEN OPT_DATA AND OBSERVED_DATA, UNDERSTAND WHY
    
    
    
    







    
    
    



        
