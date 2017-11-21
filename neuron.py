# TODO: create __new__ constructor
import numpy as np

# -*- coding: utf-8 -*-
class Weight(np.float64):
    
    def __init__(self, value, from_neuron=None, to_neuron=None):
        self.val = value
        self.cost = 0.0
        self.momentum = 0.0
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        
    
class Neuron(np.float64):
    
    pass
    #def __new__(cls, n_in)
    #def __init__(self, n_in):
        #self.IN_WEIGHTS = np.array([Weight(r) for r in np.random.normal(0, 1.0/np.sqrt(n_in), n_in)])
        
    


