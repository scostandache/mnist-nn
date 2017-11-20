import numpy as np

# -*- coding: utf-8 -*-
class Weight(np.float64):
    
    def __init__(self, value):
        self.val = value
        self.cost = 0.0
        self.momentum = 0.0
    
    
class Neuron(np.float64):
    
    
    #def __new__(cls, n_in)
    def __init__(self, n_in):
        self.WEIGHTS = np.array([Weight(r) for r in np.random.normal(0, 1.0/np.sqrt(n_in), n_in)])
        
    


