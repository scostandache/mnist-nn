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
        self.gradient = 0.0
        self.rms_cache = 0.0
        self.rms = 0.0
        
    
class Neuron(np.float64):
    
    def __init__(self, value=None):
        self.val = value
        self.WEIGHTS_IN=np.array([])
        self.error = 0.0
        self.drop = 1
        
        
    def __mul__(self, other):
        res = self.val * other.val
        self.val = res
        
    def get_weights_values(self):
        return np.array([weight.val for weight in self.WEIGHTS_IN], dtype=np.float64)
  
   # def __init__(self, n_in):
        #self.IN_WEIGHTS = np.array([Weight(r) for r in np.random.normal(0, 1.0/np.sqrt(n_in), n_in)])
        
    


