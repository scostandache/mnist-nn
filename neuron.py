import numpy as np

# -*- coding: utf-8 -*-
class Weight(object):
    
    def __init__(self, value):
        self.val = value
        self.cost = 0.0
        self.momentum = 0.0
    
    
class Neuron(object):
    
    def __init__(self, n_in):
        self.WEIGHTS = np.array([Weight(r) for r in np.random.normal(0, 1.0/np.sqrt(n_in), n_in)])
        self.val = 0.0
    
        

    
        
s = np.random.normal(0, 1.0/10,100)
my_neuron = Neuron(1000)
print(my_neuron.WEIGHTS[0].val)