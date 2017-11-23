# -*- coding: utf-8 -*-
from neuron import Neuron, Weight
import numpy as np

class Layer(np.ndarray):
        
    def __new__(cls, size, activation=None, dtype=object):
        obj = np.asarray([Neuron() for _ in range(size)], dtype=dtype).view(cls)
        obj.activation = activation
        return obj
    
    def connect_to(self,connect_layer):
        for neuron in connect_layer:
           neuron.WEIGHTS_IN = np.array([Weight(r, to_neuron = neuron) for r in np.random.normal(0, 1.0/np.sqrt(self.size), self.size)], dtype=object)
           for (w,f_neuron) in zip(neuron.WEIGHTS_IN,self):
               w.from_neuron = f_neuron
    
    def get_values(self):
        return np.array([neuron.val for neuron in self])
    
            
    
           
