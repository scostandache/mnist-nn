# -*- coding: utf-8 -*-
from neuron import Neuron
import numpy as np

class Layer(np.ndarray):
        
    def __new__(cls, neuron_arr, dtype=object):
        obj = np.asarray(neuron_arr, dtype=dtype).view(cls)
        return obj
    
        
    
