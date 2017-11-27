# -*- coding: utf-8 -*-
from neuron import Neuron, Weight
import numpy as np
import types
import activations

class Layer(np.ndarray):
        
    def __new__(cls, size, activation=activations.dummy, dtype=object, dropout_prob = None):
        obj = np.asarray([Neuron() for _ in range(size)], dtype=dtype).view(cls)
        obj.activation = types.MethodType(activation, obj)
        obj.dropout_prob = dropout_prob
        obj.dropout_arr = np.ones([obj.size])
        return obj
    
        
    def connect_to(self,connect_layer):
        for neuron in connect_layer:
           neuron.WEIGHTS_IN = np.array([Weight(r, to_neuron = neuron) for r in np.random.normal(0, 1.0/np.sqrt(self.size), self.size)], dtype=object)
           for (w,f_neuron) in zip(neuron.WEIGHTS_IN,self):
               w.from_neuron = f_neuron
               f_neuron.WEIGHTS_OUT = np.append(f_neuron.WEIGHTS_OUT,w)
    
    def get_neuron_values(self):
        return np.array([neuron.val for neuron in self])
    
    def get_neuron_acts(self):
        return np.array([neuron.act for neuron in self])
    
    def get_neuron_costs(self):
        return np.array([neuron.cost for neuron in self])
        
    def load_data(self, data_arr):
        for neuron, d in zip(self, data_arr):
            neuron.act = d
    
    def feed_to(self, next_layer):
        
        if(self.dropout_prob != None):
            self.dropout_arr = np.random.binomial(1, 1.0 - self.dropout_prob, size=(self.size,))/(1.0 - self.dropout_prob)
                
        for neuron in next_layer:
            neuron.val = np.sum(self.get_neuron_acts()*self.dropout_arr*neuron.get_weights_values())

        
    def dropout(self):
        if(self.dropout_prob != None):
            probs = np.random.binomial(1, 1.0 - self.dropout_prob, size=(self.size,))
            for neuron, p in zip(self, probs):
                neuron.drop = p/(1.0 - self.dropout_prob)

                
                