# -*- coding: utf-8 -*-
import numpy as np
import layer

class Network:
    
    def __init__(self, layers=None, target_activation = None, cost_func = None):
        self.LAYERS = layers
        self.target_activation = target_activation
        self. cost_func = cost_func
        for layer_from, layer_to in zip(self.LAYERS, self.LAYERS[1:]):
            layer_from.connect_to(layer_to)

        
    def feed_forward(self):

        for layer_from, layer_to in zip(self.LAYERS, self.LAYERS[1:]):
            layer_from.feed_to(layer_to)
            layer_to.activation()
    
    def compute_cost(self, target_arr):
        
        cost_arr = self.cost_func(self.LAYERS[-1], self.target_activation(target_arr))
        for neuron, c in zip(self.LAYERS[-1], cost_arr):
            neuron.cost = c

        
    def backprop(self):
        pass
        
                
        