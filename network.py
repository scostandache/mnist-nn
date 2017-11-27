# -*- coding: utf-8 -*-
import numpy as np
import layer

class Network:
    
    def __init__(self, layers_arr=None, target_activation = None, cost_func = None):
        self.LAYERS = layers_arr
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
        
        for actual_layer, prev_layer in zip(self.LAYERS[-2::-1], self.LAYERS[-1::-1]):
            for neuron in actual_layer:
                neuron.cost = neuron.act*(1.0-neuron.act)*np.sum(neuron.WEIGHTS_OUT*prev_layer.get_neuron_costs())
                for weight in neuron.WEIGHTS_OUT:
                    weight.cost = weight.to_neuron.cost * neuron.act
    
    def connect_layers(self):
        for actual_layer, next_layer in zip(self.LAYERS, self.LAYERS[1:]):
            actual_layer.connect_to(next_layer)
    
    def train(self, x_train, y_train):
        for train_sample, train_target in zip(x_train, y_train):
            pass
            
                
        
        
        
                
        