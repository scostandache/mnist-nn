# -*- coding: utf-8 -*-
import numpy as np
import layer

class Network:
    
    def __init__(self, layers_arr=None, target_activation = None, cost_func = None, lrate = None, lbd = None, friction = None ):
        
        self.LAYERS = layers_arr
        self.target_activation = target_activation
        self.cost_func = cost_func
        self.lrate = lrate
        self.lbd = lbd
        self.friction = friction
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
                neuron.cost = neuron.act*(1.0-neuron.act)*np.sum(neuron.get_out_weights_values()*prev_layer.get_neuron_costs())
                for weight in neuron.WEIGHTS_OUT:
                    print(type(weight))
                    weight.cost = weight.to_neuron.cost * neuron.act
    
    def update_weights(self, set_size):
        
        for l in self.LAYERS:
            for neuron in l:
                for weight in neuron.WEIGHTS_OUT:
                    weight.velocity = self.friction * weight.velocity - self.lrate*weight.cost
                    weight.val = (1.0 - self.lrate*self.lbd/set_size) * weight.val + weight.velocity
    

    def train(self, x_train, y_train, epochs):
        for _ in range(epochs):
            for train_sample, train_target in zip(x_train, y_train):
                self.LAYERS[0].load_data(train_sample)
                self.feed_forward()
                self.compute_cost(train_target)
                self.backprop()
                self.update_weights(x_train.size)
            
            
            
            
                
        
        
        
                
        