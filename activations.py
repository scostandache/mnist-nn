# -*- coding: utf-8 -*-
import numpy as np

def dummy():
    pass

def sigmoid(self):
    for neuron in self:
        neuron.val = 1.0/(1+np.e**(-neuron.val))

def softmax(self):
    exp_sum = sum(np.e ** neuron.val for neuron in self)
    for neuron in self:
        neuron.val = (np.e ** neuron.val)/exp_sum
    
    
    