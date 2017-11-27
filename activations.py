# -*- coding: utf-8 -*-
import numpy as np

def dummy():
    pass

def sigmoid(self):
    for neuron in self:
        neuron.act = 1.0/(1+np.e**(-neuron.val))

def softmax(self):
    exp_sum = np.sum(np.e ** neuron.val for neuron in self)
    for neuron in self:
        neuron.act = (np.e ** neuron.val)/exp_sum
    
    
    