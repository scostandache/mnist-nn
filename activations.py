# -*- coding: utf-8 -*-
import numpy as np
from neuron import Neuron

def sigmoid(z):
    return 1.0/(1+np.e**(-z))


def softmax(layer_arr):
    pass
    