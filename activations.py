# -*- coding: utf-8 -*-
import numpy as np
from neuron import Neuron

def sigmoid(layer_arr):
    for i in range(layer_arr.size):
        layer_arr[i] = (1.0/(1+ np.exp(-layer_arr[i])))


def softmax(layer_arr):
    pass
    