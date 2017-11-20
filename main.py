from neuron import Neuron
from layer import Layer
import numpy as np

my_neuron = Neuron(10)
L1 = Layer([Neuron(0) for _ in range(10)])
L2 = Layer([Neuron(10) for _ in range(100)])

print(L2)
#mul = L1 * L2[0].WEIGHTS
