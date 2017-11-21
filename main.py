from neuron import Neuron
from layer import Layer
import numpy as np
import activations

my_neuron = Neuron(10)
IL = Layer(size = 784)
HL = Layer(size = 100)
OL = Layer(size = 10)


IL.connect_to(HL)
print(HL[0].WEIGHTS_IN[0].to_neuron)

