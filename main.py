from neuron import Neuron
from layer import Layer
import numpy as np
import activations

my_neuron = Neuron(2)
IL = Layer(size = 784)
HL = Layer(size = 100)
OL = Layer(size = 10)

for i in range(IL.size):
    IL[i] = Neuron(3.45)

IL.connect_to(HL)

print(HL[0].WEIGHTS_IN[0])
#for neuron in HL:
    #res = IL.dot(neuron.WEIGHTS_IN)
    #print(activations.sigmoid(res))

a=np.float64(4.5)
print( id(my_neuron))
my_neuron = my_neuron*my_neuron
print (id(my_neuron))