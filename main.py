from neuron import Neuron
from layer import Layer
from network import Network

import numpy as np
import activations
import timeit
import _pickle as cPickle, gzip
import data_funcs
import cost_funcs


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='iso-8859-1')
f.close()


my_neuron = Neuron(2)
IL = Layer(size = 784, dropout_prob = 0.5)
HL = Layer(size = 100, activation = activations.sigmoid, dropout_prob = 0.5)
OL = Layer(size = 10, activation = activations.softmax)

IL.connect_to(HL)
HL.connect_to(OL)


IL.load_data(train_set[0][0])

start = timeit.default_timer()

for _ in range(1):
    IL.feed_to(HL)
    HL.activation()
    HL.feed_to(OL)
    OL.activation()
    
end = timeit.default_timer()

print(end-start)

a=np.array([1,2,3,4,5,6,7])

for c, d in zip(a[1:], a[2:]):
    print (c,d)

######################
    
NN = Network(np.array([IL,HL,OL]), target_activation = data_funcs.softmax, cost_func = cost_funcs.cross_entropy)




