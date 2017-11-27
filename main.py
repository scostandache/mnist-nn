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

start = timeit.default_timer()

    
end = timeit.default_timer()

train_target_arrs = []
test_target_arrs = []

for target in train_set[1]:
    train_target_arrs.append(data_funcs.target_arr(10, target))

for target in test_set[1]:
    test_target_arrs.append(data_funcs.target_arr(10, target))

######################

IL = Layer(size = 784, dropout_prob = 0.5)
HL = Layer(size = 100, activation = activations.sigmoid, dropout_prob = 0.5)
OL = Layer(size = 10, activation = activations.softmax)

NN = Network(np.array([IL,HL,OL]), target_activation = data_funcs.softmax, cost_func = cost_funcs.cross_entropy, lrate = 0.05, lbd = 3, friction = 0.45)
NN.train(train_set[0], train_target_arrs, 10)

