import numpy as np


def target_arr(size, pos):
    t_arr = np.zeros(size)
    t_arr[pos] = 1.0
    return t_arr

def softmax(arr):
    exp_sum = np.sum(np.e ** var for var in arr)    
    for i in range(len(arr)):
        arr[i] = (np.e**arr[i])/exp_sum
    return arr