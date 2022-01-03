# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
import copy

# Function to calculate Euclidean distance

def euclidean_distance(x1, x2):

    square  = np.square(x1-x2)
    sum_r = np.sum(square)
    result = np.sqrt(sum_r)
    return result

# Function to calculate Manhattan distance

def manhattan_distance(x1, x2):
    
    dist = 0
    for i in range(len(x1)):
        dist += abs(x1[i]-x2[i])
    return dist



# Function to calculate identity function

def identity(x, derivative = False):
    
    if derivative:

        
        return np.ones(np.shape(x))
    
    else:

        return x

# Function to calculate sigmoid functon

def sigmoid(x, derivative = False):
    
    if derivative:

        denom = 1 + np.exp(-x)
        sig = 1/(denom)
        derivative = sig*(1-sig)

        return derivative
    
    else:

        denom = 1 + np.exp(-x)
        sig = 1/(denom)

        return sig


# Function to calculate tanh function

def tanh(x, derivative = False):
    
    if derivative:
        
        tanh_value =np.tanh(x)
        return 1-(tanh_value**2)
    
    else:

        tanh_value =np.tanh(x)
        return tanh_value



# Function to calculate Relu funcion

def relu(x, derivative = False):
    
    if derivative:

        return 1. * (x > 0)
    
    else:

        return x * (x > 0)
         

def softmax(x, derivative = False):

    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    
    p = np.clip(p, 1e-10, 1 - 1e-10)
    a = np.shape(y)[0]
    b= -np.log(p)
    c = y*b
    d = np.sum(c)

    result = d/a
    
    return result


# Function to compute one hot encoding

def one_hot_encoding(y):
    
    binary_zero = [0]*(max(y)+1)

    dict_data = {}


    bin_list = []


    for k in enumerate(set(y)):

        def_bin = copy.deepcopy(binary_zero)
        def_bin[k[0]] = 1

        dict_data[k[1]] = def_bin
        
    for i in y:
    
        bin_list.append(dict_data[i])

    return np.array(bin_list)

