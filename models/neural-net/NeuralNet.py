import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from sklearn.preprocessing import LabelEncoder

def relu(x):
    return x.clip(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(1)

def forward(x, parameters):
    caches = []
    
    L = len(parameters) // 2

    a_prev = x

    for l in range(1, L):
        # Retrieve the parameters
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]

        z = np.dot(W, a_prev) + b
        linear_cache = (a_prev, W, b)
        
        a_prev = relu(z)

        activation_cache = a_prev
        cache = (linear_cache, activation_cache)
        caches.append(cache)


    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]

    z = np.dot(W, a_prev) + b
    linear_cache = (a_prev, W, b)
    
    a_prev = sigmoid(z)
    
    activation_cache = a_prev
    cache = (linear_cache, activation_cache)
    caches.append(cache)

    return a_prev, caches

def linear_backward(dZ, linear_cache):
    a_prev, W, b = linear_cache

    dW = np.dot(dZ, a_prev.T) / a_prev.shape[1]
    db = np.sum(dZ, axis=1, keepdims=True) / a_prev.shape[1]
    dA_prev = np.dot(W.T, dZ)

    assert dW.shape == W.shape
    assert dA_prev.shape == a_prev.shape

    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    a = activation_cache
    dZ = dA * np.ceil(a).clip(0, 1)
    print("dA: {}".format(dZ))
    print("a: {}".format(a))
    print("dZ: {}".format(dZ))
    return dZ

def sigmoid_backward(dA, activation_cache):
    a = activation_cache
    dZ = dA * a * (1 - a)
    print(dZ.shape)
    print(dA.shape)
    print(a.shape)
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    return linear_backward(dZ, linear_cache)

def backward(output, Y, caches):
    grads = {}
    L = len(caches)

    dAL = - Y / (output) - (1 - Y) / (output)
    print("dAL shape: {}".format(dAL.shape))
    
    current_cache = caches[L - 1]

    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, 'sigmoid')

    grads['dA' + str(L - 1)] = dA_prev
    grads['dW' + str(L)] = dW
    grads['db' + str(L)] = db

    for l in reversed((range(1, L))):

        current_cache = caches[l - 1]
        dA_prev, dW, db = linear_activation_backward(dA_prev, current_cache, 'relu')
        grads['dA' + str(l - 1)] = dA_prev
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db

    return grads

def get_loss(output, Y):
    return -np.mean((Y * np.log(output) + (1 - Y) * np.log(1 - output))) 
        

parameters = dict()

m = 2
layers = [3, 1]


for l in range(1, len(layers)):
    parameters['W' + str(l)] = np.random.randn(layers[l], layers[l-1]) * np.sqrt(2/layers[l-1])
    parameters['b' + str(l)] = np.random.randn(layers[l], 1) * np.sqrt(2/layers[l-1])


X = np.random.randn(layers[0], m)
output, caches = forward(X, parameters)
Y = np.random.randint(2, size=(1,m))

# sigmoid(np.dot(parameters['W2'], relu(np.dot(parameters['W1'], X) + parameters['b1'])) + parameters['b2'])

grads = backward(output, Y, caches)

def gradient_check(X, Y, parameters, weight, epsilon=1e-5):
    for i in range(parameters[weight].shape[0]):
        for j in range(parameters[weight].shape[1]):

            parameters[weight][i][j] += epsilon
            output_a, caches = forward(X, parameters)
            j_a = get_loss(output_a, Y)

            parameters[weight][i][j] -= 2*epsilon
            output_b, caches = forward(X, parameters)
            j_b = get_loss(output_b, Y)

            grad_check = (j_a - j_b) / (2 * epsilon)
            grad_observed = grads['d' + weight][i][j]

            print(grad_check, grad_observed)
            if grad_check or grad_observed:
                print(np.abs(grad_check - grad_observed) / max(grad_check, grad_observed))

gradient_check(X, Y, parameters, 'W1', 1e-5)

# epsilon = 1e-5
# grad_a = sigmoid_backward(1, sigmoid(np.array([1, 1.5])))
# grad_n = (sigmoid(np.array([1, 1.5]) + epsilon) - sigmoid(np.array([1, 1.5]) - epsilon)) / (2 * epsilon)

# z = np.array([3, -1])
# dA = np.array([1, 1])
# epsilon = 1e-5
# grad_a = relu_backward(dA, relu(z))
# grad_n = (relu(z + epsilon) - relu(z - epsilon)) / (2 * epsilon)