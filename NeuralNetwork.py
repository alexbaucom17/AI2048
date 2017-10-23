import math
import numpy as np

def sigmoid_fn(x):
    return 1.0 / (1.0 + math.exp(-x))

def tanh_fn(x):
    return 2*sigmoid_fn(2*x) - 1

class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.fwd_sum = 0

    def forward(self, inputs):
        self.fwd_sum = np.sum(self.weights * inputs) + self.bias
        return self.activation_fn()

    def activation_fn(self):
        #return sigmoid_fn(self.fwd_sum)
        return tanh_fn(self.fwd_sum)


class Layer:

    def __init__(self, weights, bias, type_input=False):
        # weights should me n_neurons x n_inputs and bias should be n_neurons vector
        if weights.shape[0] != bias.shape[0]:
            raise ValueError('Number of nodes must match length of bias vector')

        self.n_neurons = weights.shape[0]
        self.type_input = type_input
        if type_input:
            self.n_inputs = self.n_neurons
        else:
            self.n_inputs = weights.shape[1]
        self.neurons = [Neuron(weights[i,:],bias[i]) for i in range(self.n_neurons)]

    def evaluate(self, inputs):
        if inputs.shape[0] != self.n_inputs:
            raise ValueError('Input size must match layer size')
        if self.type_input:
            return np.array([self.neurons[i].forward(inputs[i]) for i in range(self.n_neurons)])
        else:
            return np.array([self.neurons[i].forward(inputs) for i in range(self.n_neurons)])

    def get_layer_size(self):
        return self.n_neurons
    def get_input_size(self):
        return self.n_inputs


class Network:

    def __init__(self, weights, bias):
        # weights should be [n_neurons1 x n_inputs1, n_neurons2 x n_inputs2, etc]
        # bias should be [N1, N2, N3, etc]
        self.n_layers = len(weights)
        L0 = [Layer(weights[0],bias[0],True)]
        L1 = [Layer(weights[i],bias[i]) for i in range(1,self.n_layers)]
        self.layers = L0 + L1

    def evaluate(self, inputs):
        if inputs.shape[0] != self.layers[0].get_input_size():
            raise ValueError('Incorrectly sized input')
        intermediate_values = [inputs]
        for i in range(1,self.n_layers+1):
            intermediate_values.append(self.layers[i-1].evaluate(intermediate_values[i-1]))
        return intermediate_values[-1]

class NetworkStructure:

    def __init__(self,neurons_per_layer):
        # input is a list of the number of neurons per layer
        self.neurons_per_layer = neurons_per_layer
        self.n_layers = len(self.neurons_per_layer)
        self.n_outputs = self.neurons_per_layer[-1]
        self.n_inputs = self.neurons_per_layer[0]

    def get_number_of_weights_per_layer(self, n):
        if n == 0:
            return 2 * self.n_inputs # 1 weight and 1 bias term for inputs
        else:
            return self.neurons_per_layer[n]* (self.neurons_per_layer[n-1] + 1) #number of outputs from previous layer + a bias term

    def get_number_of_inputs_per_layer(self, n):
        if n == 0:
            return 1
        else:
            return self.neurons_per_layer[n-1]

    def get_number_of_neurons_per_layer(self, n):
        return self.neurons_per_layer[n]

    def get_number_of_outputs(self):
        return self.n_outputs

    def get_number_of_layers(self):
        return self.n_layers

    def get_number_of_inputs(self):
        return self.n_inputs


if __name__ == "__main__":
    X1 = np.array([[1],
                  [4],
                  [2],
                  [-4]])
    X2 = np.array([[1, 2, 3,-3],
                   [5,-20,12,0.26]])
    b1 = np.array([1,1,1,1])
    b2 = np.array([1,1])
    N = Network([X1,X2],[b1,b2])
    I = np.array([1,0.3,-1,3])
    print(N.evaluate(I))