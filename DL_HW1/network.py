# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:46:43 2019

@author: JamesChiou
"""
import numpy as np


class NeuralNetwork(object):
    def __init__(self):
        self.neuron_layers = []

    def forward(self, inputs):
        o = inputs
        for l in self.neuron_layers:
            o = l.forward(o)
        return o

    def backward(self, outputs):
        layer_num = len(self.neuron_layers)
        pre_deltas = []
        while layer_num != 0:
            cur_layer = self.neuron_layers[layer_num - 1]
            if len(pre_deltas) == 0:  # output layer
                for n, o in zip(cur_layer.neurons, outputs):
                    error = n.output - o
                    n.backward(error)
            else:  # hidden layer
                pre_layer = self.neuron_layers[layer_num]
                for i, n in enumerate(cur_layer.neurons):
                    error = 0
                    for d, pn in zip(pre_deltas, pre_layer.neurons):
                        error += d * pn.weights[i]
                    n.backward(error)
            pre_deltas = cur_layer.backward()
            layer_num -= 1

    def update_weights(self, learning_rate):
        for l in (self.neuron_layers):
            l.update_weights(learning_rate)

    def add_layer(self, neuron_layer):
        self.neuron_layers.append(neuron_layer)


class NeuronLayer(object):
    def __init__(self, input_num, output_num, activation='sigmoid',
                 input_layer=False):
        self.neurons = []
        for i in range(output_num):
            n = Neuron(input_num, activation, input_layer)
            self.neurons.append(n)

    def forward(self, inputs):
        outputs = []
        for n in self.neurons:
            outputs.append(n.forward(inputs))
        outputs = np.array(outputs)
        outputs = np.squeeze(outputs)
        return outputs

    def backward(self):
        return [n.delta for n in self.neurons]

    def update_weights(self, learning_rate):
        for n in self.neurons:
            n.update_weights(learning_rate)


class Neuron(object):
    def __init__(self, input_num, activation='sigmoid', input_layer=False):
        self.weights = []
        self.output = 0
        self.delta = 0
        self.grad = np.zeros(input_num)
        self.inputs = []
        self.back_count = 0
        self.input_layer = input_layer
        # init weights
        for i in range(input_num):
            he_init = np.random.randn(1)*np.sqrt(2.0 / input_num)
            self.weights.append(he_init)
        self.weights = np.array(self.weights)
        self.weights = np.squeeze(self.weights)
        # if not input_layer then init bias
        if not self.input_layer:
            self.bias = np.random.randn(1)*np.sqrt(2.0 / input_num)
            self.grad_bias = np.zeros(1)
        self.act_function = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 0
        '''
        for (w, i) in zip(self.weights, inputs):
            self.output += w * i
        '''
        self.output = np.dot(inputs, self.weights)
        output = self.output if self.input_layer else self.output + self.bias

        self.output = self.activation(output)
        return self.output

    def activation(self, x):
        if self.act_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.act_function == 'tanh':
            return np.tanh(x)
        elif self.act_function == 'relu':
            return x * (x > 0)
        elif self.act_function == 'linear':
            return x

    def backward(self, error):
        self.back_count += 1
        if self.act_function == 'sigmoid':
            self.delta = error * self.output * (1 - self.output)
        elif self.act_function == 'tanh':
            self.delta = error * (1. - np.tanh(self.output)**2)
        elif self.act_function == 'relu':
            self.delta = error * (1 if self.output > 0 else 0)
        elif self.act_function == 'linear':
            self.delta = error * 1

        self.grad += self.delta * self.inputs
        if not self.input_layer:
            self.grad_bias += self.delta

    def update_weights(self, learning_rate):
        '''
        for (i, w) in enumerate(self.weights):
            new_w = w - learning_rate *\
                    self.delta / self.back_count * self.inputs[i]
            self.weights[i] = new_w
        '''
        self.weights = self.weights - learning_rate *\
            self.grad / self.back_count
        if not self.input_layer:
            self.bias = self.bias - learning_rate *\
                self.grad_bias / self.back_count
            self.grad_bias = np.zeros(1)
        self.grad = np.zeros(len(self.weights))
        self.back_count = 0
