# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:07:16 2019

@author: JamesChiou
"""
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from network import NeuralNetwork, NeuronLayer


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    ce = -(targets*np.log(predictions))
    return ce


dataset = [((1, 1), [1]),
           ((0, 0), [1]),
           ((1, 0), [0]),
           ((0, 1), [0])]

Net = NeuralNetwork()
Net.add_layer(NeuronLayer(input_num=2, output_num=5, activation='tanh'))
Net.add_layer(NeuronLayer(input_num=5, output_num=5, activation='tanh'))
Net.add_layer(NeuronLayer(input_num=5, output_num=5, activation='tanh'))
Net.add_layer(NeuronLayer(input_num=5, output_num=1, activation='sigmoid'))

epoch_num = 3000
batch_size = 4
learning_rate = 0.01
track = []
track_ce = []

# train
best_Net = None
best_ce_loss = 1000000
for i in range(epoch_num):
    mse_loss = 0
    ce_loss = 0
    shuffle_dataset = random.sample(dataset, len(dataset))
    for (i, data) in enumerate(shuffle_dataset):
        inputs = np.array(data[0])
        outputs = np.array(data[1])
        net_outs = Net.forward(inputs)
        net_outs = np.array(net_outs, ndmin=1)
        Net.backward(outputs)
        if i % batch_size == 0 or i == (len(shuffle_dataset)-1):
            Net.update_weights(learning_rate)

        # compute error
        for o, no in zip(outputs, net_outs):
            # ce_loss += cross_entropy(no, o)
            ce_loss += (o - no)**2
    ce_loss /= len(dataset)
    track_ce.append(ce_loss)
    if ce_loss < best_ce_loss:
        best_ce_loss = ce_loss
        best_Net = copy.deepcopy(Net)

# plt.plot(track)
plt.plot(track_ce)
