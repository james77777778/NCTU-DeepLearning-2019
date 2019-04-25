import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from network import NeuralNetwork, NeuronLayer


def binary_cross_entropy(predictions, targets):
    predictions = np.clip(predictions, 1e-12, 1. - 1e-12)
    if int(targets) == 1:
        ce = -np.log(predictions)
    else:
        ce = -np.log(1 - predictions)
    return ce


def one_hot_encode(targets, n_classes=2):
    targets = targets.reshape(-1)
    one_hot_targets = np.eye(n_classes)[int(targets)]
    return one_hot_targets


# load dataset
dataset = pd.read_csv('titanic.csv')
dataset['Fare'] = (dataset['Fare'] -
                   dataset['Fare'].mean())/dataset['Fare'].std()
dataset['Age'] = (dataset['Age'] -
                  dataset['Age'].mean())/dataset['Age'].std()
# train test split
train = dataset.iloc[:800]
test = dataset.iloc[800:]
train = train.values
test = test.values

'''
good params
6-8-10-1, tanh, 300, 50, 0.01
6-8-10-1, tanh, 500, 50, 0.005, seed=1000
6-8-10-1, tanh, 500, 50, 0.005, seed=6, norm='Fare'
6-8-10-1, tanh, 500, 50, 0.005, seed=6, norm='Fare', 'Age'
'''
# set random seed
np.random.seed(6)
# build model
Net = NeuralNetwork()
Net.add_layer(NeuronLayer(input_num=6, output_num=8, activation='tanh',
                          input_layer=True))
Net.add_layer(NeuronLayer(input_num=8, output_num=10, activation='tanh'))
Net.add_layer(NeuronLayer(input_num=10, output_num=1, activation='sigmoid'))

epoch_num = 3000
batch_size = 50
lr = 0.005
track_ce = []
train_acc = []
test_acc = []
track_valid = []

# train
best_Net = None
best_test_loss = 1e10
for i in range(0, epoch_num):
    mse_loss = 0
    ce_loss = 0
    train_error = 0
    test_error = 0
    shuffle_train = np.random.permutation(train)
    for (j, data) in enumerate(shuffle_train, start=1):
        inputs = data[1:]
        # outputs = one_hot_encode(data[0])
        outputs = [data[0]]
        net_outs = Net.forward(inputs)
        net_outs = np.array(net_outs, ndmin=1)
        if (net_outs[0] < 0.5) and int(outputs[0]) > 0.5:
            train_error += 1
        elif (net_outs[0] > 0.5) and int(outputs[0]) < 0.5:
            train_error += 1
        Net.backward(outputs)
        if j % batch_size == 0:
            Net.update_weights(lr)

        # compute error
        for o, no in zip(outputs, net_outs):
            ce_loss += binary_cross_entropy(no, o)

    ce_loss /= len(shuffle_train)
    track_ce.append(ce_loss)
    train_acc.append(train_error/len(shuffle_train))

    # test (validation)
    test_loss = 0
    for data in test:
        inputs = data[1:]
        # outputs = one_hot_encode(data[0])
        outputs = [data[0]]
        net_outs = Net.forward(inputs)
        net_outs = np.array(net_outs, ndmin=1)
        if (net_outs[0] < 0.5) and int(outputs[0]) > 0.5:
            test_error += 1
        elif (net_outs[0] > 0.5) and int(outputs[0]) < 0.5:
            test_error += 1
        # compute error
        for o, no in zip(outputs, net_outs):
            test_loss += binary_cross_entropy(no, o)

    test_loss /= len(test)
    track_valid.append(test_error)
    test_acc.append(test_error/len(test))
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_Net = copy.deepcopy(Net)

    sys.stdout.write('\r' + '{:3d}'.format(i) +
                     '\tloss={:.5f}'.format(ce_loss))

# display
print()
print(best_test_loss)
plt.figure(1, figsize=(20, 15))
plt.rcParams.update({'font.size': 20})
plt.subplot(221)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Avg Cross Entropy')
plt.plot(track_ce)
track_ce = np.array(track_ce)
n_max = track_ce.argmin()
plt.annotate('{:.3f}'.format(track_ce[n_max]), xy=(n_max, track_ce[n_max]))
plt.subplot(223)
plt.title('Training Error Rate')
plt.xlabel('Epochs')
plt.ylabel('Error Rate')
plt.plot(train_acc)
train_acc = np.array(train_acc)
n_max = train_acc.argmin()
plt.annotate('{:.3f}'.format(train_acc[n_max]),
             xy=(n_max, train_acc[n_max]))
plt.subplot(224)
plt.title('Testing Error Rate')
plt.xlabel('Epochs')
plt.ylabel('Error Rate')
plt.plot(test_acc)
test_acc = np.array(test_acc)
n_max = test_acc.argmin()
plt.annotate('{:.3f}'.format(test_acc[n_max]), xy=(n_max, test_acc[n_max]))
plt.tight_layout()
plt.savefig('all_features' + '.png')
plt.show()
