import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


def dnn(num_fea):
    model = Sequential()
    model.add(Dense(10, input_dim=num_fea, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='sigmoid', name='output'))
    model.summary()
    sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# load dataset
dataset = pd.read_csv('titanic.csv')
# train test split
train = dataset.iloc[:800]
test = dataset.iloc[800:]
train = train.values
test = test.values
num_fea = len(train[0])-1
train_x = train[:, 1:]
train_y = train[:, 0]

model = dnn(num_fea)
model.fit(train_x, train_y,
          batch_size=32, epochs=100, shuffle=True)
