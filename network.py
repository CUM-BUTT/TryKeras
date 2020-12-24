import tensorflow as tf
from tensorflow.keras.layers import Dropout, SimpleRNN, Dense, LSTM
from tensorflow.keras import Input, Sequential, activations, optimizers, losses
import data_prepare
import numpy as np

data = data_prepare.Load()[0]
#print(data)
test = data['test']
train = data['train']

shape = train['x'][0].shape

model = Sequential([
    Input(shape=shape),
    LSTM(units=256,  return_sequences=True),
    LSTM(units=128),
    #SimpleRNN(units=128, activation=activations.tanh,),
    Dropout(rate=0.2),
    Dense(units=1, activation=activations.softmax,  use_bias=True),
])
#model.summary()

model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=0.0001))

history = model.fit(train['x'], train['y'], validation_data=(test['x'], test['y']),
    epochs=50, batch_size=32)

model.save('model')
