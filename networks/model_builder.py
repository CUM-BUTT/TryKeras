import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import tensorflow as tf

from tensorflow.keras.layers import Dropout, SimpleRNN, Dense, LSTM, GRU
from tensorflow.keras import Input, Sequential, activations, optimizers, losses

class ModelBuilder:

    def Build(self, x, y, test_part=0.1) -> Sequential:
        model = Sequential([
            #GRU(units=256, return_sequences=True, input_shape=(x[0].shape)),
            SimpleRNN(units=256, return_sequences=True, input_shape=(x[0].shape)),
            #LSTM(units=256, return_sequences=True, input_shape=(x[0].shape)),
            Dropout(0.2),

            #Dense(units=64, use_bias=True),
            LSTM(units=64, return_sequences=True),
            Dropout(0.2),

            GRU(units=32, use_bias=True, return_sequences=False),
            #LSTM(units=16, return_sequences=False),
            Dropout(0.2),

            Dense(len(y[0])*3, use_bias=True),
            Dropout(0.2),

            Dense(len(y[0])),
        ])

        total_len = len(x)
        test_len = int(total_len * test_part)
        train_len = total_len - test_len

        #model.summary()
        self.model = model
        model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=0.001))
        self.history = model.fit(x[:train_len], y[:train_len],
                            validation_data=(x[train_len:], y[train_len:]),
                            epochs=30, batch_size=32)

        return model

    def Load(self, path):
        self.model = tf.keras.models.load_model(path)
        return self.model

    def Save(self, path):
        self.model.save(path)

