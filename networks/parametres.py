from tensorflow.python.keras.layers import LSTM, GRU
from tensorflow.python.layers.core import Dense, Dropout

params = {
    'clf__layers': [
        [
            LSTM(units=256, return_sequences=True),
            Dropout(0.2),

            LSTM(units=64, return_sequences=False),
            Dropout(0.2),

            Dense(units=32, use_bias=True),
            Dropout(0.2),

            Dense(16, use_bias=True),
            Dropout(0.2),
        ],
        [
            GRU(units=512, return_sequences=True),
            Dropout(0.2),

            LSTM(units=64, return_sequences=False),
            Dropout(0.2),

            Dense(units=32, use_bias=True),
            Dropout(0.2),

            Dense(16, use_bias=True),
            Dropout(0.2),
        ],
        [
            LSTM(units=256, return_sequences=True),
            Dropout(0.2),

            LSTM(units=64, return_sequences=False),
            Dropout(0.2),

            Dense(units=32, use_bias=True),
            Dropout(0.2),

            LSTM(units=16, return_sequences=False),
            Dropout(0.2),

            Dense(8, use_bias=True),
            Dropout(0.2),
        ],
                   ],
    'clf__optimizer': ['rmsprop'],
    'clf__loss': ['binary_crossentropy'],
    'clf__metric': [['accuracy']],
    'clf__epochs': [25, 50]
    }