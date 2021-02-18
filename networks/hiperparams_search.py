from networks.prepare_real_data import RealPreparer
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class Searcher:

    params = []
    def __init__(self, x, y, ):
        pass

    def set_params(self,
                   layers=[],
                   optimizer='rmsprop',
                   loss='binary_crossentropy',
                   metric=['accuracy'],
                   epochs=200,
                   batch_size=5):
        # Установка параметров в качестве полей класса
        self.__layers = layers
        self.__optimizer = optimizer
        self.__loss = loss
        self.__metric = metric
        self.__epochs = epochs
        self.__batch_size = batch_size

        # Преобразования модели в классификатор sklearn
        self.model = KerasClassifier(build_fn=self.__build,
                                     epochs=epochs,
                                     batch_size=batch_size)

    # Функция для постройки модели
    def __build(self):
        model = Sequential(self.__layers)

        model.compile(optimizer=self.__optimizer,
                      loss=self.__loss,
                      metrics=self.__metric)

        return model

    # Адаптирование модели
    def fit(self, x, y):
        return self.model.fit(x=x, y=y)

    # Предсказание модели
    def predict(self, x):
        return self.model.predict(x)

    # Вычисление точности предсказания
    def score(self, x, y):
        return self.model.score(x, y)

