import tensorflow as tf
from tensorflow.keras.models import load_model
import data_prepare

model = load_model('model')
data = data_prepare.Load()[0]
test = data['test']
train = data['train']
x = [test['x']]
y = test['y'][0]

print(model.predict(x), y)