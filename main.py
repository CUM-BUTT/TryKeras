from model_builder import ModelBuilder
from prepare_real_data import RealPreparer
import matplotlib.pyplot as plt

path = 'models/lstm_3_layers'
data_builder, model_builder = RealPreparer(), ModelBuilder()
x, y = data_builder.Run()
model = model_builder.Build(x, y, test_part=0.1)
model_builder.Save(path)
model = model_builder.Load(path)

predict = model.predict(x)

plt.plot(data_builder.target)
plt.plot(predict)
plt.show()


