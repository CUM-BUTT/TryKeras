from networks.model_builder import ModelBuilder
from networks.prepare_real_data import RealPreparer
import matplotlib.pyplot as plt

model_name = 'rnn_gru_lstm_dence_ru_ru'
path = 'models/' + model_name

data_builder, model_builder = RealPreparer(), ModelBuilder()
x, y = data_builder.Run()
model = model_builder.Build(x, y, test_part=0.1)
model_builder.Save(path)
model = model_builder.Load(path)

predict = model.predict(x)


predict = [predict[0][0]]*62 + [p[0] for p in predict]

plt.plot(data_builder.y_plot, c='r')
plt.plot(data_builder.x_plot)
plt.plot(predict, c='b')
plt.savefig(path + f'/{model_name}.png')
plt.show()


