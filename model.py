from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Input, Reshape


NUM_NEURONS_FirstDenseLayer = 32
NUM_NEURONS_SecondDenseLayer = 16
day_look_back = 30 ## số ngày quan sát
number_of_parameters = 5 ## số lượng tham số dùng để dự đoán
lstm_hidden_sizes = 40

model = Sequential()
model.add(Input(shape=(day_look_back, number_of_parameters)))


model.add(LSTM(lstm_hidden_sizes, return_sequences=True))
model.add(Reshape((-1, day_look_back * lstm_hidden_sizes)))

# model.add(LSTM(lstm_hidden_sizes))

model.add(Dense(NUM_NEURONS_FirstDenseLayer, activation='relu'))
model.add(Dense(NUM_NEURONS_SecondDenseLayer, activation='relu'))
model.add(Dense(1,activation ='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics='accuracy')

print(model);