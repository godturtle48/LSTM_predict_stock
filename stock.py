# import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Reshape
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

day_look_back = 5 ## số ngày quan sát

def parse_line(l):
    sl = l.strip().split(',')
    return sl[0], list(map(float, sl[3:7])), float(sl[6])


def visualize(predictVals, trueVals, i):
    plt.plot(predictVals)
    plt.plot(trueVals)
    # plt.show()
    plt.savefig('./image-train/Figure-{}'.format(i))
    plt.clf()

def loadData(fileName):
    with open(fileName, 'r') as file:
        file.readline()
        x, y = [], []

        sc = MinMaxScaler(feature_range = (0, 1))
        temp = []
        for i in range(day_look_back):
            line = file.readline()
            temp.append(parse_line(line))
        for line in file:
            temp.append(parse_line(line))

            if temp[0][0] == temp[day_look_back][0]:
                x.append(sc.fit_transform([v[1] for v in temp[day_look_back:0:-1]]))
                y.append(temp[0][2])

            temp.pop(0)

        x = np.array(x)
        y = np.array(y)
        y = np.reshape(y, (-1, 1))  
        return x, y

# df = pd.read_csv('data.csv')
# plt.plot(df['<Close>'])
# plt.show()

NUM_NEURONS_FirstDenseLayer = 128
NUM_NEURONS_SecondDenseLayer = 32
lstm_hidden_sizes = 32
number_of_parameters = 4 ## số lượng tham số dùng để dự đoán
epochs = 20
batch_size = 4
fileNameTrain = 'data2.csv'
fileNameTest = 'data.csv'

model = Sequential()
model.add(Input(shape=(day_look_back, number_of_parameters)))
##Mô hình LSTM trả về chuỗi giá trị
# model.add(LSTM(lstm_hidden_sizes, return_sequences=True))
# model.add(Reshape((-1, day_look_back * number_of_parameters)))
opt = 'Adam'

##Mô hình LSTM trả về 1 giá trị
model.add(LSTM(lstm_hidden_sizes))

model.add(Dense(NUM_NEURONS_FirstDenseLayer, activation='relu'))
model.add(Dense(NUM_NEURONS_SecondDenseLayer, activation='relu'))
model.add(Dense(1))
opt = Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics='mean_squared_error')

x, y = loadData(fileNameTrain)
x_test, y_test =  loadData(fileNameTest)

for i in range(100):
    
    history = model.fit(x, y, epochs = epochs, batch_size = batch_size, shuffle= True)
    model.save('time_series_predict_model.h5')
    y_predict = model.predict(x_test)


    visualize(y_predict, y_test, i)    



