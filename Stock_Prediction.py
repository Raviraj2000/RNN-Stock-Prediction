import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('NSE-Tata-Global-Beverages-Limited.csv')

df = df.iloc[:, [0, 5]]

df['Date'] = pd.to_datetime(df.Date, format = '%Y-%m-%d')
df.index = df['Date']

data = df.sort_index(ascending = True, axis = 0)
data = data.drop(columns = ['Date'], axis = 1)

data = data.reset_index()

data = data.drop(columns = ['Date'], axis = 1)


data_train = data.iloc[:987, :].values
data_test = data.iloc[987:, :].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

data_train_scaled = sc.fit_transform(data_train)

X_train = []
y_train = []

for i in range(60, 987):
    X_train.append(data_train_scaled[i-60:i, 0])
    y_train.append(data_train_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

regressor = Sequential()


regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(LSTM(units = 50))


regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.summary()

history = regressor.fit(X_train, y_train, epochs = 10, batch_size = 25)


inputs = data[len(data) - len(data_test) - 60:].values
inputs = inputs.reshape(-1, 1)

inputs = sc.transform(inputs)
X_test = []

for i in range(60, 308):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock = regressor.predict(X_test)
predicted_stock = sc.inverse_transform(predicted_stock)


plt.figure(figsize=(20, 8))
plt.plot(data_test, label = 'Real Stock Price')
plt.plot(predicted_stock, label = 'Predicted Stock Price')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.show()




