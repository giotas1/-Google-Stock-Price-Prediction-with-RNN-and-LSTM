# Step 1: Import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

training_data= pd.read_csv('training_set.csv')
training_data.head(10)
training_data.info()
training_set= training_data.iloc[:, 1:2].values
training_set.shape , training_data.shape

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
training_set_scaled= sc.fit_transform(training_set)
training_set_scaled

# Creating a data structure with 60 timesteps and 1 output
x_train=[]
y_train=[]
for i in range(60,1257):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# converting x_train and y_train into numpy array
x_train , y_train = np.array(x_train), np.array(y_train)

x_train
y_train
x_train.shape
x_train= x_train.reshape(1197, 60,1)
x_train.shape

# Step 3: Building LSTM

# define an object (inilitizing RNN)
model = tf.keras.models.Sequential()
# first LSTM layer
model.add(tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True, input_shape=(60,1)))
# dropout layer
model.add(tf.keras.layers.Dropout(0.2))

# second LSTM layer
model.add(tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True))
# dropout layer
model.add(tf.keras.layers.Dropout(0.2))

# third LSTM layer
model.add(tf.keras.layers.LSTM(units=80, activation='relu', return_sequences=True))
# dropout layer
model.add(tf.keras.layers.Dropout(0.2))

# fourth LSTM layer
model.add(tf.keras.layers.LSTM(units=120, activation='relu'))
# dropout layer
model.add(tf.keras.layers.Dropout(0.2))
# output layer
model.add(tf.keras.layers.Dense(units=1))
model.summary()

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Training the model
model.fit(x_train,y_train, epochs=100, batch_size=32)

# Step 5: Making Predictions
test_data = pd.read_csv('test.csv')
test_data.shape
test_data.info()

real_stock_price= test_data.iloc[:, 1:2].values
real_stock_price.shape

# Getting predicted stock prices of month Nov 2019

# concatination
dataset_total = pd.concat((training_data['Open'], test_data['Open']), axis = 0)

# stock prices of previous 60 days for each day of Nov 2019
inputs = dataset_total[len(dataset_total) - len(test_data) - 60:].values

# reshape (convert into numpy array)
inputs = inputs.reshape(-1,1)

# feature scaling
inputs = sc.transform(inputs)

# creating a test set
x_test = []
for i in range(60, 80):
  x_test.append(inputs[i-60:i, 0])

# convert in numpy array
x_test = np.array(x_test)

# convert in 3D (required to process)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# getting predicted stock prices
predicted_stock_price= model.predict(x_test)
predicted_stock_price= sc.inverse_transform(predicted_stock_price)

print(predicted_stock_price[5]), print(real_stock_price[5])

# Step 6: Visualization
# Visualising the results
plt.plot(real_stock_price, color='red', label='real google stock price')
plt.plot(predicted_stock_price, color='blue', label='predicted google stock price')
plt.title('google stock price prediction')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()