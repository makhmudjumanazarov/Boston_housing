import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
     

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
)
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


model = Sequential()

model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse', metrics = ['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])


model.fit(x=x_train,y=y_train, validation_data=(x_test,y_test),epochs=200, batch_size = 32)

loss, mae, rmse = model.evaluate(x_test, y_test)
print(loss)
print(mae)
print(rmse)