import tensorflow as tf
import streamlit as st
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

train_loss = []
train_mae = []
train_rmse = []

# Create the chart outside the for loop
st.title('The chart for per epoch')
chart = st.line_chart(width=0, height=0, use_container_width=True)

# Modelni train qilish
for epoch in range(20):
    model.fit(x=x_train,y=y_train, validation_data=(x_test,y_test),epochs=epoch, batch_size = 32)
    train_metrics = model.evaluate(x_train, y_train)
    train_loss.append(train_metrics[0])
    train_mae.append(train_metrics[1])
    train_rmse.append(train_metrics[2])


    # Har bitta epochda grafikni yangilash
    chart_data = {"Training Loss": train_loss, "MAE":train_mae, "RMSE":train_rmse}
    chart.add_rows(chart_data)

