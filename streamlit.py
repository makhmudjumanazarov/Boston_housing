import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
)
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

model_load = tf.keras.models.load_model('/model/')
model_load.evaluate(x_test, y_test)

st.write("Enter values for your house")
data = [0 for i in range(13)]
data[0] = st.number_input('CRIM - per capita crime rate by town', step=0.1)

