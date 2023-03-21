import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
)
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

model_load = tf.keras.models.load_model('model')
model_load.evaluate(x_test, y_test)

st.write("Enter values for your house")
data = [0 for i in range(13)]
data[0] = st.number_input('CRIM - per capita crime rate by town', step=0.1)

data[1] = st.number_input('ZN - proportion of residential land zoned for lots over 25,000 sq.ft.')

data[2] = st.number_input('INDUS - proportion of non-retail business acres per town.')

data[3] = st.radio("CS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)", (1, 0))

data[4] = st.number_input('NOX - nitric oxides concentration (parts per 10 million)')

data[5] = st.number_input('RM - average number of rooms per dwelling')

data[6] = st.number_input('AGE - proportion of owner-occupied units built prior to 1940')

data[7] = st.number_input('DIS - weighted distances to five Boston employment centres')

data[8] = st.number_input('RAD - index of accessibility to radial highways')

data[9] = st.number_input('TAX - full-value property-tax rate per $10,000')

data[10] = st.number_input('PTRATIO - pupil-teacher ratio by town')

data[11] = st.number_input('B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town')

data[12] = st.number_input('LSTAT - % lower status of the population')

    
data = np.array(data)
data = data.reshape(1, 13)
data1 = (data - mean) / std

if st.button('Predict price'):
    st.write(f"Your house's price with is ${round(model_load(data)[0][0]*1000, 2)}")
#     st.write(f"Your house's price with is ${round(model_load(data1)[0]*1000, 2)}")
    st.write(f"Your house's price with is ${data}")
    st.write(f"Your house's price with is ${data1}")
