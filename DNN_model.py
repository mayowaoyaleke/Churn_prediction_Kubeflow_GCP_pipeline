import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
def get_model():
    model = Sequential()
    model.add(Dense(12,input_dim =8, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(2, activation= "sigmoid"))

    model.compile(loss= "binary_crossentropy", optimizer="adam", metrics= ["accuracy"])

    model.fit(..., epochs = 150, batch_size=10)


