from unicodedata import name
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

from churn_transform2 import ONE_HOT_FEATURES, SCALE_FEATURES
from trainer import transformed_name
def get_model():
    #One-hot Categorical Features
    input_features = []
    for key, dim in ONE_HOT_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape = (dim + 1,),
            name = transformed_name(key))
        )
    #Scale Features
    for key in SCALE_FEATURES():
        input_features.append(
            tf.keras.Input(shape = (dim + 1,),
            name = transformed_name(key))
        )

    inputs = input_features

    d = tf.keras.layers.concatenate(inputs)
    for _ in range(2):  
        d = tf.keras.layers.Dense(8, activation='relu')(d) 
        d = tf.keras.layers.Dense(64, activation='relu')(d)
        d = tf.keras.layers.Dense(16, activation='sigmoid')(d)
        outputs = tf.keras.layers.Dense(3, activation = 'sigmoid')(d)


        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs) 

        keras_model.compile(   
                   optimizer=tf.keras.optimizers.Adam(1e-2), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                             tf.keras.metrics.BinaryAccuracy(),
                             tf.keras.metrics.TruePositives()])
        keras_model.summary()
    return keras_model

     
    model = Sequential()
    model.add(Dense(12,input_dim =8, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(2, activation= "sigmoid"))

    model.compile(loss= "binary_crossentropy", optimizer="adam", metrics= ["accuracy"])

    model.fit(..., epochs = 150, batch_size=10)








def _build_keras_model() -> tf.keras.Model:
   
   d = keras.layers.concatenate(inputs) 
   for _ in range(2):  
    d = keras.layers.Dense(8, activation='relu')(d) 
    outputs = keras.layers.Dense(3)(d) 
           
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(   
                   optimizer=keras.optimizers.Adam(1e-2), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
                    metrics=[keras.metrics.SparseCategoricalAccuracy()])  
    model.summary(print_fn=logging.info)  
    return model

