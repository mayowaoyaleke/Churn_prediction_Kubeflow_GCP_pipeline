from tkinter import NUMERIC
import tensorflow as tf
import tensorflow_transform as tft

NUMERIC_FEATURE_KEYS = [
    'Age','Balance','CreditScore','CustomerId','EstimatedSalary','HasCrCard','IsActiveMember','NumOfProducts','RowNumber','Tenure'
]

VOCAB_FEATURE_DICT = {
    'Gender':2,'Geography':3,'Surname':1169
}

NUM_OOV_BUCKETS = 2

LABEL_KEY = 'Exited'

def transformed_name(key):
    key = key.replace('-','_')
    return key + '_xf'

def preprocessing_fn(inputs):
    outputs = {}

    for key in NUMERIC_FEATURE_KEYS:
        scaled = tft.scale_to_0_1(inputs[key])
        outputs[transformed_name(key)] = tf.reshape(scaled, [-1])

    for key, vocab_size in VOCAB_FEATURE_DICT.items():
        indices = tft.compute_and_apply_vocabulary(inputs[key], NUM_OOV_BUCKETS)
        one_hot = tf.one_hot(indices, vocab_size + NUM_OOV_BUCKETS)
        outputs[transformed_name(key)] = tf.reshape(one_hot, [-1, vocab_size + NUM_OOV_BUCKETS])


    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)
    return outputs 