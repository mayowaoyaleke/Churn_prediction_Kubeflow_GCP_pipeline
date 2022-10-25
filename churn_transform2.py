from tkinter import NUMERIC, ON
import tensorflow as tf
import tensorflow_transform as tft
from traitlets import default

NUMERIC_FEATURE_KEYS = [
    'Age','Balance','CreditScore','CustomerId','EstimatedSalary','HasCrCard','IsActiveMember','NumOfProducts','RowNumber','Tenure'
] 

ONE_HOT_FEATURES   = {
    'Gender':2,'Geography':3
}

BUCKETIZE = {
    'Age' : 10
}

SCALE_FEATURES = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']

LABEL_KEY = 'Exited'

def transformed_name(key):
    return key + '_xf'

def fill_in_missing(x):
    default_value = '' if x.dtype == tf.string else 0
    if type(x) == tf.SparseTensor:
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value
        )
    return tf.squeeze(x, axis=1)

def convert_num_to_one_hot(label_tensor,num_label=2):
    one_hot_tensor = tf.one_hot(label_tensor, num_label)
    return tf.reshape(one_hot_tensor, [-1,num_label])

def preprocessing_fn(inputs):
    outputs = {}

    for  key in ONE_HOT_FEATURES.keys():
        dim = ONE_HOT_FEATURES[key]
        index = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), top_k = dim+1)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            index, num_label=dim +1
        )

    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)

    return outputs




    # for key in NUMERIC_FEATURE_KEYS:
    #     scaled = tft.scale_to_0_1(inputs[key])
    #     outputs[transformed_name(key)] = tf.reshape(scaled, [-1])

    # for key, vocab_size in VOCAB_FEATURE_DICT.items():
    #     indices = tft.compute_and_apply_vocabulary(inputs[key], NUM_OOV_BUCKETS)
    #     one_hot = tf.one_hot(indices, vocab_size + NUM_OOV_BUCKETS)
    #     outputs[transformed_name(key)] = tf.reshape(one_hot, [-1, vocab_size + NUM_OOV_BUCKETS])


    # outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)
    # return outputs 