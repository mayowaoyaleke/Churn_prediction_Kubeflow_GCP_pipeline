#Define imports
from pyexpat import model
# from kerastuner.engine import base_tuner
# import kerastuner as kt
from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs  
import tensorflow as tf
import tensorflow_transform as tft
import sklearn
from sklearn.ensemble import RandomForestClassifier
from tfx.components.trainer.fn_args_utils import DataAccessor
import os
import pickle
from typing import Tuple
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.dsl.io import fileio
from tfx.utils import io_utils
from tfx_bsl.tfxio import dataset_options
from tensorflow_metadata.proto.v0 import schema_pb2
import numpy as np
import absl
# import tensorflow_decision_forests as tfdf
import tensorflow_transform as tft
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

ONE_HOT_FEATURES2   = {
    'Gender':2,'Geography':3
}

ONE_HOT_FEATURES   = [
    'Gender','Geography'
]

BUCKETIZE = {
    'Age' : 10
}

NUMERIC_FEATURE_KEYS = [
    'Age','Balance','CreditScore','CustomerId','EstimatedSalary','HasCrCard','IsActiveMember','NumOfProducts','RowNumber','Tenure'
]

CATEGORICAL_FEATURE_KEYS = ['Geography','Gender']

SCALE_FEATURES = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']

LABEL_KEY = 'Exited'

# Renamimg Features   
def transformed_name(key):
    return key + '_xf'

#Define Callbacks
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 10)

#Load compressed data
def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type= 'GZIP')

#Load data
def _input_fn(file_pattern: str, tf_transform_output: tft.TFTransformOutput, num_epochs= None, batch_size: int = 200,) -> tf.data.Dataset:

    # Get post transform feature specification
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    #create batches of features and labels
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern= file_pattern,
        batch_size= batch_size,
        features= transformed_feature_spec,
        reader = _gzip_reader_fn,
        num_epochs= num_epochs,
        label_key= transformed_name(LABEL_KEY)
    )

    return dataset
  

#Build model
def get_model():
    #One-hot Categorical Features
    input_features = []
    for key in ONE_HOT_FEATURES:
        input_features.append(
            tf.keras.Input(shape = (1,),
            dtype = "string",
            name = transformed_name(key))
        )
    #Scale Features
    for key in SCALE_FEATURES:
        input_features.append(
            tf.keras.Input(shape = (1,),
            name = transformed_name(key))
        )
    
    inputs = input_features
    all_inputs = tf.keras.layers.concatenate(inputs)
    #reshaped_narrative = tf.reshape(inputs[0], [-1])

    #x = tf.keras.layers.Reshape((3, 4), input_shape = (12,))(all_inputs)

    # d = tf.keras.layers.concatenate(inputs)
      
    x = tf.keras.layers.Dense(8, activation='relu')(all_inputs) 
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='sigmoid')(x)

    outputs = tf.keras.layers.Dense(3, activation = 'sigmoid')(x)

    keras_model = tf.keras.Model(inputs= all_inputs, outputs=outputs) 

    keras_model.compile(   
                   optimizer=tf.keras.optimizers.Adam(1e-2), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                             tf.keras.metrics.BinaryAccuracy(),
                             tf.keras.metrics.TruePositives()])
    keras_model.summary()
    return keras_model
 

def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        
        feature_spec = tf_transform_output.raw_feature_spec()
        
        feature_spec.pop("Exited")
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        
        # get predictions using the transformed features
        return model(transformed_features)
        
    return serve_tf_examples_fn

#Run
def run_fn(fn_args: FnArgs) -> None:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir = fn_args.model_run_dir, update_freq = 'batch'
    )
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_binary_accuracy', mode = 'max', verbose = 1, patience = 10)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor = 'val_binary_accuracy', mode = 'max', verbose =1, save_best_only = True)

    #
    schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

    # Load tf_transform_output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = _input_fn(fn_args.train_files, tf_transform_output, 10)
    eval_set = _input_fn(fn_args.eval_files, tf_transform_output, 10)

    train_dataset = _input_fn(fn_args.train_files,tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files,tf_transform_output)

    # Build the model
    model = get_model()
    model.fit(##tf.expand_dims(train_set, axis= -1)##,
              train_set, 
              validation_steps = 32, 
              validation_data = eval_set)
    absl.logging.info(model)

    evaluation = model.evaluate(eval_set, steps = 32)
    absl.logging.info('Accuracy: %f', evaluation)
    
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, 
                                 tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples')) 
    }
    # model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
    model.save(fn_args.serving_model_dir,
               signatures=signatures, 
               save_format='tf')

    # Export the model as a pickle named model.pkl. AI Platform Prediction expects
  # sklearn model artifacts to follow this naming convention.
    

 