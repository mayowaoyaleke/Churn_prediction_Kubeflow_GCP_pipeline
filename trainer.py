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

NUMERIC_FEATURE_KEYS = [
    'Age','Balance','CreditScore','CustomerId','EstimatedSalary','HasCrCard','IsActiveMember','NumOfProducts','RowNumber','Tenure'
]

VOCAB_FEATURE_DICT = {
    'Gender':2,'Geography':3,'Surname':1169
}


_FEATURE_KEYS = [
     'Age','Balance','CreditScore','CustomerId','EstimatedSalary','HasCrCard','IsActiveMember','NumOfProducts','RowNumber','Tenure', 'Gender','Geography','Surname'
]
_LABEL_KEY = 'Exited'

NUM_OOV_BUCKETS = 2

LABEL_KEY = 'Exited'

# Renamimg Features   
def transformed_name(key):
    key = key.replace('-','_')
    return key + '_xf'


#Define Callbacks
# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 10)

#Load compressede data
def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type= 'GZIP')






#Load data#################################################################################################
def _input_fn(file_pattern: str, tf_transform_output: tft.TFTransformOutput, num_epochs= None, batch_size = 128) -> tf.data.Dataset:
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


# def _input_fn(
#     file_pattern: str,
#     data_accessor: DataAccessor,
#     schema: schema_pb2.Schema,
#     batch_size: int = 20,
# ) -> Tuple[np.ndarray, np.ndarray]:

#   record_batch_iterator = data_accessor.record_batch_factory(
#       file_pattern,
#       dataset_options.RecordBatchesOptions(batch_size=batch_size, num_epochs=1),
#       schema)

#   feature_list = []
#   label_list = []
    
#   for record_batch in record_batch_iterator:
#     record_dict = {}
#     for column, field in zip(record_batch, record_batch.schema):
#       record_dict[field.name] = column.flatten()

#     label_list.append(record_dict[_LABEL_KEY])
#     features = [record_dict[key] for key in _FEATURE_KEYS]
#     feature_list.append(np.stack(features, axis=-1))

#   return np.concatenate(feature_list), np.concatenate(label_list)

#Build model
def model_builder():
    model = RandomForestClassifier()

    
    # num_hidden_layers = hp.Int('hidden_layers', min_value=1, max_value= 5)
    # hp_learning_rate = hp.Choice('learning_rate', values = [1e-2,1e-3,1e-4])

    # input_numeric = [
    #     tf.keras.layers.Input(name = transformed_name(colname), shape=(1,), dtype= tf.float32) for colname in NUMERIC_FEATURE_KEYS
    # ]

    # input_categorical = [
    #     tf.keras.layers.Input(name = transformed_name(colname), shape=(vocab_size + NUM_OOV_BUCKETS,), dtype= tf.float32) for colname, vocab_size in VOCAB_FEATURE_DICT.items()
    # ]

    # input_numeric = tf.keras.layers.concatenate(input_numeric)
    # input_categorical = tf.keras.layers.concatenate(input_categorical)

    # deep = tf.keras.layers.concatenate([input_numeric, input_categorical])

    # for i in range(num_hidden_layers):
    #     num_nodes = hp.Int('unit'+ str(i), min_value = 8, max_value=256, step = 64)
    #     deep = tf.keras.layers.Dense(num_nodes, activation = 'relu')(deep)

    # output = tf.keras.layers.Dense(1, activation ='sigmoid')(deep)

    # input_layers = input_numeric + input_categorical

    # model = tf.keras.Model(input_layers, output)

    # model.compile(
    #     loss = 'binary_crossentropy',
    #     optimizer = tf.keras.optimizers.Adam(learning_rate= hp_learning_rate),
    #     metrics = 'binary_accuracy'
    # )

    #print model
    # Trmodel.summary()
    return model

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

    x_train, y_train = _input_fn(fn_args.train_files, fn_args.data_accessor,
                               schema)
    x_eval, y_eval = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)

    # Build the model
    model = model_builder()

    # Train the model
    # model.fit(X=x_train,y=y_train)

    # # model.fit(x = train_set,
    # # validation_data = eval_set,
    # # callbacks = [tensorboard_callback, es, mc],
    # # epochs = 100)

    # score = model.score(x_eval, y_eval)
    
    model.feature_keys = _FEATURE_KEYS
    model.label_key = _LABEL_KEY
    model.fit(x_train, y_train)
    absl.logging.info(model)

    score = model.score(x_eval, y_eval)
    absl.logging.info('Accuracy: %f', score)
    

    # Export the model as a pickle named model.pkl. AI Platform Prediction expects
  # sklearn model artifacts to follow this naming convention.
    

 