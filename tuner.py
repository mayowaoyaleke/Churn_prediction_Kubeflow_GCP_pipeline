#Define imports
from kerastuner.engine import base_tuner
import kerastuner as kt
from tensorflow import keras

from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs
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


TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                            ('fit_kwargs', Dict[Text,Any])])
#Define Callbacks
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 10)


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type= 'GZIP')

def _input_fn(file_pattern, tf_transform_output, num_epochs= None, batch_size = 128) -> tf.data.Dataset:
    
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

def model_builder(hp):
    
    num_hidden_layers = hp.Int('hidden_layers', min_value=1, max_value= 5)
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2,1e-3,1e-4])

    input_numeric = [
        tf.keras.layers.Input(name = transformed_name(colname), shape=(1,), dtype= tf.float32) for colname in NUMERIC_FEATURE_KEYS
    ]

    input_categorical = [
        tf.keras.layers.Input(name = transformed_name(colname), shape=(vocab_size + NUM_OOV_BUCKETS,), dtype= tf.float32) for colname, vocab_size in VOCAB_FEATURE_DICT.items()
    ]

    input_numeric = tf.keras.layers.concatenate(input_numeric)
    input_categorical = tf.keras.layers.concatenate(input_categorical)

    deep = tf.keras.layers.concatenate([input_numeric, input_categorical])

    for i in range(num_hidden_layers):
        num_nodes = hp.Int('unit'+ str(i), min_value = 8, max_value=256, step = 64)
        deep = tf.keras.layers.Dense(num_nodes, activation = 'relu')(deep)

    output = tf.keras.layers.Dense(1, activation ='sigmoid')(deep)

    input_layers = input_numeric + input_categorical

    model = tf.keras.Model(input_layers, output)

    model.compile(
        loss = 'binary_crossentropy',
        optimizer = tf.keras.optimizers.Adam(learning_rate= hp_learning_rate),
        metrics = 'binary_accuracy'
    )

    model.summary()

    return model

def tuner_fn(fn_args: FnArgs)-> TunerFnResult:
    tuner = kt.Hyperband(
        model_builder,
        objective='val_binary_accuracy',
        max_epochs = 20,
        factor = 2,
        directory = fn_args.working_dir,
        project_name = 'kt_hyperband'
    )

    #load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = _input_fn(fn_args.train_files,tf_transform_output, 10)
    val_set = _input_fn(fn_args.eval_files,tf_transform_output, 10)

    return TunerFnResult(
        tuner = tuner,
        fit_kwargs={
            'callbacks':[stop_early],
            'x':train_set,
            'validation_data' : val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )