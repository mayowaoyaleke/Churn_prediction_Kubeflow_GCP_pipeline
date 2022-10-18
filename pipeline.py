
import statistics
from urllib.parse import scheme_chars
from wsgiref.validate import validator
from tfx.proto import example_gen_pb2
from tfx.orchestration import pipeline
import os
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Tuner
from tfx.components import Trainer
from tfx.proto import trainer_pb2

from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel

import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.proto import pusher_pb2

churn_transform_module_file = 'churn_transform2.py'
tuner_module_file = 'tuner.py'
trainer_module_file = 'trainer.py'

def create_pipeline(
    pipeline_name,
    pipeline_root,
    data_path,
    serving_dir,
    metadata_connection_config =None,
    beam_pipeline_args = None
):
    components= []

    #example gen
    output = example_gen_pb2.Output(
        split_config= example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name = 'train', hash_buckets = 8),
            example_gen_pb2.SplitConfig.Split(name = 'eval', hash_buckets = 2)
        ])
    )

    example_gen = CsvExampleGen(input_base = data_path, output_config = output)
    components.append(example_gen)

    #statistcs gen 
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    #schema gen
    schema_gen = SchemaGen(statistics =statistics_gen.outputs['statistics'])
    components.append(schema_gen)

    #example validator
    validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'],
    schema= schema_gen.outputs['schema'])
    components.append(validator)

    #transform  
    transform = Transform(examples=example_gen.outputs['examples'],
    schema= schema_gen.outputs['schema'],
    module_file= churn_transform_module_file)
    components.append(transform)

    #tuner
    # tuner = Tuner(examples=transform.outputs['transformed_examples'],
    # schema= schema_gen.outputs['schema'],
    # transform_graph= transform.outputs['transform_graph'],
    # module_file= tuner_module_file,
    # train_args= trainer_pb2.TrainArgs(splits = ['train'], num_steps = 200),
    # eval_args= trainer_pb2.EvalArgs(splits = ['eval'], num_steps = 50)
    # )
    # components.append(tuner)

    #trainer
    trainer = Trainer(examples=transform.outputs['transformed_examples'],
    schema= schema_gen.outputs['schema'],
    transform_graph= transform.outputs['transform_graph'],
    module_file= trainer_module_file,
    train_args= trainer_pb2.TrainArgs(splits = ['train'],num_steps = 200),
    eval_args= trainer_pb2.EvalArgs(splits = ['eval'],num_steps = 50)
    )
    components.append(trainer)

    #resolver
    model_resolver = Resolver(
    strategy_class= LatestBlessedModelStrategy,
    model = Channel(type = Model),
    model_blessing = Channel(type = ModelBlessing)
    ).with_id("latest_blessed_model_resolver")
    components.append(model_resolver)

    #evaluator
    eval_config = tfma.EvalConfig(
    model_specs = [tfma.ModelSpec(label_key = 'Exited')],
    slicing_specs = [tfma.SlicingSpec()],
    metrics_specs = [
        tfma.MetricsSpec(metrics = [
            tfma.MetricConfig(class_name = "ExampleCount"),
            tfma.MetricConfig(class_name = "AUC"),
            tfma.MetricConfig(class_name = "FalsePositives"),
            tfma.MetricConfig(class_name = "BinaryAccuracy",
                              threshold = tfma.MetricThreshold(
                                  value_threshold = tfma.GenericValueThreshold(
                                      lower_bound = {"value": 0.5}
                                  ),
                                  change_threshold = tfma.GenericChangeThreshold(
                                      direction = tfma.MetricDirection.HIGHER_IS_BETTER,
                                      absolute = {"value": 0.001}
                                  )
                              ))
        ])
    ])

    evaluator = Evaluator(examples=example_gen.outputs['examples'],
    model= trainer.outputs['model'],
    baseline_model= model_resolver.outputs['model'],
    eval_config= eval_config)
    components.append(evaluator)

    #Pusher
    pusher = Pusher(model = trainer.outputs['model'],
                model_blessing = evaluator.outputs['blessing'],
                push_destination = pusher_pb2.PushDestination(
                    filesystem = pusher_pb2.PushDestination.Filesystem(
                        base_directory = serving_dir
                    ))
                )
    components.append(pusher)

    return pipeline.Pipeline(
        pipeline_name = pipeline_name,
        pipeline_root = pipeline_root,
        components = components,
        metadata_connection_config = metadata_connection_config,
        beam_pipeline_args = beam_pipeline_args
    )




# tfx pipeline create --pipeline-path=kubeflow_dag_runner.py --endpoint=https://25430b44e036f94d-dot-us-central1.pipelines.googleusercontent.com/
