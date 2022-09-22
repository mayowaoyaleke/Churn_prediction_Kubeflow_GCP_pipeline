import os
from absl import logging
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from pipeline import create_pipeline
import time
suffix = int(time.time())

PIPELINE_NAME = f'churn_prediction_{suffix}'
PIPELINE_ROOT = 'gs://linear_hangar_pipeline/metadata'
DATA_PATH = 'gs://linear_hangar_pipeline/data'
SERVING_DIR = 'gs://linear_hangar_pipeline/models'

def run():
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    tfx_image = 'gcr.io/linear-hangar-362513/mlimage'
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config = metadata_config, tfx_image = tfx_image
    )

    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root= PIPELINE_ROOT,
            serving_dir= SERVING_DIR,
            data_path = DATA_PATH
        )
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()
 