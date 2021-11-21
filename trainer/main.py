import os
from trainer.utils import get_data, write_filepath
from trainer.config import _resource_path, serialized_tf_config, batch_size, model_path
from trainer.model import LogisticRegression
import tensorflow as tf
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
os.environ['TF_CONFIG'] = serialized_tf_config

def main():
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])

    global_batch_size = batch_size * num_workers

    multi_worker_dataset = get_data(_resource_path, global_batch_size)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        lr = LogisticRegression()
        multi_worker_model = lr.create_model()

    multi_worker_model.fit(
        multi_worker_dataset,
        verbose= 1,
        epochs= 5
    )

    task_type, task_id = (strategy.cluster_resolver.task_type,
                        strategy.cluster_resolver.task_id)
    write_model_path = write_filepath(model_path, task_type, task_id)
    multi_worker_model.save(write_model_path)


if __name__ == "__main__":
    main()