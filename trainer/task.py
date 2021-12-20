import os
from trainer.utils import get_data, write_filepath, get_args
from trainer.config import _resource_path, batch_size, model_path
from trainer.model import LogisticRegression
import tensorflow as tf
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
# os.environ['TF_CONFIG'] = serialized_tf_config

def main():
    args = get_args()
    # tf_config = json.loads(os.environ['TF_CONFIG'])
    # num_workers = len(tf_config['cluster']['worker'])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    global_batch_size = batch_size * strategy.num_replicas_in_sync
    multi_worker_dataset = get_data(_resource_path, global_batch_size)
    # multi_worker_dataset = multi_worker_dataset.repeat()

    with strategy.scope():
        lr = LogisticRegression()
        multi_worker_model = lr.create_model()

    multi_worker_model.fit(
        multi_worker_dataset,
        verbose= 1,
        epochs= args.epochs,
        steps_per_epoch= 10,
        validation_steps= None,
    )

    task_type, task_id = (strategy.cluster_resolver.task_type,
                        strategy.cluster_resolver.task_id)
    write_model_path = write_filepath(args.job_dir, task_type, task_id)
    multi_worker_model.save(write_model_path)

if __name__ == "__main__":
    main()