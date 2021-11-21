import os
import pandas as pd
import numpy as np
from trainer.config import label_to_index
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_data(_path, batch_size):
    # _path = 'resources/Iris.csv'

    df = pd.read_csv(_path,
        usecols= lambda x: x != 'Id'
    )

    df['Species'] = df.Species.apply(
                    lambda x: label_to_index[x])

    X = df[df.columns[df.columns != 'Species']]
    y = df['Species']
    y = tf.keras.utils.to_categorical(y)

    features = np.asarray(X).astype('float32')
    y = np.asarray(y).astype('int64')

    X_train, X_valid, y_train, y_valid = train_test_split(
            features, y , test_size= 0.2, random_state= 1
    )
    # features = tf.convert_to_tensor(features)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(60000).repeat().batch(64)


    return train_dataset


model_path = '/tmp/keras-model'

def _is_chief(task_type, task_id):
    return (task_type == 'worker' and task_id == 0) or task_type is None

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir

def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)


