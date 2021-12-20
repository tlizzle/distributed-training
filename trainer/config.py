import json

_resource_path = 'resources/Iris.csv'
model_path = 'gs://tessorflow-store/model_dir'


tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

serialized_tf_config = json.dumps(tf_config)


feature_names = [
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]

label_to_index = {
    'Iris-setosa': 0, 
    'Iris-versicolor': 1, 
    'Iris-virginica': 2  
} 

label_name = "Species"
num_samples = 5
batch_size = 64

DISTRIBUTED_MODE = False