import tensorflow as tf
import pandas as pd
import numpy as np

from trainer.config import feature_names, label_name, DISTRIBUTED_MODE



class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.normalizer = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(
            int(32), input_shape= (4,), activation='relu', name='fc1')
    
        self.dense2 = tf.keras.layers.Dense(
            int(32), activation='relu', name='fc2')
        self.dense3 = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs, training= False):
        x = self.normalizer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

df = pd.read_csv('resources/Iris.csv',
        usecols= lambda x: x != 'Id')
label_to_index = {
'Iris-setosa': 0, 
'Iris-versicolor': 1, 
'Iris-virginica': 2  
} 
df['Species'] = df.Species.apply(
lambda x: label_to_index[x])

X = df[df.columns[df.columns != 'Species']]
y = df['Species']
y = tf.keras.utils.to_categorical(y)

features = np.asarray(X).astype('float32')
features = tf.convert_to_tensor(features)

y = tf.convert_to_tensor(y)

model = MyModel()
model.compile(
    optimizer= 'adam',
    loss= 'categorical_crossentropy',
    metrics=['accuracy'])

model.fit(features, y, epochs=5)
model.summary()





# class MyModel(tf.keras.Model):

#   def __init__(self):
#     super().__init__()
#     self.dense1 = tf.keras.layers.Dense(4, input_shape= (3, 3),activation=tf.nn.relu)
#     self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
#     self.dropout = tf.keras.layers.Dropout(0.5)

#   def call(self, inputs, training=False):
#     x = self.dense1(inputs)
#     if training:
#       x = self.dropout(x, training=training)
#     return self.dense2(x)

# model = MyModel()
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#               loss= 'categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit()
# model.build(input_shape= (3, 3))
# model.summary()