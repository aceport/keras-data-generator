import numpy as np
import tensorflow as tf
from my_classes import DataGenerator
from tensorflow.keras import datasets, layers, models

# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = {
    'train': ['id-1', 'id-2', 'id-3'],
    'validation': ['id-4']} # IDs
labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1} # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1240,)))
model.add(layers.Dense(1, activation='linear'))
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mse']
)

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
