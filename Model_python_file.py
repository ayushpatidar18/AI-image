
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


import pathlib
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = "/kaggle/input/dalle-recognition-dataset/train"

data_dir2 = "/kaggle/input/dalle-recognition-dataset/test"

img_height,img_width=224,224
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

img_height,img_width=224,224
batch_size=32
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir2,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names

class_names = test_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


class CustomModel(Model):
    def _init_(self):
        super(CustomModel, self)._init_()
        self.resnet = ResNet50(
            include_top=True,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg',
            classes=1000,
            classifier_activation=None
        )
        for layer in self.resnet.layers:
            layer.trainable = False
        
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(2, activation='softmax')
    
    def call(self, inputs):
        x = self.resnet(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# Create an instance of the model
model = CustomModel()

# Generate dummy input data
dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Pass the dummy input through the model
_ = model(dummy_input)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming train_ds and val_ds are your training and validation datasets respectively
epochs = 5
history = model.fit(
    train_ds,
    batch_size=32,
    validation_data=val_ds,
    epochs=epochs
)

total_examples = 0
correct_predictions = 0
for images, labels in test_ds:
    predictions = model.predict(images)
    labels = tf.cast(labels, tf.int64)
    total_examples += labels.shape[0]
    
    correct_predictions += tf.reduce_sum(tf.cast(tf.argmax(predictions, axis=1) == labels, tf.int64))

accuracy = correct_predictions / total_examples



