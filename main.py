# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def test_model(model, test_im, test_lb):
    
    inference = model(test_im)
    
    total = np.sum(np.where(np.argmax(inference, axis = 1) == test_lb, 1, 0))
    return total/10000
    
fig, ax = plt.subplots(3)
mnist = keras.datasets.mnist
#ds, info = tfds.load("mnist_corrupted", split = "train")
#mnist_corrupted = keras.datasets.mnist_corrupted
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#(train_images_c, train_labels_c), (test_images_c, test_labels_c) = mnist_corrupted.load_data()
# Define network

input_layer = layers.Input((28, 28))
reshape = layers.Reshape((28,28,1))(input_layer)
conv_layer = layers.Convolution2D(filters = 32, kernel_size = (3,3), activation = "relu", input_shape=(28,28,1), name = "conv2d_1")(reshape)
conv_layer_2 = layers.Convolution2D(filters = 64, kernel_size = (3,3), activation = "relu", name = "conv2d_2")(conv_layer)
pool = layers.MaxPooling2D((2,2))(conv_layer_2)
s_dropout = layers.SpatialDropout2D(0.2)(pool)
flatten = layers.Flatten()(s_dropout)
dense_1 = layers.Dense(128, "relu", name = "dense_1")(flatten)
dropout = layers.Dropout(0.5)(dense_1)
dense = layers.Dense(10, name = "dense_2")(dropout)
model = keras.Model(inputs=input_layer, outputs=dense, name="mnist_model")
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=1, callbacks = [early_stopping], 
                    verbose = 1, validation_data=(test_images, test_labels))
rand_index = np.random.randint(0,                                                                                                                                                  len(test_labels))
new_image = tf.expand_dims(test_images[0], 0)
ax[0].imshow(test_images[0])
#print(test_model(model, test_images, test_labels))
model.save_weights("model.h5")
model.save('model_serial')

#epochs = len(history.history["loss"])
#ax[1].bar(np.arange(0,10), keras.activations.softmax(model(new_image))[0])
#ax[2].plot(np.arange(0,epochs), history.history["val_loss"])
#ax[2].plot(np.arange(0,epochs), history.history["loss"])
