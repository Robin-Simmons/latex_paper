# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:15:53 2021

@author: Robin
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from load import softmax
class Added_Weights(layers.Layer):
    def __init__(self, **kwargs):
        super(Added_Weights, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], input_shape[2]),
                                      initializer=keras.initializers.RandomUniform(minval = 0, maxval = 1),  # TODO: Choose your initializer
                                      trainable=True)
        super(Added_Weights, self).build(input_shape)

    def call(self, x, **kwargs):
        # Implicit broadcasting occurs here.
        # Shape x: (BATCH_SIZE, N, M)
        # Shape kernel: (N, M)
        # Shape output: (BATCH_SIZE, N, M)
        return x + self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape

model = keras.models.load_model("model_serial")

input_layer = layers.Input((28, 28), name = "real")
add = Added_Weights(input_shape = (28,28))(input_layer)
reshape = layers.Reshape((28,28,1))(add)
conv_layer = layers.Convolution2D(filters = 32, kernel_size = (3,3), activation = "relu", input_shape=(28,28,1), name = "conv2d_1")(reshape)
conv_layer_2 = layers.Convolution2D(filters = 64, kernel_size = (3,3), activation = "relu", name = "conv2d_2")(conv_layer)
pool = layers.MaxPooling2D((2,2))(conv_layer_2)
s_dropout = layers.SpatialDropout2D(0.2)(pool)
flatten = layers.Flatten()(s_dropout)
dense_1 = layers.Dense(128, "relu", name = "dense_1")(flatten)
dropout = layers.Dropout(0.5)(dense_1)
dense = layers.Dense(10, name = "dense_2")(dropout)
new_model = keras.Model(inputs=input_layer, outputs=dense, name="mnist_model")


for i in range(len(model.layers)-2):
    
    new_model.layers[3+i].set_weights(model.layers[2+i].get_weights())
    new_model.layers[3].trainable = False
    new_model.layers[4].trainable = False
    new_model.layers[8].trainable = False
    new_model.layers[10].trainable = False

new_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
new_model.summary()
num = 10000
data = np.repeat(np.expand_dims(np.identity(28), axis = 0), num, axis = 0)#np.zeros((num, 28, 28))
data = np.zeros((num, 28, 28))

history = new_model.fit(data, np.ones(num)*, epochs=40, verbose = 1)
for layer in new_model.layers:
    print(layer.name, layer.trainable)
#print(model.predict(np.random.rand(1,28,28)))
#print(new_model.predict(np.random.rand(1,28,28)))
eight = new_model.layers[1].get_weights()
plt.imshow(eight[0])
print(eight[0].shape)
print(softmax(model.predict(eight[0].reshape(1,28,28))))