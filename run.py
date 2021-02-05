# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:47:04 2021

@author: Robin
"""

from load import load_model_params, relu, conv_2d, dense, max_pooling_2d, softmax
from tensorflow import keras
from tensorflow.keras import datasets, activations
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,2)
def read_image(image, model_path = "model.h5"):
    
    conv1_kernel, conv1_bias, conv2_kernel, conv2_bias, dense1_weights, dense1_bias, dense2_weights, dense2_bias = load_model_params(model_path)
    
    conv1 = relu(conv_2d(np.expand_dims(image, axis = 2), conv1_kernel, conv1_bias))
    conv2 = relu(conv_2d(conv1, conv2_kernel, conv2_bias))
    pooled = max_pooling_2d(conv2, 2)
    flatten = pooled.flatten()
    dense1 = relu(dense(flatten, dense1_weights, dense1_bias))
    dense2 = dense(dense1, dense2_weights, dense2_bias)
    output = dense2
    return conv1, output

mnist = datasets.mnist

model = keras.models.load_model("model_serial")

intermediate_layer_model = keras.Model(inputs=model.input,
                                 outputs=model.get_layer("conv2d_1").output)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

arg = np.random.randint(0, len(test_labels))
conv1_kernel, conv1_bias, conv2_kernel, conv2_bias, dense1_weights, dense1_bias, dense2_weights, dense2_bias = load_model_params("model.h5")


image = test_images[arg]
print(image.shape)
intermediate_output = intermediate_layer_model.predict(test_images[arg:arg+1,:,:])
keras_out = model.predict(test_images[arg:arg+1,:,:])
mine_conv_out, mine_out = read_image(image)
ax[0,1].imshow(np.average(intermediate_output, axis = (0,3)))
ax[0,0].imshow(np.average(mine_conv_out, axis = 2))
ax[1,0].imshow(test_images[arg,:,:])

ax[1,1].plot(np.arange(0,10), keras_out[0])
ax[1,1].plot(np.arange(0,10), mine_out)
plt.show()
