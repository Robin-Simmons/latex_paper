# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 21:59:56 2021

@author: rs177
"""
import numpy as np
from scipy.signal import correlate2d
from numpy import correlate


class layer:
    """
    Base class for neural net layers.

    Parameters
    ----------
    in_shape : array
        shape of input array.
    out_shape : array
         shape of output array.
    activation : str, optional
        activation function to use, none is idenity. The default is "none".
    bias_bool : bool, optional
        use bias or not. The default is True.
    """

    def __init__(self, in_shape, out_shape, activation="none", bias_bool=True):

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.bias_bool = bias_bool
        if bias_bool:
            self.bias = np.zeros(out_shape)

        try:
            self.activation = globals()[activation]
        except KeyError:
            print(f"Error: {activation} is not a valid activation")

    def evalulate(self, x):
        """
        Evaluate layer for input x.

        Parameters
        ----------
        x : array
            input array

        Returns
        -------
        array
            output array.

        """
        if self.bias_bool:
            return self.activation(self._layer_func(x) + self.bias)
        else:
            return self.activation(self._layer_func(x))

    def init_weights(self, dist="random_sample", **kwargs):
        """
        Initalise layer weights.

        Parameters
        ----------
        dist : str, optional
            numpy.random method. The default is "random_sample".
        **kwargs : any
            arguments for random method.

        Returns
        -------
        None.

        """
        try:
            rand_func = getattr(np.random, dist)
        except KeyError:
            print(f"Error: {dist} is not a valid random function")

        self.weights = rand_func(self.weights.shape, **kwargs)
        if self.bias_bool:
            self.bias = rand_func(self.out_shape)


class dense(layer):
    """
    Standard Dense layer. Inherits from layer class.

    Parameters
    ----------
    in_shape : array
        shape of input array.
    out_shape : array
         shape of output array.
    activation : str, optional
        activation function to use, none is idenity. The default is "none".
    bias_bool : bool, optional
        use bias or not. The default is True.
    layer_name : str, optional
        layer name. The default is "dense".

    Returns
    -------
    None.

    """

    def __init__(self, in_shape, out_shape, activation="relu", bias_bool=True,
                 layer_name="dense"):

        layer.__init__(self, in_shape, out_shape,
                       activation=activation, bias_bool=bias_bool)
        self.layer_name = layer_name
        self.weights = np.zeros((in_shape, out_shape))

    def _layer_func(self, x):
        if x.shape[0] != self.in_shape:
            raise Exception(f"x shape {x.shape[0]}"
                            "must match in_shape {self.in_shape}")
        return np.dot(x, self.weights) + self.bias


class conv1d(layer):
    """
    1d convolution (cross-correlation) layer. Inherits from layer class.

    Parameters
    ----------
    in_shape : array
        shape of input array.
    kernel_size : int
    n_filters : int
    packing : str, optional
        packing type. The default is "valid".
    activation : str, optional
        activation function to use, none is idenity. The default is "relu".
    bias_bool : bool, optional
        use bias or not. The default is True.
    layer_name : str, optional
        layer name. The default is "dense".

    Returns
    -------
    None.

    """

    def __init__(self, in_shape, kernel_size, n_filters, packing="valid",
                 activation="relu", bias_bool=True, layer_name="conv1d"):

        if packing == "valid":
            out_shape = in_shape - kernel_size + 1

        layer.__init__(self, in_shape, out_shape,
                       activation=activation, bias_bool=bias_bool)
        self.layer_name = layer_name
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.weights = np.zeros((n_filters, kernel_size))

    def _layer_func(self, x):
        conv_sum = 0
        for i in range(self.n_filters):
            conv_sum += correlate(x, self.weights[i, :], "valid")
        return conv_sum


class max_pool1d(layer):
    """
    Maxpooling layer. Inherits from layer class.

    Parameters
    ----------
    in_shape : int
        input shape.
    pool_size : int
        max pooling size.
    packing : str, optional
        packing type. The default is "valid".
    layer_name : str, optional
        layer name. The default is "max_pool1d".

    Returns
    -------
    None.

    """

    def __init__(self, in_shape, pool_size, packing="valid",
                 layer_name="max_pool1d"):

        out_shape = int((in_shape - pool_size)/2) + 1
        layer.__init__(self, in_shape, out_shape,
                       activation="none", bias_bool=False)
        self.pool_size = pool_size

    def _layer_func(self, x):
        out_array = np.empty(self.out_shape)

        for i in range(self.out_shape):
            out_array[i] = np.max(x[i:i+2])

        return out_array

    @property
    def init_weights(self, **kwargs):
        raise AttributeError("max_pool1d has no init_weights, as it has no weights")


def relu(x):
    """ReLU activation."""
    return x.clip(0)


def none(x):
    """Idenity activation."""
    return x


def softmax(x):
    """SoftMax activation."""
    exp = np.exp(x)
    return exp/np.sum(exp)

layer1 = conv1d(10, 10, 10, activation = "test")