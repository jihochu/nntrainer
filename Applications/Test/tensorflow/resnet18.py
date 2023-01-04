#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file resnet18.py.py
# @date 02 November 2020
# @brief Resnet 18 model file
# @author Jihoon lee <jhoon.it.lee@samsung.com>
# @author Parichay Kapoor <pk.kapoor@samsung.com>
# @note tested with tensorflow 2.6


import random
import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, \
    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

sys.path.insert(1, "../../../test/input_gen")
from transLayer import attach_trans_layer as TL

# Fix the seeds across frameworks
SEED = 412349
random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
np.random.seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# enable to verify values of the model layer by layer
DEBUG = False

def resnet_block(x, filters, kernel_size, downsample = False):
    y = TL(Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same"))(x)
    y = TL(BatchNormalization())(y)
    y = TL(ReLU())(y)
    y = TL(Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same"))(y)
    # y = TL(BatchNormalization())(y)

    if downsample:
        x = TL(Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same"))(x)
        # x = TL(BatchNormalization())(x)
    out = Add()([y, x])
    out = TL(BatchNormalization())(out)
    return ReLU()(out)


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(1031, 320, embeddings_regularizer=tf.keras.regularizers.l2(1e-6)))
    input_array=np.random.randint(1, size=(1,1))

    model.compile('sgd', 'mse' )
    model.summary()

    def write_fn(*items, filename='./initial.bin'):
        with open(filename, 'wb') as f:
            for item in items:
                try:
                    item.numpy().tofile(f)
                except AttributeError:
                    pass
            return items

    initial_weights = []
    for layer in model.layers:
        initial_weights += layer.weights.copy()

    print(initial_weights)

    print("---------------")
    write_fn(*initial_weights, './initial.bin')
    print("---------------")
    output_array=model.predict(input_array)
    print(output_array)

