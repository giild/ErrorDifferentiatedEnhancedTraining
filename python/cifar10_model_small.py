import tensorflow as tf
import time
import os
import sys

def main():
    args = sys.argv[0:]

def createModel():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(1,1), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(1,1), activation='relu', name='L1_conv2d', use_bias=False),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='L2_MaxP'),
    tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(1,1), activation='relu', name='L3_conv2d', use_bias=False),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='L4_MaxP'),
    tf.keras.layers.Conv2D(256, kernel_size=(1, 1), activation='relu', name='L5_conv2d'),
    tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', name='L6_conv2d'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='L7_MaxP'),
    tf.keras.layers.Conv2D(256, kernel_size=(1, 1), activation='relu', name='L8_conv2d'),
    tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', name='L9_conv2d'),
    tf.keras.layers.Dropout(0.290, name='L10_Drop'),
    tf.keras.layers.Flatten(name='L11_flat'),
    tf.keras.layers.Dense(128, activation='relu', name='L12_Dense'),
    tf.keras.layers.Dropout(0.5683, name='L13_Drop'),
    tf.keras.layers.Dense(10, activation='softmax', name='Dense_output')
    ], "cifar10-example-model-small")
    return model

if __name__ == "__main__":
    main()