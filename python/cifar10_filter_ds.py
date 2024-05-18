import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def filterByLabel(train, indexlist):
    def filterfn(ing, label):
        return tf.reduce_any(tf.equal(label, indexlist))
    filtered = train.filter(filterfn)
    return filtered

def createDataset(batchsize):
        # create the dataset
        # the benchmark loads the CIFAR10 dataset from tensorflow datasets
    (train,test),info  = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    cindexs = [2,3,4,6] # [2,3,4,6] # bird, cat, deer, dog, frog
    #cindexs = [0,5,7,9] # [2,3,4,6] # dog, airplane, truck, 
    # train = train.filter(lambda img, label: label == 3)
    train = filterByLabel(train, cindexs)
   
    train = train.map(normalize_img)
    train = train.cache()
    train = train.shuffle(info.splits['train'].num_examples)
    train = train.batch(batchsize)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    # test = test.filter(lambda img, label: label == 3)
    test = filterByLabel(test, cindexs)
    test = test.map(normalize_img)
    test = test.batch(batchsize)
    test = test.cache()
    test = test.prefetch(tf.data.experimental.AUTOTUNE)
    return (train, test), info
