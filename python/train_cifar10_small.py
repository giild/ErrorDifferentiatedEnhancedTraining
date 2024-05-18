import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import cifar10_model_small
import cifar10_prepdata

def main():
    args = sys.argv[0:]
    savepath = args[1] + "/weights.{epoch:02d}-{accuracy:.3f}-{loss:.3f}-{val_accuracy:.3f}-{val_loss:.3f}.h5"
    epoch = int(args[2])
    batch_size = int(args[3])
    logpath = args[1] + "/cifar_training.csv"
    train(savepath, epoch, batch_size, logpath)

def train(savepath, epoch, batchsize, logpath):
    #create the model with the given python script
    model = cifar10_model_small.createModel()
    #compile the model 
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )

    # create the dataset
        # the benchmark loads the CIFAR10 dataset from tensorflow datasets
    (train, test), info = cifar10_prepdata.createDataset(batchsize)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=savepath,
                                                    save_weights_only=False,
                                                    verbose=1,
                                                    monitor='accuracy',
                                                    save_freq='epoch')
    csv_logger = tf.keras.callbacks.CSVLogger(logpath,append=True)
    print(model.summary())

    start_time = time.time()

    model.fit(
        train,
        epochs=epoch,
        validation_data=test,
        batch_size=batchsize,
        callbacks=[cp_callback, csv_logger]
    )
    end_time = time.time()
    print('Test loss:', model.loss)
    print(model.summary())
    print('Elapsed Time: %0.4f seconds' % (end_time - start_time))
    print('Elapsed Time: %0.4f minutes' % ((end_time - start_time)/60))

if __name__ == "__main__":
    main()