import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import cifar10_filter_ds

def main():
    args = sys.argv[0:]
    modelfile = args[1]
    savepath = args[2] + "/weights.{epoch:02d}-{accuracy:.3f}-{loss:.3f}-{val_accuracy:.3f}-{val_loss:.3f}.h5"
    epoch = int(args[3])
    batch_size = int(args[4])
    learnrate = 0.001
    if len(args) == 6:
        learnrate = float(args[5])
        print(learnrate)
    logpath = args[2] + "/cifar_training.csv"
    trainlog = args[2] + "/train_command.log"
    train(modelfile, savepath, epoch, batch_size, learnrate, logpath)
    # save the command that was used so we don't have to guess
    tlogfile = open(trainlog,"w")
    tlogfile.write(str(args))
    tlogfile.close()

def train(modelfile, savepath, epoch, batchsize, learnrate, logpath):
    #create the model with the given python script
    model = tf.keras.models.load_model(modelfile)
    #compile the model 
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learnrate),
        metrics=['accuracy']
    )

    # create the dataset
        # the benchmark loads the CIFAR10 dataset from tensorflow datasets
    (train, test), info = cifar10_filter_ds.createDataset(batchsize)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=savepath,
                                                    save_weights_only=False,
                                                    verbose=1,
                                                    monitor='accuracy',
                                                    save_freq='epoch')
    csv_logger = tf.keras.callbacks.CSVLogger(logpath,append=True)

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