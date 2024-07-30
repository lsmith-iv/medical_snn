# Code to train and evaluate an SNN

# Prepare environment
import numpy as np
import keras
import keras_spiking
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.losses import CategoricalCrossentropy

DATA_PATH = '/home/lsmithiv/Documents/Code/'
HIST_PATH = '/home/lsmithiv/Documents/Code/Model_hist/7-27_SNN_30_nospike_ptb/'


# Define training method
def train(model, epochs):
    # Retrieve preprocessed dataset from files
    X_train = np.load(DATA_PATH+'ptb-xl_X_train_scaled.npy').astype('float16')
    train_labels = np.load(DATA_PATH+'ptb-xl_train_labels.npy').astype('int')
    #X_valid = np.load(DATA_PATH+'X_valid2.npy').astype('float16')
    #valid_labels = np.load(DATA_PATH+'valid_labels2.npy')
    X_train, X_valid, train_labels, valid_labels = train_test_split(X_train, train_labels, test_size=.20, random_state=3)
    X_test = np.load(DATA_PATH+'ptb-xl_X_test_scaled.npy').astype('float16')
    test_labels = np.load(DATA_PATH+'ptb-xl_test_labels.npy').astype('int')

    # Compile model, prepare for training
    optim = keras.optimizers.Adam(learning_rate=0.0025)
    model.compile(
            optimizer = optim,
            loss = CategoricalCrossentropy(from_logits=False),
            metrics = ['accuracy'],
    )
    start_time = datetime.now()
    print('Begin training at: ', start_time)

    # define callback to print time
    class dtime(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("\nTime elapsed: ", datetime.now()-start_time)

    # define learning scheduler callback
    def scheduler(epoch, lr):
        if (epoch >= 100): return 0.00025
        elif (epoch >= 30): return 0.001
        else: return 0.0025
    ratescheduler = keras.callbacks.LearningRateScheduler(scheduler)

    # define model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath = HIST_PATH+'model.weights.h5',
        monitor = 'val_loss',
        save_weights_only = True,
        save_best_only = True
        )

    # define early stopping callback
    earlystop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=.0001)

    # define dt scheduler callback
    dt = tf.Variable(0.5)
    dt_scheduler = keras_spiking.callbacks.DtScheduler(
            dt, tf.optimizers.schedules.ExponentialDecay(
                1.0, decay_steps=10000, decay_rate=0.01
            )
            )

    # train model, save training accuracy, loss
    history = model.fit(X_train, train_labels,
            epochs=epochs,
            verbose=1,
            validation_data=(X_valid, valid_labels),
            #validation_split=0.2,
            callbacks=[ratescheduler, dtime(), checkpoint]
            )
    acc_hist = history.history['accuracy']
    loss_hist = history.history['loss']
    vloss_hist = history.history['val_accuracy']
    vacc_hist = history.history['val_loss']

    # evaluate trained model
    model.load_weights(HIST_PATH+'model.weights.h5')
    model.evaluate(X_test, test_labels)

    return loss_hist, acc_hist, vloss_hist, vacc_hist
# end training function


# construct SNN for training
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Input, GlobalAveragePooling1D
SNN = keras.Sequential()

# input shape: (time, channels)
SNN.add(Input(shape=(1000,12)))
SNN.add(TimeDistributed(Dense(units=30)))
SNN.add(keras_spiking.SpikingActivation('relu', dt=0.05, spiking_aware_training=False))
SNN.add(TimeDistributed(Dense(units=70)))
SNN.add(keras_spiking.SpikingActivation('relu', dt=0.05, spiking_aware_training=False))
SNN.add(GlobalAveragePooling1D())
SNN.add(Dropout(0.1))
SNN.add(Dense(units=6, activation='softmax'))


# train RNN, save history
losses, accs, valid_l, valid_a = train(SNN, 200)
with open(HIST_PATH+'training_results.txt', 'w') as file:
    file.write("loss history:\n")
    for loss in losses:
        file.write("%s\n" % loss)
    file.write("accuracy history:\n")
    for acc in accs:
        file.write("%s\n" % acc)
    file.write("validation loss history:\n")
    for vloss in valid_l:
        file.write("%s\n" % vloss)
    file.write("validation accuracy history:\n")
    for vacc in valid_a:
        file.write("%s\n" % vacc)
    file.close()
