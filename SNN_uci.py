# Implement an SNN to solve Breast Cancer dataset

import keras 
import numpy as np
import pandas as pd
import keras_spiking
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout, Reshape, GlobalAveragePooling1D, TimeDistributed

# Global variables
epochs = 50
HIST_PATH = '/home/lsmithiv/Documents/Code/Model_hist/6-21_SNN_bc/'
accs = []
trains = 200
def scheduler(epoch, lr):
    if (epoch >= 100): return 0.00025
    elif (epoch >= 20): return 0.001
    else: return 0.005

# Load, prepare dataset
dataset = load_breast_cancer()
X_train = dataset.data
train_labels = dataset.target
X_train, X_test, train_labels, test_labels = train_test_split(X_train, train_labels, test_size=0.2, random_state=37321)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define repeatable training function
def train():
    # Define model architecture
    model = keras.Sequential()
    model.add(Reshape((-1, X_train.shape[1])))
    model.add(TimeDistributed(Dense(units=50)))
    model.add(keras_spiking.SpikingActivation('relu', dt=0.05, spiking_aware_training=True))
    model.add(TimeDistributed(Dense(units=100)))
    model.add(keras_spiking.SpikingActivation('relu', dt=0.05, spiking_aware_training=True))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile model
    optim = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(
        optimizer = optim,
        loss = keras.losses.BinaryCrossentropy(),
        metrics = ['accuracy'],
    )

    ratescheduler = keras.callbacks.LearningRateScheduler(scheduler)
    #earlystop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=.0001)

    # Train model
    history = model.fit(X_train, train_labels,
            epochs=epochs,
            verbose=0,
            callbacks=[ratescheduler]
    )
    acc_hist = history.history['accuracy']
    loss_hist = history.history['loss']

    # Eval trained model
    results = model.evaluate(X_test, test_labels)
    return results[1]
# end train

# loop training, save results
for i in range(trains):
    accs.append(train())
print("Average test accuracy: ", np.average(accs))
print("Max test accuracy: ", max(accs))

# Save training history
with open(HIST_PATH+'training_results.txt', 'w') as file:
    for a in accs:
        file.write("%s\n" % a)
    file.close()
