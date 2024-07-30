# Implement an MLP to solve Breast Cancer dataset

import keras 
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout

# Global variables
epochs = 50
HIST_PATH = '/home/lsmithiv/Documents/Code/Model_hist/test/'
trains = 200
accs = []

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

# Repeatable training func
def train():
    # Define model architecture
    model = keras.Sequential()
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile model
    optim = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(
        optimizer = optim,
        loss = keras.losses.BinaryCrossentropy(),
        metrics = ['accuracy'],
    )

    ratescheduler = keras.callbacks.LearningRateScheduler(scheduler)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath = HIST_PATH+'model_weights.tf',
        monitor = "loss",
        mode = 'min',
        verbose = 1,
        save_best_only = True
    )

    # Train model
    history = model.fit(X_train, train_labels,
            epochs=epochs,
            verbose=0,
            callbacks=[ratescheduler, checkpoint]
    )
    acc_hist = history.history['accuracy']
    loss_hist = history.history['loss']

    # Eval trained model
    test = model.evaluate(X_test, test_labels)
    return test[1]
# end train

# Loop training, save results
for i in range(trains):
    accs.append(train())
print("Average test accuracy: ", np.average(accs))
print("Max test accuracy: ", max(accs))

# Save training history
with open(HIST_PATH+'training_results.txt', 'w') as file:
    for a in accs:
        file.write("%s\n" % a)
    file.close()
