# Code to prepare, partition SEED dataset

# Prepare environment
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat

PATH = '/home/lsmithiv/Downloads/Preprocessed_EEG/'
DATA_PATH = '/home/lsmithiv/Documents/Code/'

# Define data formatting method
def format_trials(in_trials):
    X_out = np.empty((len(in_trials), 30000, 62), dtype='float32')
    for t, trial in enumerate(in_trials):
        temp = np.swapaxes(trial, 0, 1)
        temp = np.asarray(temp[100:30100])
        sc = StandardScaler().fit(temp)
        temp = sc.fit_transform(temp)
        X_out[t] = temp
    return X_out

# Define data preparation
def data_prep():
    print("Collecting dataset...")
    # Collect file list for dataset
    experiments = []
    for root, directories, file in os.walk(PATH):
        for file in file:
            if (file.endswith(".mat")):
                if (file == "label.mat"): labels = loadmat(PATH+file)
                else: experiments.append(file)

    # Retrieve dataset
    trials = []
    for file in experiments:
        data = loadmat(PATH+file)
        del data['__globals__'], data['__header__'], data['__version__']
        # 15 trials per experiment file
        for trial in data:
            trials.append(np.matrix(list(data[trial]), dtype='float32'))

    # Format labels
    print("Preprocessing dataset...")
    labels = np.matrix(labels['label'])+1
    labels = np.swapaxes(labels, 0, 1) #shape: (15,1)
    labels = np.tile(labels, (int(len(trials)/15), 1)) #shape: (675,1)

    # Format trials
    trials = format_trials(trials)

    # Select 3sec snippets for training
    print("Subsampling dataset...")
    X_train = np.empty((int(len(trials)*50), 1000, trials.shape[2]), dtype='float16')
    train_labels = np.repeat(labels, 50, axis=0) #shape: (675'000,1)
    for t, trial in enumerate(trials):
        # 100 snippets per trial
        for i in range(50):
            # random index [0,(30000-600)]
            r = np.random.randint(trial.shape[0]-1000)
            # select 600 pts after r
            X_train[t*50+i] = trial[r:r+1000]
    print(X_train.shape)

    # Split snippets 60/20/20
    print("Splitting dataset...")
    test_trials = np.empty((0, X_train.shape[1], X_train.shape[2]), dtype='float16')
    valid_trials = np.empty((0, X_train.shape[1], X_train.shape[2]), dtype='float16')
    test_labels = np.empty((0, 1), dtype=int)
    valid_labels = np.empty((0, 1), dtype=int)
    for i in range(int(len(X_train)*.2)):
        if (i % 100 == 0): print(i)
        # test split
        r = np.random.randint(len(X_train))
        test_trials = np.append(test_trials, np.reshape(X_train[r], (1, X_train.shape[1], X_train.shape[2])), axis=0)
        X_train = np.delete(X_train, r, axis=0)
        test_labels = np.append(test_labels, np.reshape(train_labels[r], (1,1)), axis=0)
        train_labels = np.delete(train_labels, r, axis=0)
        # valid split
        r = np.random.randint(len(X_train))
        valid_trials = np.append(valid_trials, np.reshape(X_train[r], (1, X_train.shape[1], X_train.shape[2])), axis=0)
        X_train = np.delete(X_train, r, axis=0)
        valid_labels = np.append(valid_labels, np.reshape(train_labels[r], (1,1)), axis=0)
        train_labels = np.delete(train_labels, r, axis=0)

    # Write train, valid, test dataset to files
    print("Saving dataset...")
    np.save(DATA_PATH+'X_test5half', test_trials)
    np.save(DATA_PATH+'test_labels5half', test_labels)
    np.save(DATA_PATH+'X_valid5half', valid_trials)
    np.save(DATA_PATH+'valid_labels5half', valid_labels)
    np.save(DATA_PATH+'X_train5half', X_train)
    np.save(DATA_PATH+'train_labels5half', train_labels)
    print("Data preparation complete.")

# END data_prep

data_prep()

