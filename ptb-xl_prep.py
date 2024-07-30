# Add mean-sigma scaling to ptb-xl dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_PATH = '/home/lsmithiv/Documents/Code/'

X_train = np.load(DATA_PATH + 'ptb-xl_X_train.npy')
X_test = np.load(DATA_PATH + 'ptb-xl_X_test.npy')
print(np.max(X_train))

for c, channel in enumerate(X_train):
    sc = StandardScaler().fit(channel)
    X_train[c] = sc.fit_transform(channel)

for c, channel in enumerate(X_test):
    sc = StandardScaler().fit(channel)
    X_test[c] = sc.fit_transform(channel)

print(np.max(X_train))

np.save(DATA_PATH + 'ptb-xl_X_train_scaled.npy', X_train)
np.save(DATA_PATH + 'ptb-xl_X_test_scaled.npy', X_test)
