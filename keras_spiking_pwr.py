# Script to evaluate power consumption of ML models
from keras.models import load_model
from keras.layers import LSTM
import keras
import numpy as np
import keras_spiking

# register LSTM layer with kerasSpiking
@keras_spiking.ModelEnergy.register_layer(LSTM)
def LSTM_stats(node):
    cell_num = np.prod(node.output_shapes[1:])
    connect = cell_num*node.input_shapes[1]*3 - cell_num
    return {"connections": connect, "neurons": cell_num, "spiking": False}

X_test = np.load('ptb-xl_X_test_scaled.npy').astype('float16')

# create SNN shell
model = keras.Sequential()
model.add(keras.layers.TimeDistributed(keras.layers.Dense(100)))
model.add(keras_spiking.SpikingActivation("relu", dt=0.05, spiking_aware_training=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(30)))
model.add(keras_spiking.SpikingActivation("relu", dt=0.05, spiking_aware_training=True))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(units=6, activation='softmax'))
model.compile(optimizer='adam')

# SNN energy summary
model.predict(X_test)
model.summary()
energy = keras_spiking.ModelEnergy(model)
energy.summary(print_warnings=False)

# create RNN shell
model = keras.Sequential()
model.add(keras.layers.LSTM(units=62, activation='relu', return_sequences=False))
model.add(keras.layers.Dense(units=40, activation='relu'))
model.add(keras.layers.Dense(units=6, activation='softmax'))

# RNN energy summary
model.predict(X_test)
model.summary()
energy = keras_spiking.ModelEnergy(model)
energy.summary(print_warnings=False)
