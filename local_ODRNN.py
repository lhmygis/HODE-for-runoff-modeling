import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Concatenate
from keras import optimizers, callbacks
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

## Import libraries developed by this study
from libs_tf.hydrolayer_tf import PRNN_NeuralODE, PRNN_NeuralODE_v1
from libs_tf.hydrodata_tf import DataforIndividual
from libs_tf import prnn_neuralode_outils

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'


working_path = 'D:\\MyGIS\\hydro_dl_project'

basin_id = '04115265'

hydrodata = DataforIndividual(working_path, basin_id).load_data()

fig, [ax1, ax2, ax3, ax4, ax5, ax6] = plt.subplots(nrows=6, ncols=1, sharex='row', figsize=(15, 18))

ax1.plot(hydrodata['prcp(mm/day)'])
ax2.plot(hydrodata['tmean(C)'])
ax3.plot(hydrodata['dayl(day)'])
ax4.plot(hydrodata['srad(W/m2)'])
ax5.plot(hydrodata['vp(Pa)'])
ax6.plot(hydrodata['flow(mm)'])

ax1.set_title(f"Basin {basin_id}")
ax1.set_ylabel("prcp(mm/day)")
ax2.set_ylabel("tmean(C)")
ax3.set_ylabel("dayl(day)")
ax4.set_ylabel("srad(W/m2)")
ax5.set_ylabel("vp(Pa)")
ax6.set_ylabel("flow(mm)")

#plt.show()



training_start = '1980-10-01'
training_end= '2000-09-30'

# The REAL evaluation period is from '2000-10-01', while the model needs one-year of data for spinning up the model
testing_start = '2000-10-01'
testing_end= '2010-09-30'

# Split data set to training_set and testing_set
train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]

print(f"The training data set is from {training_start} to {training_end}, with a shape of {train_set.shape}")
print(f"The testing data set is from {testing_start} to {testing_end}, with a shape of {test_set.shape}")


def generate_train_test(train_set, test_set, wrap_length):
    train_x_np = train_set.values[:, :-1]
    train_y_np = train_set.values[:, -1:]
    test_x_np = test_set.values[:, :-1]
    test_y_np = test_set.values[:, -1:]

    wrap_number_train = (train_set.shape[0] - wrap_length) // 365 + 1

    train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))

    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)

    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i * 365:(wrap_length + i * 365), :]
        train_y[i, :, :] = train_y_np[i * 365:(wrap_length + i * 365), :]

    return train_x, train_y, test_x, test_y


wrap_length = 2190  # It can be other values, but recommend this value should not be less than 5 years (1825 days).
train_x, train_y, test_x, test_y = generate_train_test(train_set, test_set, wrap_length=wrap_length)

print(f'The shape of train_x, train_y, test_x, and test_y after wrapping by {wrap_length} days are:')
print(f'{train_x.shape}, {train_y.shape}, {test_x.shape}, and {test_y.shape}')


def create_model(input_shape, seed, num_filters, model_type='physical'):
    x_input = Input(shape=input_shape, name='Input')

    if model_type == 'physical':
        hydro_output= PRNN_NeuralODE(mode='normal', name= 'Hydro')(x_input)
        model =Model(x_input, hydro_output)

    return model


def train_model(model, train_x, train_y, ep_number, lrate, save_path):


    es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=0, patience=20, min_delta=0.005,
                                 restore_best_weights=True)
    reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                         min_delta=0.005, cooldown=0, min_lr=lrate / 100)
    tnan = callbacks.TerminateOnNaN()

    model.compile(loss=prnn_neuralode_outils.nse_loss, metrics=[prnn_neuralode_outils.nse_metrics], optimizer=Adam(learning_rate=lrate))
    save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                     save_weights_only=True)
    history = model.fit(train_x, train_y, epochs=ep_number, batch_size=10000, callbacks=[save, es, reduce, tnan])

    return history


def test_model(model, test_x, save_path):

    model.load_weights(save_path, by_name=True)
    pred_y = model.predict(test_x, batch_size=10000)

    return pred_y



Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
save_path_physical = f'{working_path}/results/local_prnn_neuralode_models/{basin_id}.h5'

model = create_model((train_x.shape[1], train_x.shape[2]), seed = 200, num_filters = 8, model_type='physical')
model.summary()
hybrid_history = train_model(model, train_x, train_y, ep_number=200, lrate=0.01, save_path=save_path_physical)




