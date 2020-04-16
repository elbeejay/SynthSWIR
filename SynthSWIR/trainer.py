import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# read the csv
df = pd.read_csv('/home/jayh/Documents/SynthSWIR/training_data/training_data.csv')

# drop unnamed column
df.pop('Unnamed: 0')

# split dataset into train and test
train_df = df.sample(frac=0.6, random_state=0)
test_df = df.drop(train_df.index)

# break out the variables we are trying to fit
train_labels_1 = train_df.pop('SWIR1')
train_labels_2 = train_df.pop('SWIR2')

test_labels_1 = test_df.pop('SWIR1')
test_labels_2 = test_df.pop('SWIR2')

train_int = {'SWIR1': train_labels_1,
             'SWIR2': train_labels_2}

test_int = {'SWIR1': test_labels_1,
            'SWIR2': test_labels_2}

train_labels = pd.DataFrame(train_int)

test_labels = pd.DataFrame(test_int)


# try to build a model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_df.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

# create checkpoint and save the model weights
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# train the model
EPOCHS = 1000

history = model.fit(train_df, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots(),cp_callback])

# make predictions using the model
test_predictions = model.predict(test_df)

# plot test predictions against real data
fig = plt.figure()

plt.subplot(1,2,1)
test_truth = test_labels.to_numpy()
plt.scatter(test_truth[:,0],test_predictions[:,0])
m_pre = np.max(test_predictions[:,0])
m_tru = np.max(test_truth[:,0])
lims = [0, np.max([m_pre,m_tru])]
plt.plot(lims, lims)
plt.xlabel('SWIR-1 Truth')
plt.ylabel('SWIR-1 Predictions')

plt.subplot(1,2,2)
test_truth = test_labels.to_numpy()
plt.scatter(test_truth[:,1],test_predictions[:,1])
m_pre = np.max(test_predictions[:,1])
m_tru = np.max(test_truth[:,1])
lims = [0, np.max([m_pre,m_tru])]
plt.plot(lims, lims)
plt.xlabel('SWIR-2 Truth')
plt.ylabel('SWIR-2 Predictions')

fig.savefig('results.png')
