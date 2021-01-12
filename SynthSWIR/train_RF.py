"""Script to train a random forest (RF) model and save trained model."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump

# read the csv
df = pd.read_csv('training_data.csv')

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

# convert dataframes to numpy arrays for scikit-learn RF
train_arr = train_df.to_numpy()
test_arr = test_df.to_numpy()

train_labels_arr = train_labels.to_numpy()
test_labels_arr = test_labels.to_numpy()

# multioutput regressor
regr = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                  max_depth=30,
                                                  random_state=0))

regr.fit(train_arr, train_labels_arr)

# # predict on test data
# y_regr = regr.predict(test_arr)

# save the trained model
dump(regr, 'RF_model.joblib')
