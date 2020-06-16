"""Test SWIR predictions on a LANDSAT image against the actual values."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdal
from simple_model import build_model


def predict_comparison(file_name):
    """Run prediction and comparison for a file."""
    # trim file extension if provided
    len_f = len(file_name)
    if file_name[len_f-4:len_f] == '.tif':
        file_name = file_name[0:len_f-4]

    # load the image to be tested
    og = gdal.Open(file_name + '.tif')

    img = og.ReadAsArray()
    [a, b, c] = np.shape(img)

    # pull out 2d arrays
    b1 = img[0, :, :]
    b2 = img[1, :, :]
    b3 = img[2, :, :]
    b4 = img[3, :, :]
    b5 = img[4, :, :]
    b6 = img[5, :, :]

    # make dictionary
    temp_d = dict()
    temp_d[0] = np.ravel(b1)
    temp_d[1] = np.ravel(b2)
    temp_d[2] = np.ravel(b3)
    temp_d[3] = np.ravel(b4)

    sample = pd.DataFrame(data=temp_d)
    sample.columns = ['Blue', 'Green', 'Red', 'NIR']

    # create model instance
    model = build_model()

    # load the weights from the checkpoint
    latest = 'checkpoints/cp.ckpt'
    model.load_weights(latest)

    # apply model to make predictions
    predictions = model.predict(sample)
    # reshape vectors back into properly sized arrays
    b5_pred = np.reshape(predictions[:, 0], (b, c))
    b6_pred = np.reshape(predictions[:, 1], (b, c))

    # Now there are the original 'truth' SWIR bands b5 and b6
    # And there are the 'predicted' SWIR bands b5_pred and b6_pred
    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.scatter(np.ravel(b5), np.ravel(b5_pred))
    m_pre = np.max(np.ravel(b5_pred))
    m_tru = np.max(np.ravel(b5))
    lims = [0, np.max([m_pre, m_tru])]
    plt.plot(lims, lims, 'k')
    plt.xlabel('SWIR-1 Truth')
    plt.ylabel('SWIR-1 Predictions')

    plt.subplot(1, 2, 2)
    plt.scatter(np.ravel(b6), np.ravel(b6_pred))
    m_pre = np.max(np.ravel(b6_pred))
    m_tru = np.max(np.ravel(b6))
    lims = [0, np.max([m_pre, m_tru])]
    plt.plot(lims, lims, 'k')
    plt.xlabel('SWIR-2 Truth')
    plt.ylabel('SWIR-2 Predictions')

    plt.tight_layout()
    fig.savefig('ModelPerformance.png')
