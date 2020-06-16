"""Function that applies trained ML model to PlanetScope data."""
import argparse
import numpy as np
import pandas as pd
import gdal
from simple_model import build_model


def predict(file_name):
    """Run ML prediction on PlanetScope data."""
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

    # make dictionary
    temp_d = dict()
    temp_d[0] = np.ravel(b1)
    temp_d[1] = np.ravel(b2)
    temp_d[2] = np.ravel(b3)
    temp_d[3] = np.ravel(b4)

    # planet data is 16 bit and landsat is 8 bit so do conversion
    for i in range(0, len(temp_d.keys())):
        temp_d[i] = (temp_d[i]/256).astype('uint8')

    sample = pd.DataFrame(data=temp_d)
    sample.columns = ['Blue', 'Green', 'Red', 'NIR']

    # create model instance
    model = build_model()

    # load the weights from the checkpoint
    latest = 'checkpoints/cp.ckpt'
    model.load_weights(latest)

    print('applying model')
    # apply model to make predictions
    predictions = model.predict(sample)

    # reshape vectors back into properly sized arrays
    b1_pred = np.reshape(temp_d[0], (b, c))
    b2_pred = np.reshape(temp_d[1], (b, c))
    b3_pred = np.reshape(temp_d[2], (b, c))
    b4_pred = np.reshape(temp_d[3], (b, c))
    b5_pred = np.reshape(predictions[:, 0], (b, c))
    b6_pred = np.reshape(predictions[:, 1], (b, c))

    # create raster file to save 'prediction' geotif
    rs = gdal.GetDriverByName('GTiff').Create(file_name + '_predicted.tif',
                                              c, b, 6, gdal.GDT_Byte)

    # apply original geo-reference to the new 'prediction' geotif
    geotransform = og.GetGeoTransform()
    rs.SetGeoTransform(geotransform)

    # apply original projection to the new 'prediction' geotif
    geoproj = og.GetProjection()
    rs.SetProjection(geoproj)

    # write all of the bands to the new geotif, bands 5 and 6 are the predicted
    # synthetic SWIR bands from the model results
    rs.GetRasterBand(1).WriteArray(b1_pred)
    rs.GetRasterBand(2).WriteArray(b2_pred)
    rs.GetRasterBand(3).WriteArray(b3_pred)
    rs.GetRasterBand(4).WriteArray(b4_pred)
    rs.GetRasterBand(5).WriteArray(b5_pred)
    rs.GetRasterBand(6).WriteArray(b6_pred)
    rs.FlushCache()
    rs = None


# add ability to call function directly from command-line
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str,
                        help="Path to file")
    args = parser.parse_args()
    predict(args.file_name)
