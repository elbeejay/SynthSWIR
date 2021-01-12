"""Making alternative fake bands to replace the SWIR-1 and SWIR-2 bands."""
import numpy as np
import gdal


def make_alt_band(file_name, color='white'):
    """Instead of creating synthetic SWIR bands just make a dummy one.

    Parameters
    ----------
    file_name : `str`
        Path to a specific multi-band image file

    color : `str`, Optional
        String for color to be used for the dummy bands. Default is 'white',
        other options are 'black' and 'noisy' for a band of white noise.

    Returns
    -------
    Saves a new file to disk with the same name as `file_name` but with
    the string input from `color` appended to the end of the file name.

    """
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
    max_bvals = [np.max(b1), np.max(b2), np.max(b3), np.max(b4)]

    # make the dummy arrays (0-255 in 8-bit color scheme)
    if np.max(max_bvals) < 256:
        if color == 'white':
            b5 = np.zeros_like(b1, dtype='uint8') + 255
            b6 = np.zeros_like(b1, dtype='uint8') + 255
        elif color == 'black':
            b5 = np.zeros_like(b1, dtype='uint8')
            b6 = np.zeros_like(b1, dtype='uint8')
        else:
            b5 = np.random.randint(0, 256, size=b1.shape, dtype='uint8')
            b6 = np.random.randint(0, 256, size=b1.shape, dtype='uint8')

        # create raster file to save 'prediction' geotif
        rs = gdal.GetDriverByName('GTiff').Create(file_name + '_' + color +
                                                  '.tif',
                                                  c, b, 6, gdal.GDT_Byte)

    else:
        # else it is 16-bit so range is 0-65536
        if color == 'white':
            b5 = np.zeros_like(b1, dtype='uint16') + 65536
            b6 = np.zeros_like(b1, dtype='uint16') + 65535
        elif color == 'black':
            b5 = np.zeros_like(b1, dtype='uint16')
            b6 = np.zeros_like(b1, dtype='uint16')
        else:
            b5 = np.random.randint(0, 65536, size=b1.shape, dtype='uint16')
            b6 = np.random.randint(0, 65536, size=b1.shape, dtype='uint16')

        # create raster file to save 'prediction' geotif
        rs = gdal.GetDriverByName('GTiff').Create(file_name + '_' + color +
                                                  '.tif',
                                                  c, b, 6, gdal.GDT_UInt16)

    # apply original geo-reference to the new 'prediction' geotif
    geotransform = og.GetGeoTransform()
    rs.SetGeoTransform(geotransform)

    # apply original projection to the new 'prediction' geotif
    geoproj = og.GetProjection()
    rs.SetProjection(geoproj)

    # write all of the bands to the new geotif, bands 5 and 6 are the predicted
    # synthetic SWIR bands from the model results
    rs.GetRasterBand(1).WriteArray(b1)
    rs.GetRasterBand(2).WriteArray(b2)
    rs.GetRasterBand(3).WriteArray(b3)
    rs.GetRasterBand(4).WriteArray(b4)
    rs.GetRasterBand(5).WriteArray(b5)
    rs.GetRasterBand(6).WriteArray(b6)
    rs.FlushCache()
    rs = None


def make_dup_bands(file_name, color='nir'):
    """Instead of creating synthetic SWIR bands, recycle an existing band.

    Parameters
    ----------
    file_name : `str`
        Path to a specific multi-band image file

    band : `str`
        Either 'red', 'blue', 'green' or 'nir' (default) to specify the
        band that is to be recycled

    Returns
    -------
    Saves a new file to disk with the same name as `file_name` but with
    the string input from `band` appended to the end of the file name.

    """
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
    max_bvals = [np.max(b1), np.max(b2), np.max(b3), np.max(b4)]

    # make the dummy arrays (0-255 in 8-bit color scheme)
    if np.max(max_bvals) < 256:
        if color == 'red':
            b5 = b3
            b6 = b3
        elif color == 'green':
            b5 = b2
            b6 = b2
        elif color == 'blue':
            b5 = b1
            b6 = b1
        else:
            b5 = b4
            b6 = b4

        # create raster file to save 'prediction' geotif
        rs = gdal.GetDriverByName('GTiff').Create(file_name + '_' + color +
                                                  '.tif',
                                                  c, b, 6, gdal.GDT_Byte)

    else:
        # else it is 16-bit so range is 0-65536
        if color == 'red':
            b5 = b3
            b6 = b3
        elif color == 'green':
            b5 = b2
            b6 = b2
        elif color == 'blue':
            b5 = b1
            b6 = b1
        else:
            b5 = b4
            b6 = b4

        # create raster file to save 'prediction' geotif
        rs = gdal.GetDriverByName('GTiff').Create(file_name + '_' + color +
                                                  '.tif',
                                                  c, b, 6, gdal.GDT_UInt16)

    # apply original geo-reference to the new 'prediction' geotif
    geotransform = og.GetGeoTransform()
    rs.SetGeoTransform(geotransform)

    # apply original projection to the new 'prediction' geotif
    geoproj = og.GetProjection()
    rs.SetProjection(geoproj)

    # write all of the bands to the new geotif, bands 5 and 6 are the predicted
    # synthetic SWIR bands from the model results
    rs.GetRasterBand(1).WriteArray(b1)
    rs.GetRasterBand(2).WriteArray(b2)
    rs.GetRasterBand(3).WriteArray(b3)
    rs.GetRasterBand(4).WriteArray(b4)
    rs.GetRasterBand(5).WriteArray(b5)
    rs.GetRasterBand(6).WriteArray(b6)
    rs.FlushCache()
    rs = None
