# Code to sample training data from a set of geotifs
# Expect to read geotifs with 6 bands: Blue, Green, Red, NIR, SWIR-1, SWIR-2

# This code randomly samples the set of geotifs and then saves data reshaped as 1-D vectors in a csv that will be used for the tensorflow workflow

import numpy as np
import os
import pandas as pd
import gdal

def get_files(file_dir):
    """
    Create list of all the geotif files that will be sampled to make the training set

    Parameters
    ----------

    file_dir : `str`
        Path to the directory holding all of the geotifs to be sampled

    Returns
    -------

    f_names : `list`
        List of the individual files contained in the directory with their full pathing
    """

    f_names = []

    # check if file_dir provided with or without a forward slash as the final value -- if not there then append one
    if file_dir[-1] != '/':
        file_dir = file_dir + '/'

    for i in os.listdir(file_dir):
        f_names.append(file_dir + i)

    return f_names


def pull_bands(img,band_ind,x_vals,y_vals):
    """
    Pull a single (x,y) data point for a specified band

    Parameters
    ----------

    img : `numpy.ndarray`
        Image read as a numpy array

    band_ind : `int`
        Integer value corresponding to the band to pull data for

    x_vals : `list`
        Integer list of x values to query array

    y_vals : `list`
        Integer list of y values to query array

    Returns
    -------

    val : `list`
        List of values acquired from array
    """

    mdata = map(lambda x, y: img[band_ind,x,y], x_vals, y_vals)
    val = list(mdata)

    return val


def get_sample(file_name,num_pts=500):
    """
    Get a sample of values from a specified file

    Parameters
    ----------

    file_name : `str`
        Path to a specific file

    num_pts : `int`
        Number of samples to collect, default is 500

    Returns
    -------

    sample : `pandas dataframe`
        Collected sample data as a pandas dataframe
    """

    # open the image
    og = gdal.Open(file_name)
    img = og.ReadAsArray()

    # check shape of the read image
    [a,b,c] = np.shape(img)
    if a != 6:
        raise ValueError('Unexpected number of bands')

    # define values to grab
    x_vals = np.random.randint(0,b,num_pts)
    y_vals = np.random.randint(0,c,num_pts)

    # get values and put into a dataframe
    temp_d = dict()
    for i in range(0,6):
        temp_d[i] = pull_bands(img,i,x_vals,y_vals)

    sample = pd.DataFrame(data=temp_d)
    sample.columns = ['Blue','Green','Red','NIR','SWIR1','SWIR2']

    return sample


def create_training(file_dir,num_pts=500):
    """
    Get samples from all of the images to create the training dataset.

    Parameters
    ----------

    file_dir : `str`
        Path to the directory holding all of the geotifs to be sampled

    num_pts : `int`
        Number of samples to collect per image, default is 500


    Returns
    -------

    Saves a file called training_data.csv to the disk.
    """

    # init dataframe to hold all data
    df = pd.DataFrame(columns=['Blue','Green','Red','NIR','SWIR1','SWIR2'])

    # get list of file names
    f_names = get_files(file_dir)

    # loop and populate dataframe
    for i in f_names:
        ind_sample = get_sample(i,num_pts)
        df = pd.concat([df,ind_sample], ignore_index=True)

    # save to disk
    df.to_csv('training_data.csv')
