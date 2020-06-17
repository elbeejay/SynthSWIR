# SynthSWIR
Estimating SWIR-1 and SWIR-2 bands from the RGB and NIR bands using a regression model trained pixel by pixel.

<p align="center">
<img src="https://github.com/elbeejay/SynthSWIR/blob/master/model_info/WorkFlow.png" alt="SynthSWIR Workflow" width="750"/>
</p>

SynthSWIR is an auxiliary Tensorflow 2 model designed to pre-process [PlanetScope](https://www.planet.com/) and other high resolution satellite data without shortwave infrared (SWIR) so that they may be used with the deep convolutional neural networks [DeepWaterMap](https://github.com/isikdogan/deepwatermap) and [DeepRiver](https://github.com/isikdogan/deepriver). With SynthSWIR, these powerful neural networks can be extended to higher resolution data that lacks the Landsat SWIR-1 and SWIR-2 bands.

As an auxiliary model, SynthSWIR attempts to approximate the SWIR-1 and SWIR-2 bands present in LANDSAT data. **Do not use this model if you are interested in high fidelity predictions of SWIR-1 and SWIR-2 bands**. This model was designed to help augment RGB+NIR satellite data so that it can be used with the DeepWaterMap/DeepRiver neural networks. As such, the synthetic SWIR data that is predicted by SynthSWIR is not meant to be perfect, instead the robust design and training (band mixing, random noise) of the DeepWaterMap/DeepRiver models is relied upon to overcome errors in the synthetic SWIR data. Due to their large training datasets (1 TB+) and robust traning methods, it is computationally costly to attempt to re-train DeepWaterMap/DeepRiver with fewer input bands. The advantage of using SynthSWIR, is that the pre-trained DeepWaterMap/DeepRiver models can be applied directly to 4-band (RGB+NIR) data without re-training.

## Contents
  - [Dependencies](#dependencies)
  - [Quick Start](#quick-start)
  - [Model Architecture](#model-architecture)
  - [Pre-Trained Model](#pre-trained-model)
  - [Training On Your Own Data](#training-on-your-own-data)

## Dependencies
*Tested on Python 3.8.2*
  - tensorflow 2+
  - numpy
  - pandas
  - gdal

## Quick Start
From the command line:

  1. Clone the repository: `git clone https://github.com/elbeejay/SynthSWIR`

  2. To apply the model and prediction synthetic SWIR band data for a Landsat GeoTIF, run:
  ```
  python apply_model.py --file_name 'your_landsat_img.tif'
  ```

  3. To apply the model to a PlanetScope GeoTIF, run:
  ```
  python apply_model_planet.py --file_name 'your_planetscope_img.tif'
  ```

The output GeoTIF from the model will be in the same directory as the input image with the suffix *"_predicted"*. 

## Model Architecture
SynthSWIR employs a relatively simply methodology and attempts to learn the relationship between RGB+NIR data and the SWIR-1 and SWIR-2 band data on a pixel-by-pixel basis. The model itself consists of 4 layers as shown below.

**Add image of model architecture**

Model performance and reduction in loss as training occurs can be seen in the [model_info](model_info) subdirectory. Other model architectures that were tested and their associated error metrics are provided there as well.

## Pre-Trained Model
A pre-trained model and checkpoint have been created using primarily coastal river data. The training data used for this is available in the [training_data](training_data) subdirectory. These pixel values provided in the `.csv` file were extracted from a set of Landsat geoTIFs (available [here](https://utexas.box.com/s/t67iptubwdpvyims0afutiipv8qqrqg5)).

## Training On Your Own Data
To train the model on a different set of Landsat data (highly recommended if the intended use is not for coastal river systems), then do the following.

  1. Collect your Landsat GeoTIFs into a single folder 

  2. Run the function `gen_data.create_training(file_dir)` contained in the `gen_data.py` script to create a new `training_data.csv` file with a selection of pixels from your training GeoTIFs. The number of pixels sampled from each GeoTIF in the training data can be specified using the `num_pts` argument. 

  3. Run `trainer.py` to train the model on your newly generated `training_data.csv`. Check the script and your working directory to ensure that the `training_data.csv` file being pointed to is the new one you just generated.

  4. Now you have a newly trained model and a checkpoint should have been saved in the `/checkpoints` directory. When SynthSWIR is next applied via `apply_model.py` or `apply_model_planet.py` it will be using the model trained on your training dataset!

## Misc/Other Scripts
The relationship between the SWIR bands and the RGB+NIR bands can be seen using `visualizing_data.py`. This script uses the seaborn package to produce pair-wise plots showing the relationship between each RGB+NIR band and the SWIR-1 and SWIR-2 bands. 

To see how well the model does in your area of interest, the `test_model.py` script may be helpful. The function `test_model.predict_comparison(file_name)` takes a Landsat GeoTIF as input, applies the SynthSWIR model, and creates scatter plots of the predcted SWIR values against the true Landsat SWIR values.
