# Texture Synthesis in PyTorch

This is a PyTorch implementation of the *Texture Synthesis Using Convolutional Neural Networks* paper (https://arxiv.org/abs/1505.07376) by Gatys et al. (2015). The code is based on the original Caffe codebase by the authors of the paper (https://github.com/leongatys/DeepTextures).

## Example Inputs (*left*) and Outputs (*right*)

![pebbles_in](img/pebbles.jpg) ![pebbles_out](img/output_pebbles.png)

<img src="img/flowers.png" width="256"> <img src="img/output_flowers.png" width="256">

## How can I generate my own texture?

1.  Create a Conda environment using the included [definition file](environment_win_cpu.yml) for Windows + CPU. (Also see [instructions](#how-can-i-run-the-code-on-gpu-or-on-maclinux) for GPU computation and Mac/Linux).
2.  Run `python synthesis.py` for a run with default arguments. This will generate a pebble texture similar to [this one](img/output_pebbles.png).
3.  Run `python synthesis.py -h` to see how you can choose your own input textures or tweak the optimizer settings.

## How can I run the code on GPU, or on Mac/Linux?

Create an empty Conda environment with Python 3.7.7 and install the following dependencies:
* `conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=10.2 -c pytorch` (replace `cudatoolkit=10.2` with `cpuonly` if you want a CPU version)
* `conda install -c conda-forge matplotlib=3.2.2`
* `conda install -c conda-forge scipy=1.5.0`
* *(optional, needed for unit testing)* `conda install -c conda-forge h5py=2.10.0`
* *(optional, needed for unit testing)* `conda install -c conda-forge pytest=5.4.3`

## Can you explain more about the model? What are [`convert_model.py`](convert_model.py) and [`caffemodel2pytorch.py`](caffemodel2pytorch.py) for?

Gatys et. al use pretrained [VGG19](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) with the following modifications:
* Fully connected (FC) layers are removed
* MaxPool layers are replaced by AvgPool
* Weights are scaled, so that the mean activation of each filter over images and positions is equal to 1.

Gatys et al. provide the model in Caffe format and here it is converted to PyTorch via the following steps:
1.  Weigths are converted from `.caffemodel` to `.pt` using [`caffemodel2pytorch.py`](caffemodel2pytorch.py) which comes from [https://github.com/vadimkantorov/caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch).
2.  Pretrained `torchvision.models.vgg19` is loaded, FC layers are removed and MaxPool replaced by AvgPool
3. `.pt` weights are loaded into the PyTorch model.
4. The PyTorch model along with the weights is saved using a custom function into a file called [`VGG19_normalized_avg_pool_pytorch`](models/VGG19_normalized_avg_pool_pytorch).

Dependecies for these conversion scripts are not part of the Conda environment definition file as the converted PyTorch model can be used directly.

## How do you know that the reproduced results are correct?

A set of [unit tests](unit_test.py) is included in the repository. These tests compare values from the converted PyTorch model with those from the original code. Inputs and expected outputs for these tests are stored in a [separate file](data/reference_values.hdf5).

# Disclaimer

Like the original code, this software is published for academic and non-commercial use only.
