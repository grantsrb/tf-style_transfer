# Style Transfer
### May 28th, 2017
## Satchel Grant


## Overview
This notebook serves as a walk through of a TensorFlow implementation of style transfer.

Style transfer is the process of blending the style of a image with the form of another image. For example the left and right images are style transferred to generate the middle image.

![image1](./imgs/satch_gogh_combined.jpg)
![image2](./imgs/abbas_starry_blend.jpg)
![image3](./imgs/benj_picasso_combined.jpg)

## Project Navigation

Launch the [style_transfer](./style_transfer.ipynb) notebook or view the [style_transfer.html](./style_transfer.html) file for a project walk through that discusses the concepts and code.

To launch the notebook, first install the system requirements (listed in the next section). Next, open terminal and clone this repository. Then navigate to the project folder:

```
$ git clone https://github.com/grantsrb/Image_Style_Transfer
$ cd Image_Style_Transfer
```

Next, type these commands in the command line:

```
$ jupyter notebook style_transfer.ipynb
```

Otherwise run the `style_transfer.py` script.

```
$ python3 style_transfer.py
```


## Setup and Installation

#### Requirements

The required technologies for this project to run are:

- Python 3.5
- NumPy
- TensorFlow
- MatPlotLib
- SciPy
- PIL

You must download the `vgg16.npy` file for the Vgg16 network to work. [Download here](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM). Save to the `utils` directory within this repository.

#### Miniconda Environment

These instructions assume basic knowledge of terminal and bash.

The required packages for this project are listed in the environment files located in the [environment directory](./environments/).

The easiest way to set up the required packages is using `miniconda`.

First install [`miniconda`](https://conda.io/miniconda.html) onto your computer using the python 3.6 version.

Then clone this repository:

```
$ git clone https://github.com/grantsrb/Image_Style_Transfer
```

Navigate to the main repo directory:

```
$ cd Image_Style_Transfer
```

Make a new `conda` environment using the environments/environment.yml file:

```
$ conda env create -f environments/environment.yml
```

Verify that the environment was created:

```
$ conda info --envs
```

Clean up residual files:

```
$ conda clean -tp
```

While the `conda` environment is active, launch Jupyter Notebook by typing:

```
$ jupyter notebook
```

And then select the `style_transfer.ipynb` file.

#### Uninstallation

If you want to uninstall the environment use:

```
$ conda env remove -n foo_cpu
```

Now that the environment exists, you must activate it using:

```
$ source activate foo_cpu
```

This runs the conda environment so that you can use the required packages. When you're finished deactivate the environment using:

```
source deactivate
```
