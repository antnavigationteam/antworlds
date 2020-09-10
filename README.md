# Background
## Purpose
This package can be used to simulate insect vision in 3D virtual environments. 

## Source material
This software package was derived from Titus Neuman's paper [1]. This concept was previously developed by Andrew Straw as 
detailed [here](https://strawlab.org/2011/03/23/grand-unified-fly/) and published in [2]. This repository constitutes a 
lightweight python 3 implementation that can be easily installed and does not depend on the Coin 3D environment. 
Furthermore, a geometry shader, OpenGL based cubemap rendering pipeline has been developed from scratch using GLSL scripts.
We have found this approach renders very large meshes deposited [here](https://insectvision.dlr.de/3d-reconstruction-tools/habitat3d)
in real time whereas previous implementations cannot achieve this. 
This project was initially developed in python 2.7 by Jan Stankiewicz to support his MSc thesis work. The code base has subsequently 
been refactored by Florent Le Moël to support Python 3 and to tidy and prune the code base.

1. Neumann T.R. (2002) Modeling Insect Compound Eyes: Space-Variant Spherical Vision. In: Bülthoff H.H., Wallraven C., Lee SW., Poggio T.A. (eds) Biologically Motivated Computer Vision. BMCV 2002. Lecture Notes in Computer Science, vol 2525. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-36181-2_36
2. William B. Dickson, Andrew D. Straw, Michael H. Dickinson (2008) Integrative Model of Drosophila Flight. AIAA Journal 46 (9). doi:10.2514/1.29862

# Getting started
##Hardware requirements

This repo currently only runs on systems that support OpenGL v4.5 or greater. 

##Installation
We currently recommend using [pipenv](https://github.com/pypa/pipenv) to handle the installation process.

The project is designed to run on python3. If not already installed go [here](https://www.python.org/downloads/) to get the correct version for your operating system.

Make sure pipenv is installed in your python 3 distribution:
 
```$ pip3 install pipenv``` 

Clone this github repository in a desirable location:
 
```<project_folder>$ git clone https://github.com/antnavigationteam/antworlds```

Install the project dependencies (could take a while):
 
 ```$ pipenv install --python 3```   
 
## Testing the installation
 
## Performance tips
 
* Ensure that numpy is installed with BLAS or mkl support according to your processor


# Future development priorities 
* Optimise insect eye rendering speed by moving matrix operations to the GPU (requires OpenCL integration OR new program shaders) 

# Installation notes
N.B. (Jan) - installing pyopengl-accelerate seems to cause problems at the moment, do not install this in the pyeye environment
Making better use of the pyOpengl methods (e.g. glbuffer objects) may resolve this.

