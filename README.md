
# OpenVINO Advanced Custom Layer Tutorial

This is an advanced tutorial that teaches how to use custom layers in OpenVINO.

The earlier tutorial demonstrated the process of converting and running customer 
layers in the OpenVINO Inference Engine, but used a simple hyperbolic cosine function  
as the custom layer.  The cosh algorithm was chosen for simplicity of conversion, and didn't
require additional parameters to be provided to the Model Optimizer.

This tutorial walks through using a practical real-world example: the argmax function. 


## Directory Layout  
Model Optimizer Custom Layer Extensions are located in this directory tree:
  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/


There are 2 subdirectories:

  front:
    Python scripts that are used for extracting layer attributes are located here. 
    The scripts are organized into subdirectories for different frameworks: Caffe, TensorFlow, MXNet, ONNX.

  ops:
    Python scripts that tell IE how to calculate the output shape (during inference) are located here. 


# Example Custom Layer: The ArgMax Function

We will showcase the steps involved for implementing a custom layer using the *argmax* (arguments of the maxima) function.  The equation for *argmax* is:

![](pics/argmax.png)

Argmax is a layer used in some Deep Learning models.  It calculates the input value that delivers the maximum output of a function.


# Getting Started

## Setting Up the Environment

To begin, always ensure that your environment is properly setup for working with the Intel® Distribution of OpenVINO™ toolkit by running the command:

```bash
source /opt/intel/openvino/bin/setupvars.sh
```

## Installing Prerequisites

1. The Model Extension Generator makes use of *Cog* which is a content generator allowing the execution of embedded Python code to generate code within source files.  Install *Cog* (*cogapp*) using the command:

   ```bash
   sudo pip3 install cogapp
   ```

2. This tutorial will be running a Python sample from the Intel® Distribution of OpenVINO™ toolkit which needs the OpenCV library for Python to be installed.  Install the OpenCV library using the command:

   ```bash
   sudo pip3 install opencv-python
   ```

## Downloading and Setting Up the Tutorial

The first things we need to do are to create a place for the tutorial and then download it.  We will create the top directory "cl_tutorial" as the workspace to store the Git repository of the tutorial along with all the other files created.

1. Create the "cl_tutorial" top directory in the user's home directory and then change into it:
    ```bash
    cd ~
    mkdir cl_tutorial
    cd cl_tutorial
    ```
2. Download the tutorial by cloning the repository:
    ```bash
    git clone https://github.com/david-drew/cl-adv.git
    ```
3. Create some environment variables as shorter, more convenient names to the directories that will be used often:
    ```bash
    export CLWS=~/cl_adv_tutorial
    export CLT=$CLWS/2019.r1.1

    From here, we will now use "$CLWS" to reference the "cl_tutorial" workspace directory and "$CLT" to reference the directory containing the files for this tutorial.
    ```

## Download the Example Segmentation Model:

We'll download a segmentation model, then later we'll convert the model to an Intel-compatible format.

   ```bash
    mkdir $CLWS/maskrcnn
    cd $CLWS/maskrcnn
    https://github.com/matterport/Mask_RCNN/releases
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5
    wget https://github.com/matterport/Mask_RCNN/archive/v2.1.tar.gz

   ```

# Creating the *argmax* Custom Layer



## Generate the Extension Template Files Using the Model Extension Generator

We will use the Model Extension Generator tool to automatically create templates for all the extensions that will be needed by the Model Optimizer to convert and the Inference Engine to execute the custom layer.  The extension template files will be partially replaced by Python and C++ code to implement the functionality of *argmax* as needed by the different tools.  To create the four extensions for the *argmax* custom layer, we run the Model Extension Generator with the following options:

- --mo-tf-ext = Generate a template for a Model Optimizer TensorFlow extractor
- --mo-op = Generate a template for a Model Optimizer custom layer operation
- --ie-cpu-ext = Generate a template for an Inference Engine CPU extension
- --ie-gpu-ext = Generate a template for an Inference Engine GPU extension
- --output_dir = set the output directory.  Here we are using *$CLWS/argmax* as the target directory to store the output from the Model Extension Generator.

To create the four extension templates for the *argmax* custom layer, we run the command:

```bash
python3 /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --mo-tf-ext --mo-op --ie-cpu-ext --ie-gpu-ext --output_dir=$CLWS/argmax
```

The Model Extension Generator will start in interactive mode and prompt the user with questions about the custom layer to be generated.  Use the text between the []'s to answer each of the Model Extension Generator questions as follows:

```
Enter layer name:
[argmax]

Do you want to automatically parse all parameters from the model file? (y/n)
...
[y]

Do you want to change any answer (y/n) ? Default 'no'
[n]

Do you want to use the layer name as the operation name? (y/n)
[y]

Does your operation change shape? (y/n)
[y]

## INCOMPLETE ##

Do you want to change any answer (y/n) ? Default 'no'
[n]
```

## Edit the "front" template files so MO will know how to extract *argmax* attributes 

## Edit the "ops" template files so IE will know the output shape of the *argmax* layer during inference 

## Compile a C++ library for IE to use for calculating the *argmax* values

## Convert and Optimize an Instance Segmentation NN Topology


## Run the sample code
 

# Reference

[Custom Layers in the Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html)

[Custom Layers Support in Inference Engine](https://software.intel.com/en-us/articles/OpenVINO-Custom-Layers-Support-in-Inference-Engine)

[Mask R-CNN by Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick](https://arxiv.org/abs/1703.06870)



