 # OpenVINO 2019 R1.1 Custom Layer Tutorial for Linux
This tutorial has been tested and confirmed on Ubuntu 16.04.

## Before You Start
It's assumed that you've installed `OpenVINO 2019 R1.1 for Linux`, including the Model Optimizer, in the default /opt/intel directory. If you're using an earlier version, refer to this [document](./README.md). If you've installed to a different directory you may need to change the directory pathways in the commands below.

If you haven't already, download the Intel® Distribution of OpenVINO™ 2019 R1 toolkit package file from  [Intel® Distribution of OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/openvino-toolkit/choose-download). Select the Intel® Distribution of OpenVINO™ toolkit for Linux package from the dropdown menu.

---

The `classification_sample` code is located here:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`~/inference_engine_samples/intel64/Release`<br>

There are 2 directories with C++ and Python source code for the cosh layer. 

### Custom Layer Outline
 
1. Requirements:
   * OpenVINO
   * cogapp (python lib, install via pip3)
   * Your trained model 
2. Setup the OpenVINO environment.
3. Run the Model Extension Generator (MEG).
   * This creates “code stubs” that will be edited in steps 7 and 8 with the custom algorithm.
4. Edit C++ Code (produced by MEG).
5. Edit Python Scripts (produced by MEG).
6. Workaround for Linux:
   * Move a python custom layer script to the Model Optimizer operations directory:
   * `/opt/intel/openvino/deployment_tools/model_optimizer/mo/ops/`
7. Run the Model Optimizer.
8. Compile your C++ code.
9. Test with Python and/or C++ sample apps.
		
These steps allow the OpenVINO Inference Engine to run the custom layer. The cosh function used in this tutorial allows a simple example of the process.

### Custom Layers
Custom layers are NN (Neural Network) layers that are not explicitly supported by a given framework. This tutorial demonstrates how to run inference on topologies featuring custom layers allowing you to plug in your own implementation for existing or completely new layers.

The list of known layers is different for any particular framework. To see the layers supported by OpenVINO, refer to the OpenVINO Documentation: https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html#intermediate-representation-notation-catalog 
<br><br>

*If your topology contains layers that are not in the list of known layers, the Model Optimizer considers them to be custom.*

The Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model and producing the Intermediate Representation.

### Custom Layers implementation workflow in OpenVINO
When implementing the custom layer in OpenVINO toolkit for your pre-trained model, you will need to add extensions in both the Model Optimizer and the Inference Engine. The following figure shows the work flow for the custom layer implementation. 
<br>

![image of CL workflow](https://raw.githubusercontent.com/david-drew/OpenVINO-Custom-Layers/tree/master/pics/workflow.png "CL Workflow")

<br>

### Example custom layer: Hyperbolic Cosine (cosh) function 
We showcase custom layer implementation using a simple function; hyperbolic cosine (cosh). It's mathematically represented as:

![](https://latex.codecogs.com/gif.latex?cosh%28x%29%3D%5Cfrac%7Be%5E%7Bx%7D&plus;e%5E%7B-x%7D%7D%7B2%7D)

### Extension Generator (MEG)
This tool generates extension source files with stubs for the core functions. To get the workable extension, you will add your implementation of these functions to the generated files.  

### Steps to implement custom layers on Ubuntu 16.04

1. Prep and git clone this repository:<br>
    ```
    cd ~
    ```
    ```
    mkdir cl_tutorial
    ```
    ```
    cd cl_tutorial
    ```
    ```
    git clone https://github.com/david-drew/OpenVINO-Custom-Layers.git
    ```

2. Setup your environment for OpenVINO:<br>
    ```
    source /opt/intel/openvino/bin/setupvars.sh 
    ```

3. Install prerequisites (code generator for running Python snippets):<br>
    ```
    sudo pip3 install cogapp
    ```

4. Run the Model Optimizer extension generator:
   We're using `~/cl_cosh` as the target extension path:<br><br>
   This will create templates that will be partially replaced by Python and C++ code for executing the layer.
    ```
    mkdir /tmp/cl_cosh
    ```
    ```
    python3 /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --mo-tf-ext --mo-op --ie-cpu-ext --output_dir=/tmp/cl_cosh
    ```

   Answer the Model Optimizer extension generator questions as follows:
    ```
    Enter layer name:
    [Cosh]

    Do you want to automatically parse all parameters from the model file? (y/n)
    [y]

    Do you want to change any answer (y/n) ?
    [n]

    Do you want to use the layer name as the operation name? (y/n)
    [y]

    Input all attributes that should be output in IR or needed for shape calculation in format:
    [q]

    Input all internal operation attributes, which will be omitted in IR, in format:
    [q]

    Does your operation change shape? (y/n)
    [n]

    Do you want to change any answer (y/n) ?
    [n]
    ```

   The output will be similar to the following:
     
<br>

![image of extgen output](https://github.com/david-drew/OpenVINO-Custom-Layers/tree/master/pics/extgen_output.png "extgen output")

<br>

5. Move the cl_cosh directory:<br><br>

    ```
    mv /tmp/cl_cosh ~
    ```

6. Add Custom (cosh) Python Layers:
   Copy to the Model Optimizer Ops Directory:<br>
   This allows the Model Optimizer to find the Python implementation of cosh.<br><br>
    ```
    sudo cp ~/cl_tutorial/OpenVINO-Custom-Layers/2019.r1/cosh.py /opt/intel/openvino/deployment_tools/model_optimizer/mo/ops/
    ```

    Rejmove an erroneous cosh extension file:<br><br>
    ```
    rm ~/cl_cosh/user_mo_extensions/ops/cosh_ext.py
    ```

7. Copy CPU and GPU source code to the Model Optimizer extensions directory:<br>
   This will be used for building a back-end library for applications that implement cosh.<br><br>
    ```
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/2019.r1/ext_cosh.cpp ~/cl_cosh/user_ie_extensions/cpu/
    ```

8. Create the TensorFlow model (weights, graphs, checkpoints):<br>
   We create a simple model. The weights are random and untrained, but sufficient for demonstrating Custom Layer conversion.<br><br>
    ```
    cd ~/cl_tutorial/OpenVINO-Custom-Layers/create_tf_model
    ```
    ```
    ./build_cosh_model.py
	```

9. Convert the TensorFlow model to Intel IR format:<br>
   We run the Model Optimizer for TensorFlow to convert and optimize the new model for OpenVINO. We explicitly set the batch to 1 because the model has an input dim of "-1". TensorFLow allows "-1" as a variable indicating "to be filled in later", but the Model Optimizer requires explicit information for the optimization process. The output is the full name of the final output layer.<br><br>
	```
    cd ~/cl_new
	```
	```
    mo_tf.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCosh/Activation_8/softmax_output" --extensions ~/cl_cosh/user_mo_extensions --output_dir ~/cl_ext_cosh
	```

10. Compile the C++ extension library:<br>
   Here we're building the back-end C++ library to be used by the Inference Engine for executing the cosh layer.<br><br>
    ```
	cd ~/cl_cosh/user_ie_extensions/cpu
    ```
    ```
	cp ~/cl_tutorial/OpenVINO-Custom-Layers/2019.r1/CMakeLists.txt .
    ```
    ```
	mkdir -p build && cd build
    ```
    ```
	cmake ..
    ```
    ```
	make -j$(nproc)
    ```
    <br>
    
    ```
	cp libcosh_cpu_extension.so ~/cl_ext_cosh/
    ```

11. Test your results:<br>
    
    <b>Using a C++ Sample:</b><br>
    ```
    ~/inference_engine_samples_build/intel64/Release/classification_sample -i ~/cl_tutorial/OpenVINO-Custom-Layers/pics/dog.bmp -m ~/cl_ext_cosh/model.ckpt.xml -d CPU -l ~/cl_ext_cosh/libcosh_cpu_extension.so
    ```
    <br><b>Using a Python Sample:</b><br>
    
    Prep: Install the OpenCV library and copy an appropriate sample to your home directory for ease of use:<br>
    ```
    sudo pip3 install opencv-python
    ```
    ```
    cp 
    /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample/classification_sample.py .
    ```
   
    <br>Try running the Python Sample without including the cosh extension library:<br>
    ```
    python3 classification_sample.py -i ~/cl_tutorial/OpenVINO-Custom-Layers/pics/dog.bmp -m ~/cl_ext_cosh/model.ckpt.xml -d CPU
    ```
    
    <br>Now run the command with the cosh extension library:<br>
    ```
    python3 classification_sample.py -i ~/cl_tutorial/OpenVINO-Custom-Layers/pics/dog.bmp -m ~/cl_ext_cosh/model.ckpt.xml -l ~/cl_ext_cosh/libcosh_cpu_extension.so -d CPU
    ```

Thank you for following this tutorial. Your feedback answering this brief survey will help us to improve it:
[Intel Custom Layer Survey](https://intelemployee.az1.qualtrics.com/jfe/form/SV_1ZjOKaEIQUM5FpX)



