

## Before You Start
It's assumed that you've installed `OpenVINO r4 or r5`, including the Model Optimizer.  For 2019.r1, see the other [document](./README.2019.md).

Sample code, specifically the `classification_sample`, is located at:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`~/inference_engine_samples/intel64/Release`<br>

The Model Optimizer is abbreviated `MO` for the remainder of this document.

There are 2 directories with C++ and Python source code for the cosh layer. When <b>`r_XX`</b> is used below, substitute `2018.r4`, `2018.r5`, or `2019.r1` as appropriate.

### Custom Layers
Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation.

The list of known layers is different for each of supported frameworks. To see the layers supported by your framework, refer to the OpenVINO website. 
Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

This tutorial demonstrates how to run the inference on the topologies featuring custom layers. This way you can plug your own implementation for existing or completely new layers.

the basic steps are:
Custom Layer Outline
 
1.      Have your trained model ready.
2.      Git clone tutorial
3.      Setup environment
 
4.      Requirements:
a.      OpenVINO
b.      cogapp (python lib, install via pip3)
c.      The trained model you want to convert to OpenVINO
 
5.      Setup OpenVINO environment
6.      Run Model Extension Generator (tool for Model Optimizer)
a.      This creates “code stubs” that will be edited in steps 7 and 8 with the custom algorithm.
7.      Edit C++ Code (produced by MEG)
8.      Edit Python Scripts (produced by MEG)
9.      Workaround for Linux:
a.      Move a python custom layer script to the Model Optimizer operations directory:
b.  /opt/intel/openvino/deployment_tools/model_optimizer/mo/ops/
 
10.   Run the Model Optimizer
11.   Compile your C++ code.
12.   Test with Python and/or C++ sample apps.

### Example custom layer- Hyperbolic Cosine (cosh) function 
We showcase custom layer implementation using a simple function, hyperbolic cosine (cosh). Mathematically, it is represented as: 

![](https://latex.codecogs.com/gif.latex?cosh%28x%29%3D%5Cfrac%7Be%5E%7Bx%7D&plus;e%5E%7B-x%7D%7D%7B2%7D)

---

1. Prep and git clone this repository.<br><br>
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
2. Setup your environment for OpenVINO.<br><br>
    * For 2018.r4 and 2018.r5, use this:
    ```
    source /opt/intel/computer_vision_sdk/bin/setupvars.sh 
    ```

3. Install prerequisites.
    `sudo pip3 install cogapp`

4. Run the MO extension generator and answer questions as appropriate 
    * We're using `~/cl_cosh` as the target extension path<br><br>
    ```
    python3 /opt/intel/openvino/deployment_tools/extension_generator/extgen.py new mo-op ie-cpu-ext output_dir=~/cl_cosh
    ```
    * Answer questions as follows:
    ```
    1.  Is your layer Pythonic (y/n)?       y
    2.  Please enter operation name:        cosh
    3.  Does your operation change shape?   n
    Please enter operation name:            cosh
    Please enter all parameters in format
    <param1> <type>                         q
    ```

5. Add Custom (cosh) Python Layers<br>
    1. Copy to the Model Optimizer Ops Directory<br><br>
    ```
    sudo cp ~/cl_tutorial/OpenVINO-Custom-Layers/r_XX/cosh.py /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo/ops/
    ```

    2. Copy to Extension Generation Python Target Dir<br><br>
    ```
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/r_XX/cosh_ext.py ~/cl_cosh/user_mo_extensions/ops/cosh_ext.py
    ```


6. Copy CPU and GPU source code to the M.O. extensions directory<br>
    ```
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/r_XX/ext_cosh.cpp ~/cl_cosh/user_ie_extensions/cpu/
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/r_XX/cosh.cl ~/cl_cosh/user_ie_extensions/gpu/
    ```


7. Create the TensorFlow graph files (weights, graphs, checkpoints)<br>
    ```
    cd ~/cl_tutorial/create_tf_model
    ```
    ```
    ./build_cosh_model.py
    ```


8. Convert the TensorFlow model to Intel IR format<br>
    ```
    mo_tf.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCosh/Activation_8/softmax_output" --extensions ~/cl_cosh/user_mo_extensions --output_dir ~/cl_ext_cosh
    ```


9. Compile the C++ extension library<br>
    ```cd ~/cl_cosh/user_ie_extensions/cpu```<br>
    ```mkdir build && cd build```<br>
    ```cmake ..```<br>
    ```make -j$(nproc)```<br>
    ```cp libuser_cpu_extension.so ~/cl_ext_cosh/```<br>


10. Test your results<br>
    ```
    ~/inference_engine_samples/intel64/Release/classification_sample -i pics/dog.bmp -m ~/cl_ext_cosh/model.ckpt.xml -d CPU -l ~/cl_ext_cosh/libuser_cpu_extension.so 
    ```

