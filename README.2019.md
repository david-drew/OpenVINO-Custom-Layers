

## Before You Start
It's assumed that you've installed `OpenVINO 2019.r1`, including the Model Optimizer.  If you're using an earlier version, refer to the other [document](./README.md).

---

Sample code, specifically the `classification_sample`, is located at:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`~/inference_engine_samples/intel64/Release`<br>

The Model Optimizer is abbreviated `MO` for the remainder of this document.

There are 2 directories with C++ and Python source code for the cosh layer. When <b>`r_XX`</b> is used below, substitute `2018.r4`, `2018.r5`, or `2019.r1` as appropriate.

### Custom Layers
Custom layers are NN layers that are not explictly supported by a given framework.  This tutorial demonstrates how to run inference on the topologies featuring custom layers. This way you can plug in your own implementation for existing or completely new layers.

The list of known layers is different for any particular framework. To see the layers supported by your framework, refer to the [OpenVINO MO Documentation](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer#intermediate-representation-notation-catalog).  If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation.

### Example custom layer- Hyperbolic Cosine (cosh) function 
We showcase custom layer implementation using a simple function, hyperbolic cosine (cosh). Mathematically, it is represented as:

![](https://latex.codecogs.com/gif.latex?cosh%28x%29%3D%5Cfrac%7Be%5E%7Bx%7D&plus;e%5E%7B-x%7D%7D%7B2%7D)


---

1. Prep and git clone this repository.<br>
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

2. Setup your environment for OpenVINO.<br>
    ```
    source /opt/intel/openvino/bin/setupvars.sh 
    ```

3. Install prerequisites (code generator for running Python snippets).
    `sudo pip3 install cogapp`

3. Run the MO extension generator and answer questions as appropriate 
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

4. Add Custom (cosh) Python Layers
    1. Copy to the Model Optimizer Ops Directory<br><br>
    ```
    sudo cp ~/cl_tutorial/OpenVINO-Custom-Layers/2019.r1/cosh.py /opt/intel/openvino/deployment_tools/model_optimizer/mo/ops/
    ```

    2. Copy to Extension Generation Python Target Dir<br><br>
    ```
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/2019.r1/cosh_ext.py ~/cl_cosh/user_mo_extensions/ops/
    ```

5. Copy CPU and GPU source code to the M.O. extensions directory<br>
    ```
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/2019.r1/ext_cosh.cpp ~/cl_cosh/user_ie_extensions/cpu/
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/2019.r1/cosh_kernel.cl ~/cl_cosh/user_ie_extensions/gpu/
    ```

6. Create the TensorFlow graph files (weights, graphs, checkpoints)<br>
    `cd ~/cl_tutorial/create_tf_model`<br>
    `./build_cosh_model.py`


7. Convert the TensorFlow model to Intel IR format<br>
    `cd ~/cl_new`<br>
    `mo_tf.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCosh/Activation_8/softmax_output" --extensions ~/cl_cosh/user_mo_extensions --output_dir ~/cl_ext_cosh`<br>


8. Compile the C++ extension library<br>
    ```cd ~/cl_cosh/user_ie_extensions/cpu```<br>
    ```cp ~/cl_tutorial/OpenVINO-Custom-Layers/2019.r1/CMakeLists.txt .```<br>
    ```mkdir build && cd build```<br>
    ```cmake ..```<br>
    ```make -j$(nproc)```<br>
    ```cp libuser_cpu_extension.so ~/cl_ext_cosh/```<br>


9. Test your results<br>
    ```~/inference_engine_samples/intel64/Release/classification_sample -i pics/dog.bmp -m ~/cl_ext_cosh/model.ckpt.xml -d CPU -l ~/cl_ext_cosh/libuser_cpu_extension.so```

10. Programming<br>
    Here are the input dimensions needed for the cosh layer:<br>

    | Dim | Val | Element | Definition         |
    |-----|-----|---------|--------------------|
    | N   | 1   |  0      | Number of Images   |
    | H   | 1   |  2      | Height of Image    |
    | W   | 1   |  3      | Width of Image     |
    | C   | 4   |  1      | Number of Channels |

    <br>

    ```
    N: Number of images
    H: Height of image
    W: Width of image
    C: Number of channels
    ```
    <br>
