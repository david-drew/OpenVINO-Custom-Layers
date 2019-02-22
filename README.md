

## Before You Start
It's assumed that you've installed `OpenVINO r5.01`, including the Model Optimizer.  

Sample code, specifically the `classification_sample`, is located at:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`~/inference_engine_samples/intel64/Release`<br>

The Model Optimizer is abbreviated `MO` for the remainder of this document.

There are 2 directories with C++ and Python source code for the cosh layer. When <b>`r_XX`</b> is used below, substitute `2018.r4`, `2018.r5`, or `2019.r1` as appropriate.

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
    * For 2019.r1 and later, use this:
    ```
    source /opt/intel/openvino/bin/setupvars.sh 
    ```
3. Install prerequisites.
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
    sudo cp ~/cl_tutorial/OpenVINO-Custom-Layers/r_XX/cosh.py /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo/ops/
    ```

    2. Copy to Extension Generation Python Target Dir<br><br>
    ```
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/r_XX/cosh_ext.py ~/cl_cosh/user_mo_extensions/ops/cosh_ext.py
    ```

5. Copy CPU and GPU source code to the M.O. extensions directory<br><br>
    ```
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/r_XX/ext_cosh.cpp ~/cl_cosh/user_ie_extensions/cpu/
    cp ~/cl_tutorial/OpenVINO-Custom-Layers/r_XX/cosh.cl ~/cl_cosh/user_ie_extensions/gpu/
    ```

7. Create the TensorFlow graph files (weights, graphs, checkpoints)<br><br>
    ```
    cd ~/cl_tutorial/create_tf_model
    ```
    ```
    ./build_cosh_model.py
    ```

8. Convert the TensorFlow model to Intel IR format<br><br>
    ```
    mo_tf.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCosh/Activation_8/softmax_output" --extensions ~/cl_cosh/user_mo_extensions --output_dir ~/cl_ext_cosh
    ```

9. Compile the C++ extension library<br><br>
    ```cd ~/cl_cosh/user_ie_extensions/cpu```<br>
    * If using 2019.r1, copy the CMakeLists.txt to this directory.
    ```cp ~/cl_tutorial/OpenVINO-Custom-Layers/r_XX/CMakeLists.txt .```<br>
    ```mkdir build && cd build```<br>
    ```cmake ..```<br>
    ```make -j$(nproc)```<br>
    ```cp libuser_cpu_extension.so ~/cl_ext_cosh/```<br>

10. Test your results<br><br>
    ```
    ~/inference_engine_samples/intel64/Release/classification_sample -i pics/dog.bmp -m ~/cl_ext_cosh/model.ckpt.xml -d CPU -l ~/cl_ext_cosh/libuser_cpu_extension.so 
    ```

