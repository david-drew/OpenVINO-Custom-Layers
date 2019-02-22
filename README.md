

## Before You Start
It's assumed that you've installed `OpenVINO r5.01`, including the Model Optimizer.  

Sample code, specifically the `classification_sample`, is located at:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`~/inference_engine_samples/intel64/Release`<br>

The Model Optimizer is abbreviated `MO` for the remainder of this document.

---

1. Setup your environment for OpenVINO.<br><br>
    ```
    source /opt/intel/computer_vision_sdk/bin/setupvars.sh 
    ```

2. Run the MO extension generator and answer questions as appropriate 
    * We're using `/home/user/cl_cosh` as the target extension path<br><br>
    ```
    /opt/intel/computer_vision_sdk/deployment_tools/extension_generator/extgen.py new mo-op ie-cpu-ext output_dir=~/cl_cosh
    ```

3. Add Custom (cosh) Python Layers
    1. Copy to the Model Optimizer Ops Directory<br><br>
    ```
    cp ~/CustomLayers/Shubha_Code/Misc/cosh.py /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/model_optimizer/mo/ops/
    ```

    2. Copy to Extension Generation Python Target Dir<br><br>
    ```
    cp ~/CustomLayers/Shubha_Code/Misc/cosh_ext.py ~/cl_cosh/user_mo_extensions/ops/cosh_ext.py
    ```

4. Copy CPU and GPU source code to the M.O. extensions directory<br><br>
    ```
    cp ext_cosh.cpp /home/vino/cl_cosh/user_ie_extensions/cpu/
    cp cosh.cl /home/vino/cl_cosh/user_ie_extensions/gpu/New_cosh.cl
    ```

5. Fix the ie_parallel header file<br><br>
    ```
    sudo vi /opt/intel/computer_vision_sdk_2018.5.455/deployment_tools/inference_engine/include/ie_parallel.hpp
    ```

6. Copy the cosh.py algorithm file to the M.O. ops library path.<br><br>
    ```
    sudo cp cosh.py /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo/ops/
    ```

7. Create the TensorFlow graph files (weights, graphs, checkpoints)<br><br>
    ```
    ./build_cosh_model.py
    ```

8. Convert the TensorFlow model to Intel IR format<br><br>
    ```
    mo_tf.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCosh/Activation_8/softmax_output" --extensions ~/cl_cosh/user_mo_extensions
~/inference_engine_samples/intel64/Release/classification_sample -i pics/dog.bmp -m model.ckpt.xml -d CPU -l libuser_cpu_extension.so 
    ```

9. Compile the C++ extension library<br><br>
    ```cmake ..```<br>
    ```make -j$(nproc)```<br>
    ```cp libuser_cpu_extension.so ~/cl_cosh/```<br>

10. Test your results<br><br>
    ```
    ~/inference_engine_samples/intel64/Release/classification_sample -i pics/dog.bmp -m model.ckpt.xml -d CPU -l libuser_cpu_extension.so 
    ```

