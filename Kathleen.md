cd %CLWS%\tf_model

%OV%\deployment_tools\model_optimizer\mo_tf.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCosh/Activation_8/softmax_output" --extensions %CLWS%\cl_cosh\user_mo_extensions --output_dir %CLWS%\cl_ext_cosh


<br><br><b>
DONE
...done, I say!</b>
<br><br>

Requirements (New Env Vars):

  set OV="C:\Program Files (x86)\IntelSWTools\openvino"
  

python %OV%\deployment_tools\tools\extension_generator\extgen.py new --help

copy %CLT%\ext_cosh.cpp %OV%\deployment_tools\inference_engine\src\extension

python %OV%\deployment_tools\tools\extension_generator\extgen.py new --mo-tf-ext --mo-op --ie-cpu-ext --ie-gpu-ext --output_dir=%CLWS%\cl_cosh

