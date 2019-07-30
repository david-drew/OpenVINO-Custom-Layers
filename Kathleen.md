python %OV%\deployment_tools\tools\extension_generator\extgen.py new --mo-tf-ext --mo-op --ie-cpu-ext --ie-gpu-ext --output_dir=%CLWS%\cl_cosh

<br><br><b>
DONE
...done, I say!</b>
<br>

Requirements (New Env Vars):

  set OV="C:\Program Files (x86)\IntelSWTools\openvino"
  

python %OV%\deployment_tools\tools\extension_generator\extgen.py new --help

copy %CLT%\ext_cosh.cpp %OV%\deployment_tools\inference_engine\src\extension

