
copy %CLT%\ext_cosh.cpp %OV%\deployment_tools\inference_engine\src\extension


DONE
...done, I say!

Requirements (New Env Vars):

  set OV="C:\Program Files (x86)\IntelSWTools\openvino"
  

python %OV%\deployment_tools\tools\extension_generator\extgen.py new --help
