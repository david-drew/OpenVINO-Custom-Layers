#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===============================================================================
# Generated file for building library with user generated CPU extensions
#
# Contains implementation of the basic layer methods
#
# Refer to the section "Adding Your Own Kernels to the Inference Engine" in
# OpenVINO* documentation (either online or offline in
# <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
# to the corresponding section).
# ===============================================================================
cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME "user_cpu_extension")

get_filename_component(CMAKE_PREFIX_PATH "${InferenceEngine_DIR}" ABSOLUTE)

message(STATUS "Looking for inference engine configuration file at: ${CMAKE_PREFIX_PATH}")
find_package(InferenceEngine 1.0)

if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR "Inference Engine was not found")
endif()

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

file(GLOB_RECURSE SRC *.cpp)
file(GLOB_RECURSE HDR *.hpp)

if(WIN32)
    add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)
endif()

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

if ("${InferenceEngine_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "InferenceEngine_INCLUDE_DIRS was not set!")
endif()

if ("${InferenceEngine_LIBRARIES}" STREQUAL "")
    message(FATAL_ERROR "InferenceEngine_LIBRARIES was not set!")
endif()

include_directories (PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${InferenceEngine_INCLUDE_DIRS}
    /opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include
)

#enable_omp()

add_library(${TARGET_NAME} SHARED ${SRC} ${HDR})
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

#set_target_cpu_flags(${TARGET_NAME})
