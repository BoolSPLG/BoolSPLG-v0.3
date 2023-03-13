#minimal requirements
cmake_minimum_required(VERSION 3.21)

project(BoolSPLG-v0.3 LANGUAGES CUDA CXX)

find_package(CUDA 10.0 REQUIRED)
install(FILES ${PROJECT_SOURCE_DIR}/BoolSPLG/2D_DynamicArray.h ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_algorithms.cuh ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_base_funct.cuh ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_dev_funct.cuh ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_host_boolean_funct.h ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_host_boolean_funct_v03.h ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_host_sbox_funct.h ${PROJECT_SOURCE_DIR}/BoolSPLG/BoolSPLG_v03.cuh ${PROJECT_SOURCE_DIR}/BoolSPLG/cuda_help_heder.h ${PROJECT_SOURCE_DIR}/BoolSPLG/CUDA_Properties.h ${PROJECT_SOURCE_DIR}/BoolSPLG/reduction.h ${PROJECT_SOURCE_DIR}/BoolSPLG/reduction_kernel.cuh DESTINATION ${CUDA_TOOLKIT_ROOT_DIR}/include/BoolSPLG)

add_subdirectory(Examples/ExampleBooleanBoolSPLG)
add_subdirectory(Examples/ExamplePropertiesBoolSPLG)
add_subdirectory(Examples/ExampleSboxBoolSPLG)
add_subdirectory(Examples/CodeExample)

