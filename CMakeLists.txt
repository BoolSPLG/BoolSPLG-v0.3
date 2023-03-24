#Cmake build script

#Require a minimum version of cmake.
cmake_minimum_required(VERSION 3.26)

#Set the name of the project. Optional, languages C, CXX (i.e. C++), CUDA.
project(BoolSPLG-v0.3 LANGUAGES CUDA CXX)

#Find a package (usually provided by something external to the project), and load its package-specific details.
find_package(CUDA 10.0 REQUIRED)

#Add a library to the project using the specified source files.
add_library(BoolSPLG INTERFACE)

#Add sources to a target.
target_sources(BoolSPLG INTERFACE FILE_SET HEADERS BASE_DIRS ${PROJECT_SOURCE_DIR} FILES "BoolSPLG/BoolSPLG_v03.cuh")

#install BoolSPLG-v0.3 library. This command generates installation rules for a project.  

#install (copy) files from the library to CUDA include directory 
install(FILES ${PROJECT_SOURCE_DIR}/BoolSPLG/2D_DynamicArray.h ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_algorithms.cuh ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_base_funct.cuh ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_dev_funct.cuh ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_host_boolean_funct.h ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_host_boolean_funct_v03.h ${PROJECT_SOURCE_DIR}/BoolSPLG/boolsplg_host_sbox_funct.h ${PROJECT_SOURCE_DIR}/BoolSPLG/BoolSPLG_v03.cuh ${PROJECT_SOURCE_DIR}/BoolSPLG/cuda_help_heder.h ${PROJECT_SOURCE_DIR}/BoolSPLG/CUDA_Properties.h ${PROJECT_SOURCE_DIR}/BoolSPLG/reduction.h ${PROJECT_SOURCE_DIR}/BoolSPLG/reduction_kernel.cuh DESTINATION ${CUDA_TOOLKIT_ROOT_DIR}/include/BoolSPLG)
#install (copy) BoolSPLG directory from the library to CUDA include directory 
#install(DIRECTORY ${PROJECT_SOURCE_DIR}/BoolSPLG DESTINATION ${CUDA_TOOLKIT_ROOT_DIR}/include)

#Add a subdirectory to the build.
add_subdirectory(Examples/CodeExample)
add_subdirectory(Examples/ExampleBooleanBoolSPLG)
add_subdirectory(Examples/ExamplePropertiesBoolSPLG)
add_subdirectory(Examples/ExampleSboxBoolSPLG)





