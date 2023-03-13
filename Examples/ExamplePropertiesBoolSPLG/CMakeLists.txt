#Example
set_source_files_properties(exampleGPU_PropertiesKernel.cu PROPERTIES LANGUAGES "CUDA")

add_executable(Properties exampleGPU_PropertiesKernel.cu)

set_property(TARGET Properties PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#for cuda does not link properly if shared even after it is changed

target_include_directories(Properties PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include/BoolSPLG) #include directory for files that are needed for build

target_compile_features(Properties PUBLIC cxx_std_17)

#set_target_properties(sort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# allow components of a CUDA program to be compiled into separate objects

#set_property(TARGET sort PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#device linking to occur before consumption by a shared library or executable
