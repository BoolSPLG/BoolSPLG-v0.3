#Example
set_source_files_properties(exampleBooleankernel.cu PROPERTIES LANGUAGES "CUDA")

add_executable(Boolean exampleBooleankernel.cu "${HEADER_LIST_BOOL}")

set_property(TARGET Boolean PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#for cuda does not link properly if shared even after it is changed

target_include_directories(Boolean PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include/BoolSPLG) #include directory for files that are needed for build

target_compile_features(Boolean PUBLIC cxx_std_17)

#source_group(
# TREE "${PROJECT_SOURCE_DIR}/Examples/ExampleBooleanBoolSPLG"
#  PREFIX "Header Files"
#  FILES "${HEADER_LIST_BOOL}")

#set_target_properties(sort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# allow components of a CUDA program to be compiled into separate objects

#set_property(TARGET sort PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#device linking to occur before consumption by a shared library or executable




