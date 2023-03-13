set(HEADER_LIST_SBOX "funct_Sbox_CPU.h")
#Example
set_source_files_properties(exampleSboxkernel.cu PROPERTIES LANGUAGES "CUDA")

add_executable(Sbox exampleSboxkernel.cu "${HEADER_LIST_SBOX}")

set_property(TARGET Sbox PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#for cuda does not link properly if shared even after it is changed

target_include_directories(Sbox PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include/BoolSPLG) #include directory for files that are needed for build

target_compile_features(Sbox PUBLIC cxx_std_17)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/Examples/ExampleSboxBoolSPLG"
  PREFIX "Header Files"
  FILES "${HEADER_LIST_SBOX}")

 #copy file "sbox" (input S-box example) to binary directory
file(COPY ${PROJECT_SOURCE_DIR}/Examples/ExampleSboxBoolSPLG/sbox DESTINATION ${CMAKE_BINARY_DIR}/Examples/ExampleSboxBoolSPLG) 

#set_target_properties(sort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# allow components of a CUDA program to be compiled into separate objects

#set_property(TARGET sort PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#device linking to occur before consumption by a shared library or executable




