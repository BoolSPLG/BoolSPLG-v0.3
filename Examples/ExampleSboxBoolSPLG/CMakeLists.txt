#build example, ExampleSboxBoolSPLG

#Set a normal, cache, or environment variable to a given value.
set(HEADER_LIST_SBOX "funct_Sbox_CPU.h")

#Source files can have properties that affect how they are built.
set_source_files_properties(exampleSboxkernel.cu PROPERTIES LANGUAGES "CUDA")

#Find a package (usually provided by something external to the project), and load its package-specific details.
find_package(CUDA 10.0 REQUIRED)


#Add an executable to the project using the specified source files.
add_executable(Sbox)
#add_executable(Sbox exampleSboxkernel.cu "${HEADER_LIST_SBOX}")

#for cuda does not link properly if shared even after it is changed
set_property(TARGET Sbox PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#Add sources to a target.
target_sources(Sbox PUBLIC "exampleSboxkernel.cu")

#Specify libraries or flags to use when linking a given target and/or its dependents.
target_link_libraries(Sbox PUBLIC BoolSPLG)

#Add expected compiler features to a target.
#target_compile_features(Sbox PUBLIC cxx_std_17)

#Define a grouping for source files in IDE project generation.
source_group(
  TREE "${PROJECT_SOURCE_DIR}/Examples/ExampleSboxBoolSPLG"
  PREFIX "Header Files"
  FILES "${HEADER_LIST_SBOX}")

 #copy file "sbox" (input S-box example) to binary directory
file(COPY ${PROJECT_SOURCE_DIR}/Examples/ExampleSboxBoolSPLG/sbox DESTINATION ${CMAKE_BINARY_DIR}/Examples/ExampleSboxBoolSPLG) 





