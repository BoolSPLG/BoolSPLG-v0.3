#build example, ExamplePropertiesBoolSPLG

#Source files can have properties that affect how they are built.
set_source_files_properties(exampleGPU_PropertiesKernel.cu PROPERTIES LANGUAGES "CUDA")

#Find a package (usually provided by something external to the project), and load its package-specific details.
find_package(CUDA 10.0 REQUIRED)

#Add an executable to the project using the specified source files.
add_executable(Properties)

#for cuda does not link properly if shared even after it is changed
set_property(TARGET Properties PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#Add sources to a target.
target_sources(Properties PUBLIC "exampleGPU_PropertiesKernel.cu")

#Specify libraries or flags to use when linking a given target and/or its dependents.
target_link_libraries(Properties PUBLIC BoolSPLG)

#Add expected compiler features to a target.
#target_compile_features(Properties PUBLIC cxx_std_17)