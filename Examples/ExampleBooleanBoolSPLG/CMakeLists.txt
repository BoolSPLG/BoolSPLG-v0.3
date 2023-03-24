#build example, ExampleBooleanBoolSPLG

#Source files can have properties that affect how they are built.
set_source_files_properties(exampleBooleankernel.cu PROPERTIES LANGUAGES "CUDA")

#Find a package (usually provided by something external to the project), and load its package-specific details.
find_package(CUDA 10.0 REQUIRED)

#Add an executable to the project using the specified source files.
add_executable(Boolean)
#add_executable(Boolean exampleBooleankernel.cu "${HEADER_LIST_BOOL}")

set_property(TARGET Boolean PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#for cuda does not link properly if shared even after it is changed

#Add sources to a target.
target_sources(Boolean PUBLIC "exampleBooleankernel.cu")

#Specify libraries or flags to use when linking a given target and/or its dependents.
target_link_libraries(Boolean PUBLIC BoolSPLG)

#Add expected compiler features to a target.
#target_compile_features(Boolean PUBLIC cxx_std_17)



