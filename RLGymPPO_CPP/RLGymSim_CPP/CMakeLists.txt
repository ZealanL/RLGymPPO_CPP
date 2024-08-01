cmake_minimum_required (VERSION 3.8)

project("RLGymSim_CPP")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_library(RLGymSim_CPP STATIC ${FILES_SRC})
target_include_directories(RLGymSim_CPP PUBLIC "src/")

# Set C++ version to 20
set_target_properties(RLGymSim_CPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLGymSim_CPP PROPERTIES CXX_STANDARD 20)

# Include RocketSim
add_subdirectory("RocketSim")
target_link_libraries(RLGymSim_CPP RocketSim)