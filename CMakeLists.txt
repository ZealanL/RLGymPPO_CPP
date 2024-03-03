cmake_minimum_required (VERSION 3.8)

project("RLGymPPO_CPP")

# Make sure CMake finds libtorch if its in this directory
if (EXISTS "${PROJECT_SOURCE_DIR}/libtorch/")
	message("Using local libtorch folder...")
	list(APPEND CMAKE_PREFIX_PATH "libtorch")
	# Make ultra-sure we can find libtorch if its local
	set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
endif()

# Add libtorch (https://pytorch.org/cppdocs/installing.html#minimal-example)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp" "libsrc/*.cpp" "libsrc/.h" "libsrc/.hpp")

add_executable(RLGymPPO_CPP ${FILES_SRC})

# Include libtorch
target_link_libraries(RLGymPPO_CPP "${TORCH_LIBRARIES}")

# Set C++ version to 20
set_target_properties(RLGymPPO_CPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLGymPPO_CPP PROPERTIES CXX_STANDARD 20)

# Include RLGymSim_CPP
add_subdirectory(RLGymSim_CPP)
target_link_libraries(RLGymPPO_CPP RLGymSim_CPP)

# Make our python files copy over to our build dir
configure_file("./python_scripts/metrics_receiver.py" "./python_scripts/metrics_receiver.py" COPY)

# MSVC sometimes won't link to the libtorch DLLs unless you do this
# This is also from https://pytorch.org/cppdocs/installing.html#minimal-example
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	message("Adding TORCH_DLLS: ${TORCH_DLLS}")
    add_custom_command(TARGET RLGymPPO_CPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}
                 $<TARGET_FILE_DIR:RLGymPPO_CPP>)
endif (MSVC)