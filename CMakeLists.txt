cmake_minimum_required(VERSION 3.10)

# Project name and version
project(Rezvangrad VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the main library
add_library(rezvangrad STATIC
    rezvangrad/engine.cpp
    rezvangrad/nn.cpp
)

# Include directories
target_include_directories(rezvangrad PUBLIC "${PROJECT_SOURCE_DIR}/rezvangrad")

# Add the test executables
add_executable(test_engine
    test/test_engine.cpp
)

add_executable(test_nn
    test/test_nn.cpp
)

# Link the library to the test executables
target_link_libraries(test_engine PRIVATE rezvangrad)
target_link_libraries(test_nn PRIVATE rezvangrad)
