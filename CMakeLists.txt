cmake_minimum_required(VERSION 3.10)
project(Rezvangrad VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the rezvangrad library
add_library(rezvangrad INTERFACE)
target_include_directories(rezvangrad INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Add the example executable
add_executable(rezvangrad_example examples/example.cpp)
target_link_libraries(rezvangrad_example PRIVATE rezvangrad)

# Optional: Add test executables if you have any
# add_executable(test_engine test/test_engine.cpp)
# target_link_libraries(test_engine PRIVATE rezvangrad)

# add_executable(test_nn test/test_nn.cpp)
# target_link_libraries(test_nn PRIVATE rezvangrad)

# Enable testing
enable_testing()

# Add tests
# add_test(NAME TestEngine COMMAND test_engine)
# add_test(NAME TestNN COMMAND test_nn)
