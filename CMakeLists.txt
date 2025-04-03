cmake_minimum_required(VERSION 3.10)
project(hw4)

# Enable C++11 or later
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and include Eigen
find_package(Eigen3 REQUIRED)

# Include the 'include' directory for headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add the source files from 'src' directory
file(GLOB SOURCES "src/*.cpp")

# Create the executable
add_executable(my_program ${SOURCES})

# Link Eigen to the project
target_link_libraries(my_program Eigen3::Eigen)

# Include Eigen headers (Header-only library, no linking needed)
target_include_directories(my_program PRIVATE ${EIGEN3_INCLUDE_DIR})
