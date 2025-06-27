cmake_minimum_required(VERSION 3.16)
project(cpp-ray-tracer VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to enable CUDA if available
# First try the modern approach
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA Toolkit found - building both CPU and CUDA versions")
else()
    # Fallback to old CUDA detection
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CUDA_AVAILABLE TRUE)
        message(STATUS "CUDA found (legacy) - building both CPU and CUDA versions")
    else()
        set(CUDA_AVAILABLE FALSE)
        message(STATUS "CUDA not found - building CPU version only")
    endif()
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add CPU executable
add_executable(${PROJECT_NAME}-cpu src/cpu/main.cc)
target_include_directories(${PROJECT_NAME}-cpu PRIVATE external src src/cpu)

# Set compiler flags for CPU version
if(MSVC)
    target_compile_options(${PROJECT_NAME}-cpu PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME}-cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Add CUDA executable only if CUDA is available
if(CUDA_AVAILABLE)
    add_executable(${PROJECT_NAME}-cuda src/cuda/main.cu)
    target_include_directories(${PROJECT_NAME}-cuda PRIVATE external src src/cuda)

    # Minimal CUDA configuration to avoid compilation issues
    set_target_properties(${PROJECT_NAME}-cuda PROPERTIES
        CUDA_ARCHITECTURES "75"
    )
endif()


