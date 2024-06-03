cmake_minimum_required(VERSION 3.16.3)
project(
    platypus
    VERSION 0.0.1
    LANGUAGES C CXX)

include(FetchContent)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_VERBOSE_MAKEFILE YES)
cmake_policy(SET CMP0069 NEW)
# Configuration Options

find_package(OpenCV REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(platypus STATIC ${platypus_SOURCE_DIR}/src/CradleFunctions.cpp
${platypus_SOURCE_DIR}/src/DWT.cpp
${platypus_SOURCE_DIR}/src/FDCT.cpp
${platypus_SOURCE_DIR}/src/FFST.cpp
${platypus_SOURCE_DIR}/src/HaarDWT.cpp
${platypus_SOURCE_DIR}/src/MCA.cpp
${platypus_SOURCE_DIR}/src/Shearlet.cpp
${platypus_SOURCE_DIR}/src/TextureRemoval.cpp)
target_include_directories(platypus PRIVATE ${platypus_SOURCE_DIR}/include ${OPENCV_INCLUDE_DIR})
target_link_libraries(platypus ${OpenCV_LIBS})


add_executable(Demo ${platypus_SOURCE_DIR}/exe/mainDemo.cpp)
target_include_directories(Demo PRIVATE ${platypus_SOURCE_DIR}/include ${OPENCV_INCLUDE_DIR})
target_link_libraries(Demo ${OpenCV_LIBS} platypus)

add_executable(CradleRemoval ${platypus_SOURCE_DIR}/exe/mainCradleRemoval.cpp)
target_include_directories(CradleRemoval PRIVATE ${platypus_SOURCE_DIR}/include ${OPENCV_INCLUDE_DIR})
target_link_libraries(CradleRemoval ${OpenCV_LIBS} platypus)

add_executable(TextureRemoval ${platypus_SOURCE_DIR}/exe/mainTextureRemoval.cpp)
target_include_directories(TextureRemoval PRIVATE ${platypus_SOURCE_DIR}/include ${OPENCV_INCLUDE_DIR})
target_link_libraries(TextureRemoval ${OpenCV_LIBS} platypus)
