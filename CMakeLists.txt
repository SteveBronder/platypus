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

