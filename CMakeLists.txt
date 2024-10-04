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
option(OpenCV_DIR "Path to OpenCV if not found automatically" "")

# Attempt to find OpenCV first in the default paths
find_package(OpenCV)

# If still not found, provide an error message
if(NOT OpenCV_FOUND AND NOT OpenCV_DIR)
    message(FATAL_ERROR "OpenCV not found! Please set OpenCV_DIR to the path where OpenCV is installed.")
endif()

# If not found, try using the user-specified path
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)


# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define the platypus library
add_library(platypus STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CradleFunctions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/DWT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FDCT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FFST.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/HaarDWT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MCA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Shearlet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TextureRemoval.cpp)

target_include_directories(platypus PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>  # This tells CMake where to find the headers when installed
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(platypus PUBLIC ${OpenCV_LIBS})

# Export the platypus target for use by other projects
export(TARGETS platypus FILE platypusTargets.cmake)

# Install the library and headers
install(TARGETS platypus
    EXPORT platypusTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include)

# Install the export targets
install(EXPORT platypusTargets
    FILE platypusTargets.cmake
    NAMESPACE platypus::
    DESTINATION lib/cmake/platypus)

# Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/platypusConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

# Configure the Config.cmake file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/platypusConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/platypusConfig.cmake"
    INSTALL_DESTINATION lib/cmake/platypus)

# Install the Config and ConfigVersion files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/platypusConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/platypusConfigVersion.cmake"
    DESTINATION lib/cmake/platypus)

# Optionally, build executables if this is the main project
option(BUILD_EXAMPLES "Build example executables" OFF)

if(BUILD_EXAMPLES)
    add_executable(Demo ${CMAKE_CURRENT_SOURCE_DIR}/exe/mainDemo.cpp)
    target_include_directories(Demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(Demo PRIVATE platypus)

    add_executable(CradleRemoval ${CMAKE_CURRENT_SOURCE_DIR}/exe/mainCradleRemoval.cpp)
    target_include_directories(CradleRemoval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(CradleRemoval PRIVATE platypus)

    add_executable(TextureRemoval ${CMAKE_CURRENT_SOURCE_DIR}/exe/mainTextureRemoval.cpp)
    target_include_directories(TextureRemoval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(TextureRemoval PRIVATE platypus)
endif()

if (BUILD_GUI)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gui)
endif()
