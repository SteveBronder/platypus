# CMakeLists.txt

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Find the required Qt packages
find_package(Qt5 5 REQUIRED COMPONENTS Core Gui Widgets Xml)

# Collect all source files
file(GLOB_RECURSE SOURCES
    src/*.cpp
)

# Collect all header files
file(GLOB_RECURSE HEADERS
    include/platypus_gui/*.hpp
)

# Add the executable
add_executable(PlatypusGui
    ${SOURCES}
    main.cpp
    platypus.qrc
)

# Link against Qt and OpenCV libraries
target_link_libraries(PlatypusGui PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Qt5::Xml
    ${OpenCV_LIBS}
    platypus
)

# Add include directories
target_include_directories(PlatypusGui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR} # For generated files
    ${OpenCV_INCLUDE_DIRS}
)

# Add headers to the target sources to assist MOC
target_sources(PlatypusGui PRIVATE ${HEADERS})
