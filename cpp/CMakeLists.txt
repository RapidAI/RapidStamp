cmake_minimum_required(VERSION 3.16)
project(rapidstamp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Mobile builds are static" FORCE)
elseif(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

option(RAPIDSTAMP_BUILD_TOOLS "Build the demo, parity tool, and unit test" ON)
if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(RAPIDSTAMP_BUILD_TOOLS OFF)
endif()

# Object library so the demo can link the shared C API while the unit test
# and parity tool still call the internal interpreter. Those symbols stay
# hidden in the shared library.
add_library(rapidstamp_objs OBJECT
    src/runtime.cpp
    src/geometry.cpp
    src/api.cpp
)
target_include_directories(rapidstamp_objs
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_compile_definitions(rapidstamp_objs PRIVATE RAPIDSTAMP_BUILD)
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(rapidstamp_objs PUBLIC RAPIDSTAMP_STATIC)
endif()
if(MSVC)
    target_compile_options(rapidstamp_objs PRIVATE /W3 /EHsc)
else()
    target_compile_options(rapidstamp_objs PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(rapidstamp_objs PRIVATE $<$<CONFIG:Release>:-O3>)
endif()
if(NOT APPLE AND NOT MSVC)
    set_target_properties(rapidstamp_objs PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()

add_library(rapidstamp $<TARGET_OBJECTS:rapidstamp_objs>)
target_include_directories(rapidstamp
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(rapidstamp PUBLIC RAPIDSTAMP_STATIC)
endif()

if(RAPIDSTAMP_BUILD_TOOLS)
    add_executable(rapidstamp_demo examples/demo.cpp)
    target_link_libraries(rapidstamp_demo PRIVATE rapidstamp)

    # Link the objects, not the shared library: rsp::* is hidden there.
    add_executable(rapidstamp_parity examples/parity.cpp $<TARGET_OBJECTS:rapidstamp_objs>)
    target_include_directories(rapidstamp_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(rapidstamp_test tests/test_runtime.cpp $<TARGET_OBJECTS:rapidstamp_objs>)
    target_include_directories(rapidstamp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)

    enable_testing()
    add_test(NAME runtime COMMAND rapidstamp_test)
endif()

include(GNUInstallDirs)
install(TARGETS rapidstamp
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES include/rapidstamp.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(RAPIDSTAMP_BUILD_TOOLS)
    install(TARGETS rapidstamp_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
