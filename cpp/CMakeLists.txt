cmake_minimum_required(VERSION 3.16)
project(rapidstamp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RAPIDSTAMP_BUILD_TESTS "Build C++ tests" ON)
option(RAPIDSTAMP_BUILD_EXAMPLES "Build the PPM demo" ON)

if(NOT ORT_ROOT)
    message(FATAL_ERROR "Set -DORT_ROOT to an ONNX Runtime C/C++ release directory")
endif()

add_library(rapidstamp SHARED src/engine.cpp src/geometry.cpp)
add_library(rapidstamp_static STATIC src/engine.cpp src/geometry.cpp)
# Keep the static archive name distinct. On Windows both targets otherwise
# write rapidstamp.lib, and the API test links the archive without ONNX Runtime.

foreach(target rapidstamp rapidstamp_static)
    target_include_directories(${target}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${ORT_ROOT}/include)
    if(WIN32)
        target_link_libraries(${target} PRIVATE ${ORT_ROOT}/lib/onnxruntime.lib)
    else()
        target_link_directories(${target} PRIVATE ${ORT_ROOT}/lib)
        target_link_libraries(${target} PRIVATE onnxruntime)
    endif()
    if(ANDROID)
        target_link_libraries(${target} PRIVATE log)
    endif()
endforeach()

if(NOT WIN32)
    if(APPLE)
        set_target_properties(rapidstamp PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib" INSTALL_RPATH "@loader_path")
    else()
        set_target_properties(rapidstamp PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib" INSTALL_RPATH "$ORIGIN")
    endif()
endif()

if(RAPIDSTAMP_BUILD_TESTS)
    add_executable(rapidstamp_test_geom tests/test_geom.cpp src/geometry.cpp)
    target_include_directories(rapidstamp_test_geom PRIVATE src include)
    add_executable(rapidstamp_test_api tests/test_api.cpp)
    target_link_libraries(rapidstamp_test_api PRIVATE rapidstamp)
    target_include_directories(rapidstamp_test_api PRIVATE include)
    if(NOT WIN32)
        if(APPLE)
            set_target_properties(rapidstamp_test_api PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")
        else()
            set_target_properties(rapidstamp_test_api PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")
        endif()
    endif()
endif()

if(RAPIDSTAMP_BUILD_EXAMPLES)
    add_executable(rapidstamp_demo examples/demo.cpp)
    target_link_libraries(rapidstamp_demo PRIVATE rapidstamp)
    target_include_directories(rapidstamp_demo PRIVATE include)
    if(NOT WIN32)
        set_target_properties(rapidstamp_demo PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")
    endif()
endif()

install(TARGETS rapidstamp rapidstamp_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
install(FILES include/rapidstamp.h DESTINATION include)
