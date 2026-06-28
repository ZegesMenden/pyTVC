cmake_minimum_required(VERSION 3.16)

project(pytvc_cpp_core LANGUAGES CXX)

option(PYTVC_SCALAR_DOUBLE "Use double instead of float for pytvc::Scalar" ON)
option(PYTVC_BUILD_TESTS "Build C++ core tests" ON)

add_library(pytvc_core INTERFACE)
target_include_directories(pytvc_core INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_features(pytvc_core INTERFACE cxx_std_17)

if(PYTVC_SCALAR_DOUBLE)
    target_compile_definitions(pytvc_core INTERFACE PYTVC_SCALAR_DOUBLE=1)
else()
    target_compile_definitions(pytvc_core INTERFACE PYTVC_SCALAR_DOUBLE=0)
endif()

if(PYTVC_BUILD_TESTS)
    enable_testing()

    add_executable(pytvc_core_tests tests/core_tests.cpp)
    target_link_libraries(pytvc_core_tests PRIVATE pytvc_core)
    add_test(NAME pytvc_core_tests COMMAND pytvc_core_tests)
endif()
