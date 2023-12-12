cmake_minimum_required(VERSION 3.16...3.22)

# Specifically set compiler to GCC, because Clang is not supported by SGX
# Comment out these lines to set the compiler to the CMake default. This may require deleting the CMake cache.
# set(CMAKE_C_COMPILER /usr/bin/gcc-9 CACHE PATH "" FORCE)
# set(CMAKE_CXX_COMPILER /usr/bin/g++-9 CACHE PATH "" FORCE)

project(SGX4ML-CPP)

# For debugging problems related to includes, compilation, linking etc.:
# set(CMAKE_VERBOSE_MAKEFILE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/external/dnnl)
find_package(SGX REQUIRED)
find_package(MNIST PATHS cmake)
find_package(NPY PATHS cmake)
find_package(SGXDNNL PATHS cmake)

include_directories(${MNIST_INCLUDE_DIR})
include_directories(${NPY_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(E_SRCS Enclave/Enclave.cpp Enclave/neural_network_inference.cpp)
set(EDL_SEARCH_PATHS Enclave)
set(LDS Enclave/Enclave.lds)

set(BIG_ENCLAVE ON CACHE BOOL "Whether to use the big (ON) or the small (OFF) enclave config.")
if(BIG_ENCLAVE)
    set(ENCLAVE_CONFIG_NAME "Enclave.config.big.xml")
else()
    set(ENCLAVE_CONFIG_NAME "Enclave.config.small.xml")
endif()
set(ENCLAVE_FILE_NAME "enclave.signed.so")

configure_file("${PROJECT_SOURCE_DIR}/App/Config.hpp.in" "${PROJECT_BINARY_DIR}/Config.hpp")

add_subdirectory(external/NeuralNetworkLib)

add_enclave_library(enclave SRCS ${E_SRCS}
                            EDL Enclave/Enclave.edl
                            EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS}
                            LDSCRIPT ${LDS})
target_include_directories(enclave PRIVATE ${SGXDNNL_INCLUDE_DIR})
target_link_directories(enclave PRIVATE ${SGXDNNL_LINK_DIR})
target_link_libraries(enclave NNLibEnclave)

enclave_sign(enclave KEY Enclave/Enclave_private.pem CONFIG "Enclave/${ENCLAVE_CONFIG_NAME}")

set(SRCS App/App.cpp App/OCalls.cpp)
add_untrusted_executable(App SRCS ${SRCS} EDL Enclave/Enclave.edl EDL_SEARCH_PATHS ${EDL_SEARCH_PATHS})
target_link_libraries(App NNLibUnsafe)
add_dependencies(App enclave-sign)