# Unsafe NNLib

find_package(OpenMP)
find_package(DNNL)

add_library(NNLibUnsafe STATIC src/dnnl_utils.cpp src/NeuralNetwork.cpp ${HEADER_LIST})

set_target_properties(NNLibUnsafe PROPERTIES COMPILE_FLAGS ${APP_CXX_FLAGS})
target_include_directories(NNLibUnsafe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(NNLibUnsafe OpenMP::OpenMP_CXX DNNL::dnnl)

# file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

# Safe NNLib

add_trusted_library(NNLibEnclave SRCS src/dnnl_utils.cpp src/NeuralNetwork.cpp ${HEADER_LIST})
target_include_directories(NNLibEnclave PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(NNLibEnclave PRIVATE ${SGXDNNL_INCLUDE_DIR})
target_link_directories(NNLibEnclave PRIVATE ${SGXDNNL_LINK_DIR})