
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################
set(DNNL_CPU_RUNTIME "OMP")
set(DNNL_GPU_RUNTIME "NONE")

# Use a custom find module for transitive dependencies
set(DNNL_ORIGINAL_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${PACKAGE_PREFIX_DIR}/lib/cmake/dnnl)

if(DNNL_CPU_RUNTIME STREQUAL "TBB")
    find_package(TBB REQUIRED)
endif()

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    find_package(OpenCL REQUIRED)
endif()

# Reverting the CMAKE_MODULE_PATH to its original state
set(CMAKE_MODULE_PATH ${DNNL_ORIGINAL_CMAKE_MODULE_PATH})

include("${CMAKE_CURRENT_LIST_DIR}/dnnl-targets.cmake")
check_required_components("dnnl")
