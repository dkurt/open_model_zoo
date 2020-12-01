#
# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ===============================================================================
# Generated file for building library with user generated CPU extensions
#
# Contains implementation of the basic layer methods
#
# Refer to the section "Adding Your Own Kernels to the Inference Engine" in
# OpenVINO* documentation (either online or offline in
# <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
# to the corresponding section).
# ===============================================================================

set(CPU_EXTENSIONS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

macro(ie_add_cpu_extension)
    set(options EXCLUDE_CPPLINT)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES DEPENDENCIES)
    cmake_parse_arguments(IE_EXTENSION "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    set(TARGET_NAME ${IE_EXTENSION_NAME})

    # find_package(ngraph REQUIRED)
    # find_package(InferenceEngine REQUIRED)

    file(GLOB_RECURSE SRC *.cpp)

    add_library(${TARGET_NAME} SHARED ${SRC} ${IE_EXTENSION_SOURCES})

    target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
    message("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    message(${InferenceEngine_LIBRARIES})
    message(${NGRAPH_LIBRARIES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES}
                                                 ${NGRAPH_LIBRARIES}
                                                 ${IE_EXTENSION_DEPENDENCIES})
endmacro()
