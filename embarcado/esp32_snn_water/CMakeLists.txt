cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(snn_water_quality)

# Versão do projeto
set(PROJECT_VERSION "1.0.0")

# Add custom compiler flags for optimization
add_compile_options(-O2 -g)

# Mensagem de configuração
message(STATUS "===========================================")
message(STATUS "SNN Water Quality Monitor v${PROJECT_VERSION}")
message(STATUS "===========================================")
message(STATUS "Target: ${IDF_TARGET}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build directory: ${CMAKE_BINARY_DIR}")
message(STATUS "===========================================")
