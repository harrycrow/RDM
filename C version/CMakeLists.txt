cmake_minimum_required(VERSION 3.16)
project(RDM C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(ENV{MKLROOT} "/opt/intel/compilers_and_libraries_2020.3.279/mac/mkl")
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD_REQUIRED YES)   # See below (1)
set(CMAKE_C_FLAGS "-lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result" )
set(CMAKE_CXX_FLAGS "-lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result" )

set(BLA_DEFINITIONS)

if ("$ENV{MKLROOT}" STREQUAL "")
    message(FATAL_ERROR "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
endif()

# user defined options for MKL
option(MKL_USE_parallel "Use MKL parallel" True)
option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

if (BLA_VENDOR MATCHES "_seq")
    set(MKL_USE_parallel OFF)
else()
    set(MKL_USE_parallel ON)
endif()

find_package(MKL REQUIRED)

if (MKL_FOUND)
    set(BLA_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    set(BLA_DEFINITIONS USE_MKL)
endif()

message(STATUS "BLAS Libraries: ${BLAS_LIBRARIES}")

add_executable(RDM main.c)

target_compile_definitions(RDM PUBLIC ${BLA_DEFINITIONS})
target_link_libraries(RDM ${BLAS_LIBRARIES})
target_include_directories(RDM
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${BLA_INCLUDE_DIR})
