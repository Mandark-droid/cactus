cmake_minimum_required(VERSION 3.10)
project(CactusPhase3Test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../cactus)

# Collect all source files
file(GLOB ENGINE_SOURCES "${SOURCE_DIR}/engine/*.cpp")
file(GLOB GRAPH_SOURCES "${SOURCE_DIR}/graph/*.cpp")
file(GLOB KERNEL_SOURCES "${SOURCE_DIR}/kernel/*.cpp")
file(GLOB MODEL_SOURCES "${SOURCE_DIR}/models/*.cpp")
file(GLOB FFI_SOURCES "${SOURCE_DIR}/ffi/*.cpp")
set(NPU_SOURCES "${SOURCE_DIR}/npu/npu.cpp")

# Exclude real telemetry (needs curl), use stub instead
set(TELEMETRY_STUB "${CMAKE_SOURCE_DIR}/telemetry_stub.cpp")

set(ALL_SOURCES
    ${KERNEL_SOURCES}
    ${GRAPH_SOURCES}
    ${ENGINE_SOURCES}
    ${FFI_SOURCES}
    ${MODEL_SOURCES}
    ${NPU_SOURCES}
    ${TELEMETRY_STUB}
)

if(NOT ALL_SOURCES)
    message(FATAL_ERROR "No source files found! Check SOURCE_DIR: ${SOURCE_DIR}")
endif()

# Build static library
add_library(cactus_static STATIC ${ALL_SOURCES})

set(COMMON_INCLUDE_DIRS
    ${SOURCE_DIR}
    ${SOURCE_DIR}/engine
    ${SOURCE_DIR}/graph
    ${SOURCE_DIR}/kernel
    ${SOURCE_DIR}/ffi
    ${SOURCE_DIR}/models
    ${SOURCE_DIR}/npu
    ${SOURCE_DIR}/telemetry
)

target_include_directories(cactus_static PRIVATE ${COMMON_INCLUDE_DIRS})

# No curl, no telemetry network calls
target_compile_definitions(cactus_static PRIVATE PLATFORM_CPU_ONLY=1)

target_compile_options(cactus_static PRIVATE
    -pthread -Wall -Wextra -pedantic
    -Wno-c++20-designator
    -Wno-missing-field-initializers
    -fomit-frame-pointer -funroll-loops -ftree-vectorize
    -O3 -DNDEBUG -ffunction-sections -fdata-sections
)

# ARM-specific flags for Android arm64-v8a
if(ANDROID AND ${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_compile_options(cactus_static PRIVATE -march=armv8.2-a+fp16+simd+dotprod)
    target_compile_definitions(cactus_static PRIVATE
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
    )
endif()

# Build test binary
add_executable(test_qwen3_moe test_qwen3_moe.cpp)
target_link_libraries(test_qwen3_moe PRIVATE cactus_static)
target_include_directories(test_qwen3_moe PRIVATE ${SOURCE_DIR})

if(ANDROID)
    find_library(LOG_LIB log)
    if(LOG_LIB)
        target_link_libraries(test_qwen3_moe PRIVATE ${LOG_LIB})
        target_link_libraries(cactus_static ${LOG_LIB})
    endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
