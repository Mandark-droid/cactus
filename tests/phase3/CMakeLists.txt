cmake_minimum_required(VERSION 3.10)
project(CactusPhase3Test CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CACTUS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../cactus)

add_compile_definitions(
    PLATFORM_CPU_ONLY=1
    __ARM_NEON=1
    __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
    __ARM_FEATURE_DOTPROD=1
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -funroll-loops -ftree-vectorize")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffunction-sections -fdata-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod")
endif()

include_directories(
    ${CACTUS_ROOT}
    ${CACTUS_ROOT}/engine
    ${CACTUS_ROOT}/graph
    ${CACTUS_ROOT}/kernel
    ${CACTUS_ROOT}/ffi
    ${CACTUS_ROOT}/models
    ${CACTUS_ROOT}/npu
    ${CACTUS_ROOT}/telemetry
)

file(GLOB KERNEL_SRCS   ${CACTUS_ROOT}/kernel/*.cpp)
file(GLOB GRAPH_SRCS    ${CACTUS_ROOT}/graph/*.cpp)
file(GLOB ENGINE_SRCS   ${CACTUS_ROOT}/engine/*.cpp)
file(GLOB FFI_SRCS      ${CACTUS_ROOT}/ffi/*.cpp)
file(GLOB MODEL_SRCS    ${CACTUS_ROOT}/models/*.cpp)
file(GLOB NPU_SRCS      ${CACTUS_ROOT}/npu/*.cpp)

set(TELEMETRY_STUB ${CMAKE_CURRENT_SOURCE_DIR}/telemetry_stub.cpp)

list(FILTER FFI_SRCS EXCLUDE REGEX ".*cactus_telemetry\\.cpp$")

add_library(cactus_static STATIC
    ${KERNEL_SRCS}
    ${GRAPH_SRCS}
    ${ENGINE_SRCS}
    ${FFI_SRCS}
    ${MODEL_SRCS}
    ${NPU_SRCS}
    ${TELEMETRY_STUB}
)

target_compile_options(cactus_static PRIVATE -Wall -Wextra -pedantic -Wno-c++20-designator -Wno-missing-field-initializers)

add_executable(test_qwen3_moe test_qwen3_moe.cpp)
target_link_libraries(test_qwen3_moe cactus_static log)
