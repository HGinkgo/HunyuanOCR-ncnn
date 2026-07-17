cmake_minimum_required(VERSION 3.18)

project(HunyuanOCR_ncnn
    VERSION 0.4.0
    LANGUAGES CXX
)

option(HUNYUAN_OCR_BUILD_CLI "Build the hunyuan_ocr_cli executable" ON)
option(HUNYUAN_OCR_BUILD_TESTS "Build HunyuanOCR-ncnn test executables" OFF)
option(HUNYUAN_OCR_STRICT_WARNINGS "Enable compiler warnings for project sources" ON)
option(HUNYUAN_OCR_USE_NCNN_PACKAGE "Try find_package(ncnn CONFIG) before manual ncnn paths" ON)
option(HUNYUAN_OCR_ENABLE_SDPA_PROFILE
    "Compile PreciseSDPA stage timers for decoder microbenchmarks (OFF by default)"
    OFF)
set(HUNYUAN_OCR_MIN_NCNN_VERSION 20260106)
set(HUNYUAN_OCR_BUILD_TOKENIZER_TEST "AUTO" CACHE STRING "Build tokenizer file parity test: AUTO, ON, or OFF")
set_property(CACHE HUNYUAN_OCR_BUILD_TOKENIZER_TEST PROPERTY STRINGS AUTO ON OFF)

if(HUNYUAN_OCR_USE_NCNN_PACKAGE)
    if(DEFINED ENV{ncnn_DIR} AND NOT ncnn_DIR)
        set(ncnn_DIR "$ENV{ncnn_DIR}" CACHE PATH "Path to ncnnConfig.cmake" FORCE)
    endif()
    find_package(ncnn ${HUNYUAN_OCR_MIN_NCNN_VERSION} CONFIG QUIET)
endif()

if(NOT TARGET ncnn)
    set(NCNN_INCLUDE_DIR "" CACHE PATH "Path to the ncnn source include directory, for example /path/to/ncnn/src")
    set(NCNN_BUILD_INCLUDE_DIR "" CACHE PATH "Path to the ncnn generated include directory, for example /path/to/ncnn/build/src")
    set(NCNN_LIBRARY "" CACHE FILEPATH "Path to libncnn.a, libncnn.so, or ncnn.lib")

    if(NOT NCNN_INCLUDE_DIR OR NOT NCNN_BUILD_INCLUDE_DIR OR NOT NCNN_LIBRARY)
        message(FATAL_ERROR
            "ncnn ${HUNYUAN_OCR_MIN_NCNN_VERSION} or newer was not found. "
            "Either pass -Dncnn_DIR=/path/to/ncnnConfig.cmake "
            "or pass -DNCNN_INCLUDE_DIR=/path/to/ncnn/src "
            "-DNCNN_BUILD_INCLUDE_DIR=/path/to/ncnn/build/src "
            "-DNCNN_LIBRARY=/path/to/libncnn.a"
        )
    endif()

    find_package(Threads REQUIRED)
    find_package(OpenMP QUIET)

    add_library(ncnn UNKNOWN IMPORTED GLOBAL)
    set_target_properties(ncnn PROPERTIES
        IMPORTED_LOCATION "${NCNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCNN_INCLUDE_DIR};${NCNN_BUILD_INCLUDE_DIR}"
        INTERFACE_POSITION_INDEPENDENT_CODE ON
    )

    set(_hunyuan_ocr_ncnn_link_libraries Threads::Threads)
    if(OpenMP_CXX_FOUND)
        list(APPEND _hunyuan_ocr_ncnn_link_libraries OpenMP::OpenMP_CXX)
    endif()
    set_property(TARGET ncnn PROPERTY
        INTERFACE_LINK_LIBRARIES "${_hunyuan_ocr_ncnn_link_libraries}"
    )
endif()

add_library(hunyuan_ocr STATIC)
target_sources(hunyuan_ocr
    PRIVATE
        src/batch_jsonl.cpp
        src/hunyuan_ocr.cpp
        src/dflash_runtime.cpp
        src/generation_config.cpp
        src/image_preprocessor.cpp
        src/kv_cache_capacity.cpp
        src/mapped_model_file.cpp
        src/model_layout.cpp
        src/multimodal_rope.cpp
        src/prompt_builder.cpp
        src/precise_sdpa.cpp
        src/text_runtime.cpp
        src/tokenizer.cpp
        src/utf8.cpp
        src/vision_runtime.cpp
)
target_include_directories(hunyuan_ocr
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(hunyuan_ocr PUBLIC ncnn)
target_compile_features(hunyuan_ocr PUBLIC cxx_std_17)
target_compile_definitions(hunyuan_ocr PUBLIC HUNYUAN_OCR_VERSION="${PROJECT_VERSION}")
if(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
    target_compile_definitions(hunyuan_ocr PRIVATE HUNYUAN_OCR_ENABLE_SDPA_PROFILE=1)
endif()

if(HUNYUAN_OCR_STRICT_WARNINGS)
    if(MSVC)
        target_compile_options(hunyuan_ocr PRIVATE /W4)
    else()
        target_compile_options(hunyuan_ocr PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(HUNYUAN_OCR_BUILD_CLI)
    add_executable(hunyuan_ocr_cli src/main.cpp)
    target_link_libraries(hunyuan_ocr_cli PRIVATE hunyuan_ocr)
    if(WIN32)
        target_link_libraries(hunyuan_ocr_cli PRIVATE shell32)
    endif()
    target_compile_features(hunyuan_ocr_cli PRIVATE cxx_std_17)

    if(HUNYUAN_OCR_STRICT_WARNINGS)
        if(MSVC)
            target_compile_options(hunyuan_ocr_cli PRIVATE /W4)
        else()
            target_compile_options(hunyuan_ocr_cli PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()
endif()

if(HUNYUAN_OCR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
