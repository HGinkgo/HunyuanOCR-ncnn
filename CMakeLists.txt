cmake_minimum_required(VERSION 3.18)

project(HunyuanOCR_ncnn
    VERSION 0.2.0
    LANGUAGES CXX
)

option(HUNYUAN_OCR_BUILD_CLI "Build the hunyuan_ocr_cli executable" ON)
option(HUNYUAN_OCR_BUILD_TESTS "Build HunyuanOCR-ncnn test executables" OFF)
option(HUNYUAN_OCR_STRICT_WARNINGS "Enable compiler warnings for project sources" ON)
option(HUNYUAN_OCR_USE_NCNN_PACKAGE "Try find_package(ncnn CONFIG) before manual ncnn paths" ON)
set(HUNYUAN_OCR_BUILD_TOKENIZER_TEST "AUTO" CACHE STRING "Build tokenizer file parity test: AUTO, ON, or OFF")
set_property(CACHE HUNYUAN_OCR_BUILD_TOKENIZER_TEST PROPERTY STRINGS AUTO ON OFF)

if(HUNYUAN_OCR_USE_NCNN_PACKAGE)
    if(DEFINED ENV{ncnn_DIR} AND NOT ncnn_DIR)
        set(ncnn_DIR "$ENV{ncnn_DIR}" CACHE PATH "Path to ncnnConfig.cmake" FORCE)
    endif()
    find_package(ncnn CONFIG QUIET)
endif()

if(NOT TARGET ncnn)
    set(NCNN_INCLUDE_DIR "" CACHE PATH "Path to the ncnn source include directory, for example /path/to/ncnn/src")
    set(NCNN_BUILD_INCLUDE_DIR "" CACHE PATH "Path to the ncnn generated include directory, for example /path/to/ncnn/build/src")
    set(NCNN_LIBRARY "" CACHE FILEPATH "Path to libncnn.a, libncnn.so, or ncnn.lib")

    if(NOT NCNN_INCLUDE_DIR OR NOT NCNN_BUILD_INCLUDE_DIR OR NOT NCNN_LIBRARY)
        message(FATAL_ERROR
            "ncnn was not found. Either pass -Dncnn_DIR=/path/to/ncnnConfig.cmake "
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
        src/hunyuan_ocr.cpp
        src/generation_config.cpp
        src/image_preprocessor.cpp
        src/model_layout.cpp
        src/multimodal_rope.cpp
        src/prompt_builder.cpp
        src/precise_sdpa.cpp
        src/text_runtime.cpp
        src/tokenizer.cpp
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

if(HUNYUAN_OCR_STRICT_WARNINGS)
    if(MSVC)
        target_compile_options(hunyuan_ocr PRIVATE /W4)
    else()
        target_compile_options(hunyuan_ocr PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(HUNYUAN_OCR_BUILD_TESTS)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(HUNYUAN_OCR_TEST_TOKENIZER_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../models/tokenizer"
        CACHE PATH "Directory containing exported HunyuanOCR tokenizer files for tests"
    )
    string(TOUPPER "${HUNYUAN_OCR_BUILD_TOKENIZER_TEST}" _hunyuan_ocr_build_tokenizer_test)
    if(NOT _hunyuan_ocr_build_tokenizer_test STREQUAL "AUTO"
       AND NOT _hunyuan_ocr_build_tokenizer_test STREQUAL "ON"
       AND NOT _hunyuan_ocr_build_tokenizer_test STREQUAL "OFF")
        message(FATAL_ERROR
            "HUNYUAN_OCR_BUILD_TOKENIZER_TEST must be AUTO, ON, or OFF; "
            "got '${HUNYUAN_OCR_BUILD_TOKENIZER_TEST}'"
        )
    endif()

    set(_hunyuan_ocr_tokenizer_files
        "${HUNYUAN_OCR_TEST_TOKENIZER_DIR}/vocab.txt"
        "${HUNYUAN_OCR_TEST_TOKENIZER_DIR}/merges.txt"
        "${HUNYUAN_OCR_TEST_TOKENIZER_DIR}/special_tokens.json"
    )
    set(_hunyuan_ocr_tokenizer_files_exist TRUE)
    foreach(_hunyuan_ocr_tokenizer_file IN LISTS _hunyuan_ocr_tokenizer_files)
        if(NOT EXISTS "${_hunyuan_ocr_tokenizer_file}")
            set(_hunyuan_ocr_tokenizer_files_exist FALSE)
        endif()
    endforeach()
    set(_hunyuan_ocr_enable_tokenizer_test FALSE)
    if(_hunyuan_ocr_build_tokenizer_test STREQUAL "ON")
        if(NOT _hunyuan_ocr_tokenizer_files_exist)
            message(FATAL_ERROR
                "HUNYUAN_OCR_BUILD_TOKENIZER_TEST=ON but tokenizer files were not found in "
                "${HUNYUAN_OCR_TEST_TOKENIZER_DIR}"
            )
        endif()
        set(_hunyuan_ocr_enable_tokenizer_test TRUE)
    elseif(_hunyuan_ocr_build_tokenizer_test STREQUAL "AUTO" AND _hunyuan_ocr_tokenizer_files_exist)
        set(_hunyuan_ocr_enable_tokenizer_test TRUE)
    endif()

    add_executable(hunyuan_ocr_prompt_builder_test tests/prompt_builder_test.cpp)
    target_link_libraries(hunyuan_ocr_prompt_builder_test PRIVATE hunyuan_ocr)
    target_compile_features(hunyuan_ocr_prompt_builder_test PRIVATE cxx_std_17)

    add_executable(hunyuan_ocr_generation_config_test tests/generation_config_test.cpp)
    target_link_libraries(hunyuan_ocr_generation_config_test PRIVATE hunyuan_ocr)
    target_compile_features(hunyuan_ocr_generation_config_test PRIVATE cxx_std_17)

    add_executable(hunyuan_ocr_text_rope_test tests/text_rope_test.cpp)
    target_link_libraries(hunyuan_ocr_text_rope_test PRIVATE hunyuan_ocr)
    target_compile_features(hunyuan_ocr_text_rope_test PRIVATE cxx_std_17)

    add_executable(hunyuan_ocr_vision_interpolation_test tests/vision_interpolation_test.cpp)
    target_link_libraries(hunyuan_ocr_vision_interpolation_test PRIVATE hunyuan_ocr)
    target_compile_features(hunyuan_ocr_vision_interpolation_test PRIVATE cxx_std_17)

    add_executable(hunyuan_ocr_precise_sdpa_test tests/precise_sdpa_test.cpp)
    target_link_libraries(hunyuan_ocr_precise_sdpa_test PRIVATE hunyuan_ocr)
    target_compile_features(hunyuan_ocr_precise_sdpa_test PRIVATE cxx_std_17)

    add_executable(hunyuan_ocr_multimodal_rope_test tests/multimodal_rope_test.cpp)
    target_link_libraries(hunyuan_ocr_multimodal_rope_test PRIVATE hunyuan_ocr)
    target_compile_features(hunyuan_ocr_multimodal_rope_test PRIVATE cxx_std_17)

    if(_hunyuan_ocr_enable_tokenizer_test)
        add_executable(hunyuan_ocr_tokenizer_test tests/tokenizer_encode_test.cpp)
        target_link_libraries(hunyuan_ocr_tokenizer_test PRIVATE hunyuan_ocr)
        target_compile_features(hunyuan_ocr_tokenizer_test PRIVATE cxx_std_17)
    else()
        message(STATUS "Tokenizer file parity test disabled; tokenizer files not available.")
    endif()

    if(HUNYUAN_OCR_STRICT_WARNINGS)
        if(MSVC)
            target_compile_options(hunyuan_ocr_prompt_builder_test PRIVATE /W4)
            target_compile_options(hunyuan_ocr_generation_config_test PRIVATE /W4)
            if(_hunyuan_ocr_enable_tokenizer_test)
                target_compile_options(hunyuan_ocr_tokenizer_test PRIVATE /W4)
            endif()
        else()
            target_compile_options(hunyuan_ocr_prompt_builder_test PRIVATE -Wall -Wextra -Wpedantic)
            target_compile_options(hunyuan_ocr_generation_config_test PRIVATE -Wall -Wextra -Wpedantic)
            if(_hunyuan_ocr_enable_tokenizer_test)
                target_compile_options(hunyuan_ocr_tokenizer_test PRIVATE -Wall -Wextra -Wpedantic)
            endif()
        endif()
    endif()

    if(_hunyuan_ocr_enable_tokenizer_test)
        add_test(NAME tokenizer_encode
            COMMAND hunyuan_ocr_tokenizer_test
                    "${HUNYUAN_OCR_TEST_TOKENIZER_DIR}"
        )
    endif()
    add_test(NAME prompt_builder
        COMMAND hunyuan_ocr_prompt_builder_test
    )
    add_test(NAME generation_config
        COMMAND hunyuan_ocr_generation_config_test
    )
    add_test(NAME text_rope
        COMMAND hunyuan_ocr_text_rope_test
    )
    add_test(NAME vision_interpolation
        COMMAND hunyuan_ocr_vision_interpolation_test
    )
    add_test(NAME precise_sdpa
        COMMAND hunyuan_ocr_precise_sdpa_test
    )
    add_test(NAME multimodal_rope
        COMMAND hunyuan_ocr_multimodal_rope_test
    )
    add_test(NAME run_regression_prompt_manifest
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_regression_manifest_test.py"
    )
    add_test(NAME extract_tokenizer
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/extract_tokenizer_test.py"
    )
    add_test(NAME export_vision_dynamic
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/export_vision_dynamic_test.py"
    )
    add_test(NAME export_common
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/export_common_test.py"
    )
    add_test(NAME text_decoder_kv
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/text_decoder_kv_test.py"
    )
    add_test(NAME benchmark_parser
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_parser_test.py"
    )
    add_test(NAME run_hf_baseline
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_hf_baseline_test.py"
    )
    add_test(NAME prepare_regression_fixtures
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/prepare_regression_fixtures_test.py"
    )
    add_test(NAME windows_workflow
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/windows_workflow_test.py"
    )
    add_test(NAME atomic_output
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/atomic_output_test.py"
    )
    if(HUNYUAN_OCR_BUILD_CLI)
        add_test(NAME cli_options
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_options_test.py"
                    "$<TARGET_FILE:hunyuan_ocr_cli>"
        )
    endif()
endif()

if(HUNYUAN_OCR_BUILD_CLI)
    add_executable(hunyuan_ocr_cli src/main.cpp)
    target_link_libraries(hunyuan_ocr_cli PRIVATE hunyuan_ocr)
    target_compile_features(hunyuan_ocr_cli PRIVATE cxx_std_17)

    if(HUNYUAN_OCR_STRICT_WARNINGS)
        if(MSVC)
            target_compile_options(hunyuan_ocr_cli PRIVATE /W4)
        else()
            target_compile_options(hunyuan_ocr_cli PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()
endif()
