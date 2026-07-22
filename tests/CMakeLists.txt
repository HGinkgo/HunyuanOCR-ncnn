set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(HUNYUAN_OCR_TEST_TOKENIZER_DIR
    "${PROJECT_SOURCE_DIR}/../models/tokenizer"
    CACHE PATH "Directory containing exported HunyuanOCR tokenizer files for tests"
)
set(HUNYUAN_OCR_TEST_DYNAMIC_VISION_DIR
    ""
    CACHE PATH "Optional directory containing vision.ncnn.param/bin and pos_embed.bin for dynamic runtime tests"
)
set(HUNYUAN_OCR_TEST_RUNTIME_MODEL_DIR
    ""
    CACHE PATH "Optional packaged model directory for high-level runtime integration tests"
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

add_executable(hunyuan_ocr_api_test ${CMAKE_CURRENT_SOURCE_DIR}/hunyuan_ocr_api_test.cpp)
target_link_libraries(hunyuan_ocr_api_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_api_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_runtime_integration_test
    ${CMAKE_CURRENT_SOURCE_DIR}/hunyuan_ocr_runtime_integration_test.cpp
)
target_link_libraries(hunyuan_ocr_runtime_integration_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_runtime_integration_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_batch_jsonl_test ${CMAKE_CURRENT_SOURCE_DIR}/batch_jsonl_test.cpp)
target_include_directories(hunyuan_ocr_batch_jsonl_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(hunyuan_ocr_batch_jsonl_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_batch_jsonl_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_prompt_builder_test ${CMAKE_CURRENT_SOURCE_DIR}/prompt_builder_test.cpp)
target_link_libraries(hunyuan_ocr_prompt_builder_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_prompt_builder_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_generation_config_test ${CMAKE_CURRENT_SOURCE_DIR}/generation_config_test.cpp)
target_link_libraries(hunyuan_ocr_generation_config_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_generation_config_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_text_rope_test ${CMAKE_CURRENT_SOURCE_DIR}/text_rope_test.cpp)
target_link_libraries(hunyuan_ocr_text_rope_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_text_rope_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_vision_interpolation_test ${CMAKE_CURRENT_SOURCE_DIR}/vision_interpolation_test.cpp)
target_link_libraries(hunyuan_ocr_vision_interpolation_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_vision_interpolation_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_vision_dynamic_runtime_test
    ${CMAKE_CURRENT_SOURCE_DIR}/vision_dynamic_runtime_test.cpp
)
target_link_libraries(hunyuan_ocr_vision_dynamic_runtime_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_vision_dynamic_runtime_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_vision_vulkan_policy_test ${CMAKE_CURRENT_SOURCE_DIR}/vision_vulkan_policy_test.cpp)
target_include_directories(hunyuan_ocr_vision_vulkan_policy_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(hunyuan_ocr_vision_vulkan_policy_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_vision_vulkan_policy_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_precise_sdpa_test ${CMAKE_CURRENT_SOURCE_DIR}/precise_sdpa_test.cpp)
target_include_directories(hunyuan_ocr_precise_sdpa_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(hunyuan_ocr_precise_sdpa_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_precise_sdpa_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_multimodal_rope_test ${CMAKE_CURRENT_SOURCE_DIR}/multimodal_rope_test.cpp)
target_link_libraries(hunyuan_ocr_multimodal_rope_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_multimodal_rope_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_dflash_runtime_test ${CMAKE_CURRENT_SOURCE_DIR}/dflash_runtime_test.cpp)
target_link_libraries(hunyuan_ocr_dflash_runtime_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_dflash_runtime_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_memory_lifecycle_test ${CMAKE_CURRENT_SOURCE_DIR}/memory_lifecycle_test.cpp)
target_link_libraries(hunyuan_ocr_memory_lifecycle_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_memory_lifecycle_test PRIVATE cxx_std_17)
if(WIN32)
    target_link_libraries(hunyuan_ocr_memory_lifecycle_test PRIVATE psapi)
endif()

add_executable(hunyuan_ocr_mapped_model_file_test
    ${CMAKE_CURRENT_SOURCE_DIR}/mapped_model_file_test.cpp
)
target_include_directories(hunyuan_ocr_mapped_model_file_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(hunyuan_ocr_mapped_model_file_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_mapped_model_file_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_kv_cache_lifecycle_test
    ${CMAKE_CURRENT_SOURCE_DIR}/kv_cache_lifecycle_test.cpp
)
target_include_directories(hunyuan_ocr_kv_cache_lifecycle_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(hunyuan_ocr_kv_cache_lifecycle_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_kv_cache_lifecycle_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_decoder_microbench_parser_test
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder_microbench_parser_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder_microbench.cpp
)
target_include_directories(hunyuan_ocr_decoder_microbench_parser_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(hunyuan_ocr_decoder_microbench_parser_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_decoder_microbench_parser_test PRIVATE cxx_std_17)
if(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
    target_compile_definitions(hunyuan_ocr_decoder_microbench_parser_test PRIVATE
        HUNYUAN_OCR_ENABLE_SDPA_PROFILE=1)
endif()

# Model-backed microbenchmark executable: build only under tests, never CTest.
add_executable(hunyuan_ocr_decoder_microbench
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder_microbench_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder_microbench.cpp
)
target_include_directories(hunyuan_ocr_decoder_microbench PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(hunyuan_ocr_decoder_microbench PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_decoder_microbench PRIVATE cxx_std_17)
if(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
    target_compile_definitions(hunyuan_ocr_decoder_microbench PRIVATE
        HUNYUAN_OCR_ENABLE_SDPA_PROFILE=1)
endif()

# Model-backed load probe: build only under tests, never register as CTest.
add_executable(hunyuan_ocr_model_load_probe
    ${CMAKE_CURRENT_SOURCE_DIR}/model_load_probe.cpp
)
target_link_libraries(hunyuan_ocr_model_load_probe PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_model_load_probe PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_native_innerproduct_m1_test
    ${CMAKE_CURRENT_SOURCE_DIR}/native_innerproduct_m1_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/native_innerproduct_m1.cpp
)
target_include_directories(hunyuan_ocr_native_innerproduct_m1_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(hunyuan_ocr_native_innerproduct_m1_test PRIVATE ncnn)
target_compile_features(hunyuan_ocr_native_innerproduct_m1_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_native_innerproduct_m1_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/native_innerproduct_m1_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/native_innerproduct_m1.cpp
)
target_include_directories(hunyuan_ocr_native_innerproduct_m1_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(hunyuan_ocr_native_innerproduct_m1_benchmark PRIVATE ncnn)
target_compile_features(hunyuan_ocr_native_innerproduct_m1_benchmark PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_text_decoder_aux_test ${CMAKE_CURRENT_SOURCE_DIR}/text_decoder_aux_test.cpp)
target_link_libraries(hunyuan_ocr_text_decoder_aux_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_text_decoder_aux_test PRIVATE cxx_std_17)
target_include_directories(hunyuan_ocr_text_decoder_aux_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")

add_executable(hunyuan_ocr_text_decoder_vulkan_session_test
    ${CMAKE_CURRENT_SOURCE_DIR}/text_decoder_vulkan_session_test.cpp)
target_link_libraries(hunyuan_ocr_text_decoder_vulkan_session_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_text_decoder_vulkan_session_test PRIVATE cxx_std_17)
target_include_directories(hunyuan_ocr_text_decoder_vulkan_session_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")

add_executable(hunyuan_ocr_dflash_probe_integration_test
    ${CMAKE_CURRENT_SOURCE_DIR}/dflash_probe_integration_test.cpp
)
target_link_libraries(hunyuan_ocr_dflash_probe_integration_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_dflash_probe_integration_test PRIVATE cxx_std_17)

add_executable(hunyuan_ocr_utf8_test ${CMAKE_CURRENT_SOURCE_DIR}/utf8_test.cpp)
target_link_libraries(hunyuan_ocr_utf8_test PRIVATE hunyuan_ocr)
target_compile_features(hunyuan_ocr_utf8_test PRIVATE cxx_std_17)

set(_hunyuan_ocr_enable_utf8_cpp20_test FALSE)
if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(_hunyuan_ocr_enable_utf8_cpp20_test TRUE)
    add_executable(hunyuan_ocr_utf8_cpp20_test
        ${CMAKE_CURRENT_SOURCE_DIR}/utf8_test.cpp
        ${PROJECT_SOURCE_DIR}/src/utf8.cpp
    )
    target_include_directories(hunyuan_ocr_utf8_cpp20_test PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
    )
    set_target_properties(hunyuan_ocr_utf8_cpp20_test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

if(_hunyuan_ocr_enable_tokenizer_test)
    add_executable(hunyuan_ocr_tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_encode_test.cpp)
    target_link_libraries(hunyuan_ocr_tokenizer_test PRIVATE hunyuan_ocr)
    target_compile_features(hunyuan_ocr_tokenizer_test PRIVATE cxx_std_17)
else()
    message(STATUS "Tokenizer file parity test disabled; tokenizer files not available.")
endif()

if(HUNYUAN_OCR_STRICT_WARNINGS)
    if(MSVC)
        target_compile_options(hunyuan_ocr_prompt_builder_test PRIVATE /W4)
        target_compile_options(hunyuan_ocr_generation_config_test PRIVATE /W4)
        target_compile_options(hunyuan_ocr_utf8_test PRIVATE /W4)
        if(_hunyuan_ocr_enable_utf8_cpp20_test)
            target_compile_options(hunyuan_ocr_utf8_cpp20_test PRIVATE /W4)
        endif()
        if(_hunyuan_ocr_enable_tokenizer_test)
            target_compile_options(hunyuan_ocr_tokenizer_test PRIVATE /W4)
        endif()
    else()
        target_compile_options(hunyuan_ocr_prompt_builder_test PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(hunyuan_ocr_generation_config_test PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(hunyuan_ocr_utf8_test PRIVATE -Wall -Wextra -Wpedantic)
        if(_hunyuan_ocr_enable_utf8_cpp20_test)
            target_compile_options(hunyuan_ocr_utf8_cpp20_test PRIVATE -Wall -Wextra -Wpedantic)
        endif()
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
add_test(NAME hunyuan_ocr_api
    COMMAND hunyuan_ocr_api_test
)
if(HUNYUAN_OCR_TEST_RUNTIME_MODEL_DIR)
    add_test(NAME runtime_integration
        COMMAND hunyuan_ocr_runtime_integration_test
                "${HUNYUAN_OCR_TEST_RUNTIME_MODEL_DIR}"
                "${PROJECT_SOURCE_DIR}/examples/images/hf_demo_tools-dark.png"
    )
endif()
add_test(NAME batch_jsonl
    COMMAND hunyuan_ocr_batch_jsonl_test
)
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
if(HUNYUAN_OCR_TEST_DYNAMIC_VISION_DIR)
    foreach(_hunyuan_ocr_dynamic_vision_file
            vision.ncnn.param vision.ncnn.bin pos_embed.bin)
        if(NOT EXISTS
           "${HUNYUAN_OCR_TEST_DYNAMIC_VISION_DIR}/${_hunyuan_ocr_dynamic_vision_file}")
            message(FATAL_ERROR
                "HUNYUAN_OCR_TEST_DYNAMIC_VISION_DIR is missing "
                "${_hunyuan_ocr_dynamic_vision_file}: "
                "${HUNYUAN_OCR_TEST_DYNAMIC_VISION_DIR}"
            )
        endif()
    endforeach()
    add_test(NAME vision_dynamic_runtime
        COMMAND hunyuan_ocr_vision_dynamic_runtime_test
                "${HUNYUAN_OCR_TEST_DYNAMIC_VISION_DIR}"
    )
else()
    message(STATUS "Dynamic vision runtime test disabled; model files not available.")
endif()
add_test(NAME vision_vulkan_policy
    COMMAND hunyuan_ocr_vision_vulkan_policy_test
)
add_test(NAME precise_sdpa
    COMMAND hunyuan_ocr_precise_sdpa_test
)
add_test(NAME multimodal_rope
    COMMAND hunyuan_ocr_multimodal_rope_test
)
add_test(NAME dflash_runtime
    COMMAND hunyuan_ocr_dflash_runtime_test
)
add_test(NAME memory_lifecycle
    COMMAND hunyuan_ocr_memory_lifecycle_test
)
add_test(NAME mapped_model_file
    COMMAND hunyuan_ocr_mapped_model_file_test
)
add_test(NAME kv_cache_lifecycle
    COMMAND hunyuan_ocr_kv_cache_lifecycle_test
)
add_test(NAME decoder_microbench_parser
    COMMAND hunyuan_ocr_decoder_microbench_parser_test
)
add_test(NAME native_innerproduct_m1
    COMMAND hunyuan_ocr_native_innerproduct_m1_test
)
add_test(NAME text_decoder_aux
    COMMAND hunyuan_ocr_text_decoder_aux_test
)
add_test(NAME text_decoder_vulkan_session
    COMMAND hunyuan_ocr_text_decoder_vulkan_session_test
)
add_test(NAME utf8
    COMMAND hunyuan_ocr_utf8_test
)
if(_hunyuan_ocr_enable_utf8_cpp20_test)
    add_test(NAME utf8_cpp20
        COMMAND hunyuan_ocr_utf8_cpp20_test
    )
endif()
add_test(NAME run_regression_prompt_manifest
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/run_regression_manifest_test.py"
)
add_test(NAME extract_tokenizer
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/extract_tokenizer_test.py"
)
add_test(NAME export_vision_dynamic
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/export_vision_dynamic_test.py"
)
add_test(NAME export_common
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/export_common_test.py"
)
add_test(NAME export_dflash
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/export_dflash_test.py"
)
add_test(NAME text_decoder_kv
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/text_decoder_kv_test.py"
)
add_test(NAME patch_innerproduct_m1
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/patch_innerproduct_m1_test.py"
)
add_test(NAME benchmark_parser
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_parser_test.py"
)
add_test(NAME run_hf_baseline
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/run_hf_baseline_test.py"
)
add_test(NAME prepare_regression_fixtures
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/prepare_regression_fixtures_test.py"
)
add_test(NAME windows_workflow
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/windows_workflow_test.py"
)
add_test(NAME linux_workflow
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/linux_workflow_test.py"
)
add_test(NAME atomic_output
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/atomic_output_test.py"
)
add_test(NAME release_metadata
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/release_metadata_test.py"
)
add_test(NAME ncnn_patch_bundle
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/ncnn_patch_bundle_test.py"
)
add_test(NAME artifact_paths
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/artifact_paths_test.py"
)
add_test(NAME package_model
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/package_model_test.py"
)
add_test(NAME quickstart
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/quickstart_test.py"
)
if(HUNYUAN_OCR_BUILD_CLI)
    add_test(NAME windows_utf8_integration
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/windows_utf8_integration_test.py"
                "$<TARGET_FILE:hunyuan_ocr_cli>"
    )
    add_test(NAME cli_options
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cli_options_test.py"
                "$<TARGET_FILE:hunyuan_ocr_cli>"
    )
endif()
if(HUNYUAN_OCR_BUILD_EXAMPLES)
    add_test(NAME ocr_example
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/ocr_example_test.py"
                "$<TARGET_FILE:hunyuan_ocr_example>"
    )
endif()

get_property(_hunyuan_ocr_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_hunyuan_ocr_tests} PROPERTIES
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
)
