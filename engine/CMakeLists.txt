cmake_minimum_required(VERSION 3.20)
project(bare_metal_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# --- Header-only deps via FetchContent ---
include(FetchContent)

FetchContent_Declare(
        httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG v0.15.3
)
FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(httplib nlohmann_json)

# --- Engine library (so tests can link it without rebuilding main) ---
add_library(engine_lib
        src/arena.cpp
        src/math.cpp
        src/model.cpp
        src/api.cpp
)
target_include_directories(engine_lib PUBLIC include)
target_link_libraries(engine_lib PUBLIC httplib nlohmann_json::nlohmann_json)

# --- Executable ---
add_executable(bare_metal_inference src/main.cpp)
target_link_libraries(bare_metal_inference PRIVATE engine_lib)

enable_testing()


add_executable(test_arena tests/test_arena.cpp)
target_link_libraries(test_arena PRIVATE engine_lib)
add_test(NAME test_arena COMMAND test_arena)

add_executable(test_math tests/test_math.cpp)
target_link_libraries(test_math PRIVATE engine_lib)
add_test(NAME test_math COMMAND test_math)

add_executable(test_inference tests/test_inference.cpp)
target_link_libraries(test_inference PRIVATE engine_lib)
add_test(NAME test_inference COMMAND test_inference)
set_tests_properties(test_inference PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)