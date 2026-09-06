cmake_minimum_required(VERSION 3.20)
project(bare_metal_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# Vendored single-header libs (no FetchContent, no network at build time)
add_library(httplib INTERFACE)
target_include_directories(httplib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Engine sources — just the .cpp files. The headers are picked up
# automatically via target_include_directories.
add_library(engine_lib
        src/model.cpp
        src/api.cpp
        src/httplib_impl.cpp
)
target_include_directories(engine_lib PUBLIC include)
target_link_libraries(engine_lib PUBLIC httplib nlohmann_json)

# Executable
add_executable(bare_metal_inference src/main.cpp)
target_link_libraries(bare_metal_inference PRIVATE engine_lib)

# Tests
enable_testing()
add_executable(test_arena      tests/test_arena.cpp)
add_executable(test_math       tests/test_math.cpp)
add_executable(test_inference  tests/test_inference.cpp)
foreach(t test_arena test_math test_inference)
    target_link_libraries(${t} PRIVATE engine_lib)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
set_tests_properties(test_inference PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)