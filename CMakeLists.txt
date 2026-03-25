cmake_minimum_required(VERSION 3.16)
project(vecstore LANGUAGES CXX)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(VECSTORE_BUILD_BENCH "Build benchmark targets" ON)


# source
add_library(vecstore_vecmath STATIC
    src/vecmath/dot_scalar.cpp
)
add_library(vecstore::vecmath ALIAS vecstore_vecmath)
target_include_directories(vecstore_vecmath PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(vecstore_vecmath PUBLIC cxx_std_20)
target_compile_options(vecstore_vecmath PRIVATE -Wall -Wextra -Wpedantic -O2)


# ensure compiler vectorization disabled to keep the scalar baseline truly scalar
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/vecmath/dot_scalar.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-tree-vectorize")
endif()


# tests 
enable_testing()

add_executable(test_dot
    tests/test_dot.cpp
)
target_link_libraries(test_dot PRIVATE vecstore::vecmath)
add_test(NAME vecstore_test_dot COMMAND test_dot)


# benchmark
if (VECSTORE_BUILD_BENCH)
    add_executable(bench_dot
        bench/bench_dot.cpp
    )
    target_link_libraries(bench_dot PRIVATE vecstore::vecmath)
endif()
