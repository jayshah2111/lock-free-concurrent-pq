# Root CMake configuration for lockfree_pq
cmake_minimum_required(VERSION 3.15)
project(lockfree_pq_demo LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Warnings
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Header-only library target
add_library(lockfree_pq INTERFACE)
target_include_directories(lockfree_pq INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Demo executable
add_executable(lockfree_pq_demo src/main.cpp)
target_link_libraries(lockfree_pq_demo PRIVATE lockfree_pq)

# Option: sanitizers
option(ENABLE_SANITIZERS "Enable Address/UB sanitizers" OFF)
if(ENABLE_SANITIZERS)
  target_compile_options(lockfree_pq_demo PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_libraries(lockfree_pq_demo PRIVATE -fsanitize=address,undefined)
endif()

# Install rules
install(TARGETS lockfree_pq_demo RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
