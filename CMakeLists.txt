cmake_minimum_required(VERSION 3.14)
project(cancer_data_analysis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# TODO: set up xtensor include dirs. Works out of the box on my machine :tm:, so it's low priority
#find_package(xtensor REQUIRED)

include(FetchContent)

FetchContent_Declare(matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
    GIT_TAG origin/master
)

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

include_directories(include)

add_executable(kmeans src/kmeans.cpp)
target_link_libraries(kmeans PUBLIC matplot)

add_executable(linear_svm src/linear_svm.cpp)
target_link_libraries(linear_svm PUBLIC matplot)

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
