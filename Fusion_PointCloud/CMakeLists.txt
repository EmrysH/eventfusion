cmake_minimum_required(VERSION 2.8)
project(Fusion_PointCloud)

set(CMAKE_CXX_STANDARD 11)


SET (PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")


find_package(OpenCV REQUIRED)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

find_package(PCL 1.11 REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
        ${PROJECT_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        "/usr/local/include/")
link_directories(
        /home/emrys/CLionProjects/Fusion_PointCloud/lib)


add_executable(Fusion_PointCloud main.cpp)



target_link_libraries(Fusion_PointCloud

        irdirectsdk
        realsense2::realsense2
        CeleX
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}

        )