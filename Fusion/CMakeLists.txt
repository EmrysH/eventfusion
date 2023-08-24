cmake_minimum_required(VERSION 3.25)

project(Fusion)

set(CMAKE_CXX_STANDARD 11)

SET (PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
find_package(OpenCV REQUIRED)
find_package(OpenGL)
#find_package(PCL)
find_package(realsense2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Eigen3 REQUIRED)



include_directories(
        ${PROJECT_INCLUDE_DIR}
        "/usr/local/include/")
link_directories(
        /home/emrys/CLionProjects/Fusion/lib)


set(CMAKE_CXX_STANDARD 17)

add_executable(Fusion main.cpp)

target_link_libraries(Fusion

        irdirectsdk
        realsense2::realsense2
        OpenGL::OpenGL
        glfw
        CeleX
        #        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARY}
        )