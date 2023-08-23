cmake_minimum_required(VERSION 3.25)
project(Event_Vector)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(/home/emrys/CLionProjects/Event_Vector/include/celex5)
link_directories(/home/emrys/CLionProjects/Event_Vector/lib)

add_executable(Event_Vector main.cpp)
target_link_libraries(Event_Vector ${OpenCV_LIBS} CeleX)



