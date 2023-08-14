CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

set(CMAKE_CXX_STANDARD 11)


SET (PROJECT_NAME "fastMCD")
project(${PROJECT_NAME})

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
find_package(OpenCV REQUIRED) # core imgproc highgui flann video calib3d nonfree objdetect)

message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

set(HDRS )
file(GLOB "FILES" "src/*.cpp")
set( SRCS src/main.cpp ${FILES})

add_executable(${PROJECT_NAME} ${SRCS} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${PROTOBUF_LIBRARY} )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

include_directories(${PROJECT_SOURCE_DIR}/. ${PROJECT_SOURCE_DIR}/src)
# add OpenCV include directories
include_directories(${OpenCV_INCLUDE_DIRS})
