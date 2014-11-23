
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(PROJECTNAME collector)
PROJECT(${PROJECTNAME})


find_package(YARP REQUIRED)
find_package(ICUB REQUIRED)
# find_package(Gurls)
find_package(ICUBcontrib REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH} ${ICUBCONTRIB_MODULE_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${YARP_MODULE_PATH} ${ICUB_MODULE_PATH})

find_package(OpenCV)

# include(iCubOptions)
# include(iCubHelpers)
include(ICUBcontribHelpers)
include(ICUBcontribOptions)
#   add_definitions(-DGURLS_AVAILABLE=1)

file(GLOB source *.cpp)
file(GLOB header *.h)

source_group("Source Files" FILES ${source})
source_group("Header Files" FILES ${header})

include_directories(${YARP_INCLUDE_DIRS}
                    ${ICUB_INCLUDE_DIRS}
)


add_executable(${PROJECTNAME} ${source} ${header})

target_link_libraries(${PROJECTNAME} icubmod ${YARP_LIBRARIES} ${OpenCV_LIBRARIES})

# install(TARGETS ${PROJECTNAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
install(TARGETS ${PROJECTNAME} RUNTIME DESTINATION bin)
# add_subdirectory(bkp)