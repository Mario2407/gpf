cmake_minimum_required(VERSION 3.5)
project(gpf)

# Default to C11
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
endif()
# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Testing and linting
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ament_cmake
find_package(ament_cmake REQUIRED)
# ROS2 packages
find_package(rclcpp REQUIRED)
find_package(pcl_conversions REQUIRED)
# ROS2 interfaces
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometric_primitive_msgs REQUIRED)

# PCL
find_package(PCL REQUIRED)

# Include directories
include_directories(include ${PCL_INCLUDE_DIRS})

# Executables
set(EXECUTABLE_0 gpf)
set(EXECUTABLE_1 ransac_cylinder)
set(EXECUTABLE_2 show_normals)

# Executable 0
add_executable(${EXECUTABLE_0} src/${EXECUTABLE_0}.cpp)
target_link_libraries(${EXECUTABLE_0} ${PCL_LIBRARIES})
ament_target_dependencies(${EXECUTABLE_0} rclcpp pcl_conversions sensor_msgs
                          visualization_msgs geometric_primitive_msgs)

# Executable 1
add_executable(${EXECUTABLE_1} src/${EXECUTABLE_1}.cpp)
target_link_libraries(${EXECUTABLE_1} ${PCL_LIBRARIES})
ament_target_dependencies(${EXECUTABLE_1} rclcpp pcl_conversions sensor_msgs
                          visualization_msgs geometric_primitive_msgs)


# Executable 2
add_executable(${EXECUTABLE_2} src/${EXECUTABLE_2}.cpp)
target_link_libraries(${EXECUTABLE_2} ${PCL_LIBRARIES})
ament_target_dependencies(${EXECUTABLE_2} rclcpp pcl_conversions sensor_msgs
                          visualization_msgs geometric_primitive_msgs)

# Install targets
install(TARGETS ${EXECUTABLE_0} DESTINATION lib/${PROJECT_NAME})
install(TARGETS ${EXECUTABLE_1} DESTINATION lib/${PROJECT_NAME})
install(TARGETS ${EXECUTABLE_2} DESTINATION lib/${PROJECT_NAME})
# Install directories
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

# Setup the project
ament_package()
