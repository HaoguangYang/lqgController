cmake_minimum_required(VERSION 3.5)
project(lqg_control)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -march=native)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
#find_package(Armadillo REQUIRED)
#find_package(LAPACK REQUIRED)

include_directories(include)

add_executable(lqg_control_node
    src/lqg_control_node.cpp
    src/Kalman.cpp
    src/LQR.cpp
    src/lqg_control.cpp)
#target_link_libraries(lqg_control_node ${ARMADILLO_LIBRARIES})
ament_target_dependencies(lqg_control_node rclcpp std_msgs nav_msgs sensor_msgs Eigen3)# Armadillo LAPACK)
install(
  TARGETS
  lqg_control_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  #config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY
  DESTINATION share/lqg_control/
)

ament_package()
