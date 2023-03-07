cmake_minimum_required(VERSION 3.5)
project(lqg_control)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(TARGET_NAME ${PROJECT_NAME}_node)
set(TARGET_LIB_NAME ${PROJECT_NAME}_lib)

# find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
#find_package(Armadillo REQUIRED)
#find_package(LAPACK REQUIRED)

# message generation
ament_auto_generate_code()

# setup targets
ament_auto_add_library(${TARGET_LIB_NAME}
  src/Kalman.cpp
  src/LQR.cpp
  src/lqg_control.cpp
)

ament_auto_add_executable(${TARGET_NAME}
  src/lqg_control_node.cpp
)

#target_link_libraries(${TARGET_NAME} ${ARMADILLO_LIBRARIES})
#ament_target_dependencies(${TARGET_NAME} Armadillo LAPACK)

# export information to downstream packages
ament_auto_package(
  CONFIG_EXTRAS "${PROJECT_NAME}-extras.cmake"
)

# tests
if(AMENT_ENABLE_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  #ament_add_gtest(${PROJECT_NAME}_gtest test/my_test.cpp)
  #target_link_libraries(${PROJECT_NAME}_gtest ${rclcpp_LIBRARIES} ${rmw_connext_cpp_LIBRARIES} ${std_interfaces})

  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
