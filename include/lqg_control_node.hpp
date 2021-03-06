/* Copyright 2021 Haoguang Yang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef LQG_CONTROL_NODE_HPP
#define LQG_CONTROL_NODE_HPP

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <math.h>
#include <utility>
#include <vector>
#include <map>
#include <eigen3/Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "lqg_control.hpp"

namespace control
{

class LqgControlNode : public rclcpp::Node
{
  public:
  
    LqgControlNode(const rclcpp::NodeOptions& options);

    std_msgs::msg::Float64MultiArray desiredStates;

    std_msgs::msg::Float64MultiArray stateError;

    std_msgs::msg::Float64MultiArray commandVector;

    std_msgs::msg::Float64MultiArray measurements;

    std_msgs::msg::Float64MultiArray measurementsCov;

  protected:

    void registerStateSpaceIO();

    void updateMeasurement(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void updateMeasurementCov(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void updateDesiredState(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void controlCallback();

    void publishCommand();

    void publishDebugSignals();

    rcl_interfaces::msg::SetParametersResult paramUpdateCallback(const std::vector<rclcpp::Parameter> &parameters);

    bool mute_, debug_, gaussian_, discretize_, u_fb_;

    double dt_;

    control::LqgControl controller_;

    rclcpp::QoS *qos_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pubCmdVect_, pubStateErrVect_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subMeasVect_, subCovMat_, subDesVect_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr paramUpdate_handle_;

    int XDoF_, UDoF_, YDoF_;

    std::vector<double> aM_, bM_, cM_, dM_, qM_, rM_, sdM_, snM_, p0M_, nM_;
  
  private:
    enum params{
      MUTE,
      DEBUG
    };
    std::map<std::string, params> paramMap_={
      {"mute", MUTE},
      {"debug", DEBUG}
    };

}; // end of class

} // end of namespace

#endif
