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

#include "lqg_control/lqg_control_node.hpp"

namespace control
{
LqgControlNode::LqgControlNode(const rclcpp::NodeOptions& options=rclcpp::NodeOptions()):
  Node("LqgControlNode", options)
{
  qos_ = new rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 1));
  qos_->best_effort();

  // Declare Parameters
  mute_ = this->declare_parameter<bool>("mute", false);
  debug_ = this->declare_parameter<bool>("debug", true);
  // when gaussian estimator is disabled, this controller degenerates to a classic LQR controller.
  gaussian_ = this->declare_parameter<bool>("enable_gaussian_estiamtor", true);
  dt_ = this->declare_parameter<double>("control_time_step", 0.01);
  // Control Parameter
  XDoF_ = this->declare_parameter<int>("input_dims", 10);
  UDoF_ = this->declare_parameter<int>("output_dims", 3);
  YDoF_ = this->declare_parameter<int>("observable_dims", 10);
  aM_ = this->declare_parameter<std::vector<double>>("A_matrix");
  bM_ = this->declare_parameter<std::vector<double>>("B_matrix");
  discretize_ = this->declare_parameter<bool>("A_B_matrices_are_continuous", false);
  cM_ = this->declare_parameter<std::vector<double>>("C_matrix");
  dM_ = this->declare_parameter<std::vector<double>>("D_matrix");
  qM_ = this->declare_parameter<std::vector<double>>("Q_matrix");
  rM_ = this->declare_parameter<std::vector<double>>("R_matrix");
  u_fb_ = this->declare_parameter<bool>("processed_control_feedback", true);
  sdM_ = this->declare_parameter<std::vector<double>>("ProcessDisturbanceCov_matrix");
  snM_ = this->declare_parameter<std::vector<double>>("MeasurementNoiseCov_matrix");
  p0M_ = this->declare_parameter<std::vector<double>>("InitialStateCov_matrix");
  nM_ = this->declare_parameter<std::vector<double>>("N_matrix");

  // Initialize Controller
  if (gaussian_)
  {
    this->controller_ = control::LqgControl(XDoF_, UDoF_, YDoF_, discretize_, u_fb_, dt_,
                                  aM_, bM_, cM_, dM_, qM_, rM_, nM_, sdM_, snM_, p0M_);
  } else {
    this->controller_ = control::LqgControl(XDoF_, UDoF_, YDoF_, discretize_, u_fb_, dt_,
                                  aM_, bM_, cM_, dM_, qM_, rM_, nM_);
  }

  // handle parameter updates
  this->paramUpdate_handle_ = this->add_on_set_parameters_callback(
            std::bind(&LqgControlNode::paramUpdateCallback, this, std::placeholders::_1));

  // Create Callback Timers
  int timer_ms_ = static_cast<int>(this->dt_ * 1000);
  this->control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(timer_ms_), std::bind(&LqgControlNode::controlCallback, this));

  this->registerStateSpaceIO();
  // Debug Publisher
  this->pubStateErrVect_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("state_error", 10);
}

void LqgControlNode::registerStateSpaceIO(){
  // generic publishers
  this->pubCmdVect_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("command_vector", 10);
  // generic subscribers
  this->subMeasVect_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    "measurements", *qos_, std::bind(&LqgControlNode::updateMeasurement, this, std::placeholders::_1));
  this->subCovMat_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    "measurements_cov", *qos_, std::bind(&LqgControlNode::updateMeasurementCov, this, std::placeholders::_1));
  this->subDesVect_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    "desired_states", *qos_, std::bind(&LqgControlNode::updateDesiredState, this, std::placeholders::_1));
}

rcl_interfaces::msg::SetParametersResult
  LqgControlNode::paramUpdateCallback(const std::vector<rclcpp::Parameter> &parameters)
{
  //this->auto_enabled_ = this->get_parameter("auto_enabled").as_bool();
  //TODO: what else needs to be dynamically updated?
  //bool updateSystem = false;
  bool updateController = false;
  //double tmp = 0.;

  // cycle through the list of set parameters and update temporarily-stored parameter table.
  for (const auto &param: parameters)
  {
    switch (paramMap_[param.get_name()]){

      case MUTE:
        this->mute_ = param.as_bool();
        break;

      case DEBUG:
        this->debug_ = param.as_bool();
        break;

      default:
        break;

    }
  }

  // at the end of each cycle, we need to check if the set of temporarily-stored parameters meet format consistency.
  // If the format consistency is met, we automatically update the controller.

  if (updateController){
    // TODO: need to refresh the system matrices
    //buildLinearizedSystem();
  }

  // reply to the ROS2 parameter call.
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;
}

void LqgControlNode::updateMeasurement(const std_msgs::msg::Float64MultiArray::SharedPtr msg){
  this->controller_.updateMeasurement(msg);
}

void LqgControlNode::updateMeasurementCov(const std_msgs::msg::Float64MultiArray::SharedPtr msg){
  this->controller_.updateMeasurementCov(msg);
}

void LqgControlNode::updateDesiredState(const std_msgs::msg::Float64MultiArray::SharedPtr msg){
  this->controller_.updateDesiredState(msg);
}

void LqgControlNode::controlCallback()
{
  // call controller->controlCallback and update raw commands.
  this->controller_.controlCallback(this->get_logger());
  if (!this->mute_)
    this->publishCommand();
  if (this->debug_)
    this->publishDebugSignals();
}

void LqgControlNode::publishCommand()
{
  Eigen::VectorXd u = this->controller_.getControl();
  // unpack Eigen::Vector into Float64 array message
  this->commandVector.data.clear();
  for (size_t i = 0; i < (size_t)(u.size()); i ++)
  {
    this->commandVector.data.push_back( u(i) );
  }
  // publish message
  this->pubCmdVect_->publish(this->commandVector);
}

void LqgControlNode::publishDebugSignals()
{
  //pubLookaheadError_->publish(this->lookahead_error);
  //pubLatError_->publish(this->lat_error);
  //pubCurvature_->publish(this->curvature);
  Eigen::VectorXd err = this->controller_.currentError();
  this->stateError.data.clear();
  for (size_t i = 0; i < (size_t)(err.size()); i ++)
  {
    this->stateError.data.push_back( err(i) );
  }
  this->pubStateErrVect_->publish(this->stateError);
}

}// end namespace control

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<control::LqgControlNode>());
  rclcpp::shutdown();
  return 0;
}
