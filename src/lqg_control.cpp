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

#include <lqg_control.hpp>

namespace control
{
void LqgControl::LqgControl(int& XDoF, int& UDoF, int& YDoF,
                            bool& discretize, bool& u_feedback, double& dt,
                            std::vector<double>& A, std::vector<double>& B,
                            std::vector<double>& C, std::vector<double>& D,
                            std::vector<double>& Q, std::vector<double>& R,
                            std::vector<double>& N,
                            std::vector<double>& Sd, std::vector<double>& Sn,
                            std::vector<double>& P0):
                LqrControl(XDoF, UDoF, YDoF, discretize, u_feedback, dt, A, B, C, D, Q, R, N)
{
  if (Sd.size()!=XDoF*XDoF)
    throw std::runtime_error("Disturbance covariance matrix Sd is ill-formed!");
  if (Sn.size()!=YDoF*YDoF)
    throw std::runtime_error("Measurement covariance matrix Sn is ill-formed!");
  matrixPack(Sd, this->sigmaDisturbance_);
  matrixPack(Sn, this->sigmaMeasurements_);
  Eigen::MatrixXd stateCov;
  stateCov = Eigen::MatrixXd::Zero(XDoF, XDoF);
  matrixPack(Sd, stateCov);
  
  this->optimal_state_estimate = KalmanFilter(dt, this->A_, this->B_, this->C_, 
                                  this->sigmaDisturbance_, this->sigmaMeasurements_, stateCov);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> LqgControl::getPrediction(){
  std::lock_guard<std::mutex> l(_mtx);
  if (this->predicted_)
    return std::make_pair(this->Y_, this->sigmaMeasurements_);
  this->predicted_ = true;
  std::tie(this->Y_, this->sigmaMeasurements_) = this->optimal_state_estimate.predict(this->U_act_);
  return std::make_pair(this->Y_, this->sigmaMeasurements_);
}

void LqgControl::updateMeasurementCov(const std_msgs::msg::Float64MultiArray::SharedPtr msg){
  // Assuming all elements in the msg array are row-major valid readings.
  if (msg->data.size() == this->YDoF_ * this->YDoF_)
  {
    // Full matrix notation
    std::lock_guard<std::mutex> l(_mtx);
    matrixPack(msg->data, this->sigmaMeasurements_);
  }
  else if (msg->data.size() == (this->YDoF_ + 1) * this->YDoF_ / 2){
    std::lock_guard<std::mutex> l(_mtx);
    // Upper triangular matrix notation
    size_t ind = 0;
    for (size_t i = 0; i < this->YDoF_; i++){
      for (size_t j = 0; j < i; j++){
        this->sigmaMeasurements_(i,j) = this->sigmaMeasurements_(j,i);
      }
      for (size_t j = i; j < this->YDoF_; j++){
        this->sigmaMeasurements_(i,j) = msg->data[ind];
        ind ++;
      }
    }
  }
}

void LqgControl::updateMeasurementCov(Eigen::MatrixXd& cov){
  if (cov.cols()==this->YDoF_ && cov.rows()==this->YDoF_)
  {
    std::lock_guard<std::mutex> l(_mtx);
    this->sigmaMeasurements_ = cov;
  }
}

void LqgControl::controlCallback(rclcpp::Logger& logger)
{
  rclcpp::Time control_time = rclcpp::Clock().now();
  rclcpp::Duration time_diff = control_time - this->state_update_time_;
  double state_dt = static_cast<double>(time_diff.seconds()) +
              static_cast<double>(time_diff.nanoseconds())*1e-9;
  time_diff = control_time - this->last_control_time_;
  double control_dt = static_cast<double>(time_diff.seconds()) +
              static_cast<double>(time_diff.nanoseconds())*1e-9;
  this->last_control_time_ = control_time;

  if ( state_dt < this->state_timeout_ && this->optimal_state_estimate.isInitialized() )
  {
    // summarize asynchronous updates of the state measurements.
    this->optimal_state_estimate.update_time_variant_R(this->Y_, this->U_act_, this->sigmaMeasurements_, dt_);
    this->U_ = this->optimal_controller.calculateControl(this->optimal_state_estimate.state(), this->X_des_);
    
    //std::cout << "Estimated State: \n" << this->x_est_.transpose() << "\n";
    /* <<
                 "Desired State: \n" << this->x_desired_.transpose() << "\n" <<
                 "Observations: \n" << this->Y_.transpose() << "\n" <<
                 "Calculated Best Command: \n" << this->u_raw_.transpose() << "\n" <<
                 "\n----------------------------------------\n";
                 */
  }
  else
  {
    this->setCmdToZeros();
    if ( state_dt >= this->state_timeout_ ){
      RCLCPP_DEBUG(logger,
                    "State observation age: %f seconds is too stale! (timeout = %f s)\n",
                    state_dt, this->state_timeout_);
    } else {
      RCLCPP_DEBUG(logger, "Controller waiting for sensor messages to initialize...\n");
      if (!this->optimal_state_estimate.isInitialized())
        initializeSystemAndControls();
      if (this->optimal_state_estimate.isInitialized() )
        RCLCPP_DEBUG(logger, "Controller State Estimator Initialized!\n");
    }
  }

  if (!this->u_feedback_)
    this->U_act_ = this->U_;
  
}

} 
