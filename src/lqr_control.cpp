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

#include <lqr_control.hpp>

namespace control
{
void LqrControl::LqrControl(int& XDoF, int& UDoF, int& YDoF,
                            bool& discretize, bool& u_feedback, double& dt,
                            std::vector<double>& A, std::vector<double>& B,
                            std::vector<double>& C, std::vector<double>& D,
                            std::vector<double>& Q, std::vector<double>& R,
                            std::vector<double>& N):
                XDoF_(XDoF), UDoF_(UDoF), YDoF_(YDoF), dt_(dt), u_feedback_(u_feedback)
{
  // clean std::vector parameters (check !NULL and check size)
  if (A.size()!=XDoF*XDoF)
    throw std::runtime_error("State matrix A size is ill-formed!");
  if (B.size()!=XDoF*UDoF)
    throw std::runtime_error("State matrix B size is ill-formed!");
  if (C.size()!=YDoF*XDoF)
    throw std::runtime_error("State matrix C size is ill-formed!");
  //if (D.size()!=YDoF*UDoF)
  //  throw std::runtime_error("State matrix D size is ill-formed!");
  if (Q.size()!=XDoF*XDoF)
    throw std::runtime_error("Weight matrix Q size is ill-formed!");
  if (R.size()!=UDoF*UDoF)
    throw std::runtime_error("Weight matrix R size is ill-formed!");
  if (N.size()!=XDoF*UDoF)
    throw std::runtime_error("Weight matrix N size is ill-formed!");
  // convert to Eigen types
  this->A_ = Eigen::MatrixXd::Identity(XDoF, XDoF);
  this->B_ = Eigen::MatrixXd::Zero(XDoF, UDoF);
  this->C_ = Eigen::MatrixXd::Identity(YDoF, XDoF);
  //this->D_ = Eigen::MatrixXd::Zero(YDoF, UDoF);
  this->X_ = Eigen::VectorXd::Zero(XDoF);
  this->X_des_ = Eigen::VectorXd::Zero(XDoF);
  this->U_ = Eigen::VectorXd::Zero(UDoF);
  this->U_act_ = Eigen::VectorXd::Zero(UDoF);
  this->Y_ = Eigen::VectorXd::Zero(YDoF);
  this->Q_ = Eigen::MatrixXd::Identity(XDoF, XDoF);
  this->R_ = Eigen::MatrixXd::Identity(UDoF, UDoF);
  this->N_ = Eigen::MatrixXd::Zero(XDoF,UDoF);
  matrixPack(A, this->A_);
  matrixPack(B, this->B_);
  matrixPack(C, this->C_);
  //matrixPack(D, this->D_);
  matrixPack(Q, this->Q_);
  matrixPack(R, this->R_);
  matrixPack(N, this->N_);
  if (discretize)
  {
    // Apply Tustin transformation to discretize A_.
    this->A_ = (Eigen::MatrixXd::Identity(XDoF,XDoF) + 0.5*dt*this->A_)*
                (Eigen::MatrixXd::Identity(XDoF,XDoF) - 0.5*dt*this->A_).
                  colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(XDoF,XDoF));
    this->B_ *= dt;
  }
  // Create LQR object
  this->optimal_controller = dLQR(this->A_, this->B_, this->Q_, this->R_, this->N_);
  std::cout << "LQR Controller Gains: \n" << this->optimal_controller.getK() << std::endl;
}

void LqrControl::LqrControl(int& XDoF, int& UDoF, int& YDoF,
                            bool& discretize, bool& u_feedback, double& dt,
                            std::vector<double>& A, std::vector<double>& B,
                            std::vector<double>& C, std::vector<double>& D,
                            std::vector<double>& Q, std::vector<double>& R):
                LqrControl(XDoF, UDoF, YDoF,
                           discretize, u_feedback, dt,
                           A, B, C, D, Q, R,
                           std::vector<double>(XDoF*UDoF, 0.0));

int LqrControl::matrixPack(std::vector<double>& in, Eigen::MatrixXd& out){
  // conversion from std types to Eigen types
  if (out.size() != in.size())
    return -1;
  size_t ind = 0;
  for (size_t i = 0; i < out.rows(); i++)
  {
    for (size_t j = 0; j < out.cols(); j++)
    {
      out(i,j) = in[ind];
      ind ++;
    }
  }
  return 0;
}

inline void LqrControl::setCmdToZeros()
{
  std::lock_guard<std::mutex> l(_mtx);
  this->U_.setZero();
}

inline Eigen::VectorXd LqrControl::getControl()
{
  return this->U_;
}

void LqrControl::updateActualControl(const Eigen::VectorXd& u){
  std::lock_guard<std::mutex> l(_mtx);
  if (u.size() != this->UDoF_)
    return;
  this->U_act_ = u;
}

Eigen::VectorXd LqrControl::getPrediction(){
  std::lock_guard<std::mutex> l(_mtx);
  if (this->predicted_)
    return this->Y_;
  this->predicted_ = true;
  this->X_ = this->A_ * this->X_ + this->B_ * this->U_act_;
  this->Y_ = this->C_ * this->X_;
  return this->Y_;
}

void LqrControl::updateMeasurement(const std_msgs::msg::Float64MultiArray::SharedPtr msg){
  // Assuming all elements in the msg array are valid readings.
  if (msg->data.size() != this->YDoF_)
  {
    // size checking
    return;
  }
  std::lock_guard<std::mutex> l(_mtx);
  this->state_update_time_ = rclcpp::Clock().now();
  for (size_t i = 0; i < this->YDoF_; i++)
  {
    this->Y_(i) = msg->data[i];
  }
}

void LqrControl::updateMeasurement(Eigen::VectorXd& Y){
  if (Y.size() == this->YDoF_)
  {
    std::lock_guard<std::mutex> l(_mtx);
    this->state_update_time_ = rclcpp::Clock().now();
    this->Y_ = Y;
  }
}

inline Eigen::VectorXd LqrControl::getDesiredState()
{
  return this->X_des_;
}

void LqrControl::updateDesiredState(const std_msgs::msg::Float64MultiArray::SharedPtr msg){
  // Assuming all elements in the msg array are valid readings.
  if (msg->data.size() != this->XDoF_)
  {
    // size checking
    return;
  }
  std::lock_guard<std::mutex> l(_mtx);
  for (size_t i = 0; i < this->XDoF_; i++)
  {
    this->X_des_(i) = msg->data[i];
  }
}

void LqrControl::updateDesiredState(Eigen::VectorXd& X_des){
  if (X_des.size() == this->XDoF_)
  {
    std::lock_guard<std::mutex> l(_mtx);
    this->X_des_ = X_des;
  }
}

void LqrControl::controlCallback(rclcpp::Logger& logger)
{
  rclcpp::Time control_time = rclcpp::Clock().now();
  rclcpp::Duration time_diff = control_time - this->state_update_time_;
  double state_dt = static_cast<double>(time_diff.seconds()) +
              static_cast<double>(time_diff.nanoseconds())*1e-9;
  time_diff = control_time - this->last_control_time_;
  double control_dt = static_cast<double>(time_diff.seconds()) +
              static_cast<double>(time_diff.nanoseconds())*1e-9;
  this->last_control_time_ = control_time;

  if ( state_dt < this->state_timeout_ )
  {
    // summarize asynchronous updates of the state measurements.
    this->X_ = this->C_.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(this->Y_);
    this->U_ = this->optimal_controller.calculateControl(this->X_, this->X_des_);
    
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
      /*if (!this->optimal_state_estimate.isInitialized())
        initializeSystemAndControls();
      if (this->optimal_state_estimate.isInitialized() )
        RCLCPP_DEBUG(logger, "Controller State Estimator Initialized!\n");*/
    }
  }

  if (!this->u_feedback_)
    this->U_act_ = this->U_;
}

} 
