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
LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        std::vector<double>& A, std::vector<double>& B,
                        std::vector<double>& C, std::vector<double>& D,
                        std::vector<double>& Q, std::vector<double>& R,
                        std::vector<double>& N)
{
  LqgControl::_LqrControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, N);
}

LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        std::vector<double>& A, std::vector<double>& B,
                        std::vector<double>& C, std::vector<double>& D,
                        std::vector<double>& Q, std::vector<double>& R,
                        std::vector<double>& N,
                        std::vector<double>& Sd, std::vector<double>& Sn,
                        std::vector<double>& P0)
{
  LqgControl::_LqgControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, N, Sd, Sn, P0);
}

LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        std::vector<double>& A, std::vector<double>& B,
                        std::vector<double>& C, std::vector<double>& D,
                        std::vector<double>& Q, std::vector<double>& R)
{
  std::vector<double> zeroNMatrix(XDoF*UDoF, 0.0);
  LqgControl::_LqrControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, zeroNMatrix);
}

LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        std::vector<double>& A, std::vector<double>& B,
                        std::vector<double>& C, std::vector<double>& D,
                        std::vector<double>& Q, std::vector<double>& R,
                        std::vector<double>& Sd, std::vector<double>& Sn,
                        std::vector<double>& P0)
{
  std::vector<double> zeroNMatrix(XDoF*UDoF, 0.0);
  LqgControl::_LqgControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, zeroNMatrix, Sd, Sn, P0);
}

LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        std::vector<double>& A, std::vector<double>& B,
                        std::vector<double>& C, std::vector<double>& D,
                        std::vector<double>& Q, std::vector<double>& R,
                        std::vector<double>& Sd, std::vector<double>& Sn)
{
  std::vector<double> zeroNMatrix(XDoF*UDoF, 0.0);
  std::vector<double> smallP0Matrix(XDoF*XDoF, 1e-12);
  LqgControl::_LqgControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, zeroNMatrix, Sd, Sn, smallP0Matrix);
}

void LqgControl::_LqrControlFull_(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& N)
{
  this->_mtx = new std::mutex();
  this->XDoF_ = XDoF;
  this->UDoF_ = UDoF;
  this->YDoF_ = YDoF;
  this->dt_ = dt;
  this->u_feedback_ = u_feedback;
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

void LqgControl::_LqgControlFull_(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& N,
                std::vector<double>& Sd, std::vector<double>& Sn,
                std::vector<double>& P0)
{
  LqgControl::_LqrControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, N);
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

int LqgControl::matrixPack(std::vector<double>& in, Eigen::MatrixXd& out)
{
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

void LqgControl::setCmdToZeros()
{
  std::lock_guard<std::mutex> l(*_mtx);
  this->U_.setZero();
}

void LqgControl::updateActualControl(const Eigen::VectorXd& u)
{
  std::lock_guard<std::mutex> l(*_mtx);
  if (u.size() != this->UDoF_)
    return;
  this->U_act_ = u;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> LqgControl::getPrediction()
{
  std::lock_guard<std::mutex> l(*_mtx);
  if (this->predicted_)
    return std::make_pair(this->Y_, this->sigmaMeasurements_);
  this->predicted_ = true;
  std::tie(this->Y_, this->sigmaMeasurements_) = this->optimal_state_estimate.predict(this->U_act_);
  return std::make_pair(this->Y_, this->sigmaMeasurements_);
}

void LqgControl::updateMeasurement(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  // Assuming all elements in the msg array are valid readings.
  if (msg->data.size() != this->YDoF_)
  {
    // size checking
    return;
  }
  std::lock_guard<std::mutex> l(*_mtx);
  this->state_update_time_ = rclcpp::Clock().now();
  for (size_t i = 0; i < this->YDoF_; i++)
  {
    this->Y_(i) = msg->data[i];
  }
}

void LqgControl::updateMeasurement(Eigen::VectorXd& Y)
{
  if (Y.size() == this->YDoF_)
  {
    std::lock_guard<std::mutex> l(*_mtx);
    this->state_update_time_ = rclcpp::Clock().now();
    this->Y_ = Y;
  }
}

void LqgControl::updateMeasurementCov(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  // Assuming all elements in the msg array are row-major valid readings.
  if (msg->data.size() == this->YDoF_ * this->YDoF_)
  {
    // Full matrix notation
    std::lock_guard<std::mutex> l(*_mtx);
    matrixPack(msg->data, this->sigmaMeasurements_);
  }
  else if (msg->data.size() == (this->YDoF_ + 1) * this->YDoF_ / 2){
    std::lock_guard<std::mutex> l(*_mtx);
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

void LqgControl::updateMeasurementCov(Eigen::MatrixXd& cov)
{
  if (cov.cols()==this->YDoF_ && cov.rows()==this->YDoF_)
  {
    std::lock_guard<std::mutex> l(*_mtx);
    this->sigmaMeasurements_ = cov;
  }
}

void LqgControl::updateDesiredState(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  // Assuming all elements in the msg array are valid readings.
  if (msg->data.size() != this->XDoF_)
  {
    // size checking
    return;
  }
  std::lock_guard<std::mutex> l(*_mtx);
  for (size_t i = 0; i < this->XDoF_; i++)
  {
    this->X_des_(i) = msg->data[i];
  }
}

void LqgControl::updateDesiredState(Eigen::VectorXd& X_des)
{
  if (X_des.size() == this->XDoF_)
  {
    std::lock_guard<std::mutex> l(*_mtx);
    this->X_des_ = X_des;
  }
}

void LqgControl::controlCallback(const rclcpp::Logger& logger)
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
