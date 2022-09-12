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

#include "lqg_control/lqg_control.hpp"

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

LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                        const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                        const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        const Eigen::MatrixXd& N)
{
  LqgControl::_LqrControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, N);
}

LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                        const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                        const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        const Eigen::MatrixXd& N,
                        const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn,
                        const Eigen::MatrixXd& P0)
{
  LqgControl::_LqgControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, N, Sd, Sn, P0);
}

LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                        const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                        const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R)
{
  Eigen::MatrixXd zeroNMatrix;
  zeroNMatrix = Eigen::MatrixXd::Zero(XDoF,UDoF);
  LqgControl::_LqrControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, zeroNMatrix);
}

LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                        const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                        const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn,
                        const Eigen::MatrixXd& P0)
{
  Eigen::MatrixXd zeroNMatrix;
  zeroNMatrix = Eigen::MatrixXd::Zero(XDoF,UDoF);
  LqgControl::_LqgControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, zeroNMatrix, Sd, Sn, P0);
}

LqgControl::LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                        bool& discretize, bool& u_feedback, double& dt,
                        const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                        const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                        const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                        const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn)
{
  Eigen::MatrixXd zeroNMatrix;
  zeroNMatrix = Eigen::MatrixXd::Zero(XDoF,UDoF);
  Eigen::MatrixXd smallP0Matrix;
  smallP0Matrix = Eigen::MatrixXd::Ones(XDoF,XDoF) * 1e-12;
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
  if (A.size() != XDoF_*XDoF_)
    throw std::runtime_error("State matrix A size is ill-formed!");
  if (B.size() != XDoF_*UDoF_)
    throw std::runtime_error("State matrix B size is ill-formed!");
  if (C.size() != YDoF_*XDoF_)
    throw std::runtime_error("State matrix C size is ill-formed!");
  //if (D.size() != YDoF_*UDoF_)
  //  throw std::runtime_error("State matrix D size is ill-formed!");
  if (Q.size() != XDoF_*XDoF_)
    throw std::runtime_error("Weight matrix Q size is ill-formed!");
  if (R.size() != UDoF_*UDoF_)
    throw std::runtime_error("Weight matrix R size is ill-formed!");
  if (N.size() != XDoF_*UDoF_)
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
  //std::cout << "LQR Controller Gains: \n" << this->optimal_controller.getK() << std::endl;
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
  this->initializeCovariances(Sd, Sn, P0);
}

void LqgControl::_LqrControlFull_(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& N)
{
  this->_mtx = new std::mutex();
  this->XDoF_ = XDoF;
  this->UDoF_ = UDoF;
  this->YDoF_ = YDoF;
  this->dt_ = dt;
  this->u_feedback_ = u_feedback;
  // clean std::vector parameters (check !NULL and check size)
  if ((size_t)(A.rows()) != XDoF_ || (size_t)(A.cols()) != XDoF_)
    throw std::runtime_error("State matrix A size is ill-formed!");
  if ((size_t)(B.rows()) != XDoF_ || (size_t)(B.cols()) != UDoF_)
    throw std::runtime_error("State matrix B size is ill-formed!");
  if ((size_t)(C.rows()) != YDoF_ || (size_t)(C.cols()) != XDoF_)
    throw std::runtime_error("State matrix C size is ill-formed!");
  //if (D.rows() != YDoF_ || D.cols() != UDoF_)
  //  throw std::runtime_error("State matrix D size is ill-formed!");
  if ((size_t)(Q.rows()) != XDoF_ || (size_t)(Q.cols()) != XDoF_)
    throw std::runtime_error("Weight matrix Q size is ill-formed!");
  if ((size_t)(R.rows()) != UDoF_ || (size_t)(R.cols()) != UDoF_)
    throw std::runtime_error("Weight matrix R size is ill-formed!");
  if ((size_t)(N.rows()) != XDoF_ || (size_t)(N.cols()) != UDoF_)
    throw std::runtime_error("Weight matrix N size is ill-formed!");
  // convert to Eigen types
  this->A_ = A;
  this->B_ = B;
  this->C_ = C;
  //this->D_ = D;
  this->X_ = Eigen::VectorXd::Zero(XDoF);
  this->X_des_ = Eigen::VectorXd::Zero(XDoF);
  this->U_ = Eigen::VectorXd::Zero(UDoF);
  this->U_act_ = Eigen::VectorXd::Zero(UDoF);
  this->Y_ = Eigen::VectorXd::Zero(YDoF);
  this->Q_ = Q;
  this->R_ = R;
  this->N_ = N;
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
  //std::cout << "LQR Controller Gains: \n" << this->optimal_controller.getK() << std::endl;
}

void LqgControl::_LqgControlFull_(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& N,
                const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn,
                const Eigen::MatrixXd& P0)
{
  LqgControl::_LqrControlFull_(XDoF, UDoF, YDoF, discretize, u_feedback, dt,
                                A, B, C, D, Q, R, N);
  this->initializeCovariances(Sd, Sn, P0);
}

void LqgControl::initializeCovariances(std::vector<double>& Sd, std::vector<double>& Sn)
{
  std::vector<double> smallP0Matrix(XDoF_*XDoF_, 1e-12);
  this->initializeCovariances(Sd, Sn, smallP0Matrix);
}

void LqgControl::initializeCovariances(std::vector<double>& Sd, std::vector<double>& Sn,
                                      std::vector<double>& P0)
{
  if ((size_t)(Sd.size())!=XDoF_*XDoF_)
    throw std::runtime_error("Disturbance covariance matrix Sd is ill-formed!");
  if ((size_t)(Sn.size())!=YDoF_*YDoF_)
    throw std::runtime_error("Measurement covariance matrix Sn is ill-formed!");
  matrixPack(Sd, this->sigmaDisturbance_);
  matrixPack(Sn, this->sigmaMeasurements_);
  Eigen::MatrixXd stateCov;
  stateCov = Eigen::MatrixXd::Ones(XDoF_,XDoF_) * 1e-12;
  // If the initial state covariance is mal-formed, it will be ignored.
  matrixPack(P0, stateCov);
  
  this->optimal_state_estimate = KalmanFilter(dt_, this->A_, this->B_, this->C_, 
                                  this->sigmaDisturbance_, this->sigmaMeasurements_, stateCov);
}

void LqgControl::initializeCovariances(const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn)
{
  Eigen::MatrixXd smallP0Matrix;
  smallP0Matrix = Eigen::MatrixXd::Ones(XDoF_,XDoF_) * 1e-12;
  this->initializeCovariances(Sd, Sn, smallP0Matrix);
}

void LqgControl::initializeCovariances(const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn,
                                      const Eigen::MatrixXd& P0)
{
  if ((size_t)(Sd.rows())!=XDoF_ || (size_t)(Sd.cols())!=XDoF_)
    throw std::runtime_error("Disturbance covariance matrix Sd is ill-formed!");
  if ((size_t)(Sn.rows())!=YDoF_ || (size_t)(Sn.cols())!=YDoF_)
    throw std::runtime_error("Measurement covariance matrix Sn is ill-formed!");
  this->sigmaDisturbance_ = Sd;
  this->sigmaMeasurements_ = Sn;
  if ((size_t)(P0.rows())!=XDoF_ || (size_t)(P0.cols())!=XDoF_){
    this->optimal_state_estimate = KalmanFilter(dt_, this->A_, this->B_, this->C_, 
                                  this->sigmaDisturbance_, this->sigmaMeasurements_, P0);
  } else {
    Eigen::MatrixXd stateCov;
    stateCov = Eigen::MatrixXd::Ones(XDoF_,XDoF_) * 1e-12;
    this->optimal_state_estimate = KalmanFilter(dt_, this->A_, this->B_, this->C_, 
                                  this->sigmaDisturbance_, this->sigmaMeasurements_, stateCov);
  }
}

void LqgControl::initializeStates(std::vector<double>& X0)
{
  rclcpp::Time tNow = rclcpp::Clock().now();
  Eigen::VectorXd x0_;
  x0_ = Eigen::VectorXd::Zero(XDoF_);
  vectorPack(X0, x0_);
  this->optimal_state_estimate.init(
    static_cast<double>(tNow.seconds())+
    static_cast<double>(tNow.nanoseconds())*1.e-9,
    x0_);
}

void LqgControl::initializeStates(const Eigen::VectorXd& X0)
{
  rclcpp::Time tNow = rclcpp::Clock().now();
  this->optimal_state_estimate.init(
    static_cast<double>(tNow.seconds())+
    static_cast<double>(tNow.nanoseconds())*1.e-9,
    X0);
}

void LqgControl::initializeStates(std::vector<double>& X0, std::vector<double>& P0)
{
  rclcpp::Time tNow = rclcpp::Clock().now();
  Eigen::VectorXd x0_;
  x0_ = Eigen::VectorXd::Zero(XDoF_);
  vectorPack(X0, x0_);
  Eigen::MatrixXd stateCov;
  stateCov = Eigen::MatrixXd::Ones(XDoF_,XDoF_) * 1e-12;
  matrixPack(P0, stateCov);
  this->optimal_state_estimate.init(
    static_cast<double>(tNow.seconds())+
    static_cast<double>(tNow.nanoseconds())*1.e-9,
    x0_, stateCov);
}

void LqgControl::initializeStates(const Eigen::VectorXd& X0, const Eigen::MatrixXd& P0)
{
  rclcpp::Time tNow = rclcpp::Clock().now();
  this->optimal_state_estimate.init(
    static_cast<double>(tNow.seconds())+
    static_cast<double>(tNow.nanoseconds())*1.e-9,
    X0, P0);
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
  if ((size_t)(Y.size()) != this->YDoF_)
    return;
  std::lock_guard<std::mutex> l(*_mtx);
  this->state_update_time_ = rclcpp::Clock().now();
  this->Y_ = Y;
}

void LqgControl::updateMeasurementCov(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  // Assuming all elements in the msg array are row-major valid readings.
  if (msg->data.size() == (this->YDoF_ + 1) * this->YDoF_ / 2)
  {
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
    return;
  }
  else if (msg->data.size() != this->YDoF_ * this->YDoF_)
    return;

  // Full matrix notation
  std::lock_guard<std::mutex> l(*_mtx);
  matrixPack(msg->data, this->sigmaMeasurements_);
}

void LqgControl::updateMeasurementCov(Eigen::MatrixXd& cov)
{
  if ((size_t)(cov.cols()) != this->YDoF_)
    return;
  if ((size_t)(cov.rows()) != this->YDoF_)
    return;
  std::lock_guard<std::mutex> l(*_mtx);
  this->sigmaMeasurements_ = cov;
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
  if ((size_t)(X_des.size()) != this->XDoF_)
    return;
  std::lock_guard<std::mutex> l(*_mtx);
  this->X_des_ = X_des;
}

void LqgControl::controlCallback(const rclcpp::Logger& logger)
{
  rclcpp::Time control_time = rclcpp::Clock().now();
  rclcpp::Duration time_diff = control_time - this->state_update_time_;
  double state_dt = static_cast<double>(time_diff.seconds()) +
              static_cast<double>(time_diff.nanoseconds())*1e-9;
  time_diff = control_time - this->last_control_time_;
  double control_dt = static_cast<double>(time_diff.seconds()) +
              static_cast<double>(time_diff.nanoseconds())*1.e-9;
  this->last_control_time_ = control_time;

  if ( !this->optimal_state_estimate.isInitialized() )
  {
    this->setCmdToZeros();
    rclcpp::Clock clock;
    RCLCPP_WARN_THROTTLE(logger, clock, 1, "Controller still waiting for initial state to initialize...\n");
    return;
  }

  this->predicted_ = false;

  if ( state_dt >= this->dt_ ){
    this->getPrediction();
    this->optimal_state_estimate.update_time_variant_R(this->Y_, this->U_act_, this->sigmaMeasurements_, control_dt);
    this->setCmdToZeros();
    RCLCPP_WARN(logger,
                    "State observation age: %f seconds is too stale! (timeout = %f s)\n",
                    state_dt, this->dt_);
    return;
  }

  // summarize asynchronous updates of the state measurements.
  this->optimal_state_estimate.update_time_variant_R(this->Y_, this->U_act_, this->sigmaMeasurements_, control_dt);
  this->U_ = this->optimal_controller.calculateControl(this->optimal_state_estimate.state(), this->X_des_);

  //std::cout << "Estimated State: \n" << this->x_est_.transpose() << "\n";
  /* <<
                "Desired State: \n" << this->x_desired_.transpose() << "\n" <<
                "Observations: \n" << this->Y_.transpose() << "\n" <<
                "Calculated Best Command: \n" << this->u_raw_.transpose() << "\n" <<
                "\n----------------------------------------\n";
                */

  if (!this->u_feedback_)
    this->U_act_ = this->U_;
  
}

} 
