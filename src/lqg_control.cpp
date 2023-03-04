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

namespace control {
LqrControl::LqrControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
                       const bool &u_feedback, const double &dt, const std::vector<double> &A,
                       const std::vector<double> &B, const std::vector<double> &C,
                       const std::vector<double> &D, const std::vector<double> &Q,
                       const std::vector<double> &R, const std::vector<double> &N)
    : A_(Eigen::Map<const Eigen::MatrixXd>(A.data(), XDoF, XDoF)),
      B_(Eigen::Map<const Eigen::MatrixXd>(B.data(), XDoF, UDoF)),
      C_(Eigen::Map<const Eigen::MatrixXd>(C.data(), YDoF, XDoF)),
      D_(Eigen::Map<const Eigen::MatrixXd>(D.data(), YDoF, UDoF)),
      Q_(Eigen::Map<const Eigen::MatrixXd>(Q.data(), XDoF, XDoF)),
      R_(Eigen::Map<const Eigen::MatrixXd>(R.data(), UDoF, UDoF)),
      N_(Eigen::Map<const Eigen::MatrixXd>(N.data(), XDoF, UDoF)),
      X_(XDoF),
      X_des_(XDoF),
      U_(UDoF),
      U_act_(UDoF),
      Y_(YDoF),
      XDoF_(XDoF),
      UDoF_(UDoF),
      YDoF_(YDoF),
      dt_(dt),
      u_feedback_(u_feedback) {
  // clean std::vector parameters (check !NULL and check size)
  if (A.size() != XDoF_ * XDoF_) throw std::runtime_error("State matrix A size is ill-formed!");
  if (B.size() != XDoF_ * UDoF_) throw std::runtime_error("State matrix B size is ill-formed!");
  if (C.size() != YDoF_ * XDoF_) throw std::runtime_error("State matrix C size is ill-formed!");
  // if (D.size() != YDoF_*UDoF_)
  //   throw std::runtime_error("State matrix D size is ill-formed!");
  if (Q.size() != XDoF_ * XDoF_) throw std::runtime_error("Weight matrix Q size is ill-formed!");
  if (R.size() != UDoF_ * UDoF_) throw std::runtime_error("Weight matrix R size is ill-formed!");
  if (N.size() != XDoF_ * UDoF_) throw std::runtime_error("Weight matrix N size is ill-formed!");
  this->X_.setZero();
  this->X_des_.setZero();
  this->U_.setZero();
  this->U_act_.setZero();
  this->Y_.setZero();
  auto cinvTmp = this->C_.completeOrthogonalDecomposition();
  this->C_inv_ = cinvTmp.pseudoInverse();
  if (discretize) {
    // Apply Tustin transformation to discretize A_.
    this->A_ = (Eigen::MatrixXd::Identity(XDoF, XDoF) + 0.5 * dt * this->A_) *
               (Eigen::MatrixXd::Identity(XDoF, XDoF) - 0.5 * dt * this->A_)
                   .colPivHouseholderQr()
                   .solve(Eigen::MatrixXd::Identity(XDoF, XDoF));
    this->B_ *= dt;
  }
  // Create LQR object
  this->optimal_controller = new dLQR(this->A_, this->B_, this->Q_, this->R_, this->N_);
  // std::cout << "LQR Controller Gains: \n" << this->optimal_controller.getK()
  // << std::endl;
}

LqrControl::LqrControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
                       const bool &u_feedback, const double &dt, const Eigen::MatrixXd &A,
                       const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
                       const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &N)
    : A_(A),
      B_(B),
      C_(C),
      C_inv_(XDoF, YDoF),
      D_(D),
      Q_(Q),
      R_(R),
      N_(N),
      X_(XDoF),
      X_des_(XDoF),
      U_(UDoF),
      U_act_(UDoF),
      Y_(YDoF),
      XDoF_(XDoF),
      UDoF_(UDoF),
      YDoF_(YDoF),
      dt_(dt),
      u_feedback_(u_feedback) {
  // clean std::vector parameters (check !NULL and check size)
  if ((size_t)(A.rows()) != XDoF_ || (size_t)(A.cols()) != XDoF_)
    throw std::runtime_error("State matrix A size is ill-formed!");
  if ((size_t)(B.rows()) != XDoF_ || (size_t)(B.cols()) != UDoF_)
    throw std::runtime_error("State matrix B size is ill-formed!");
  if ((size_t)(C.rows()) != YDoF_ || (size_t)(C.cols()) != XDoF_)
    throw std::runtime_error("State matrix C size is ill-formed!");
  // if (D.rows() != YDoF_ || D.cols() != UDoF_)
  //   throw std::runtime_error("State matrix D size is ill-formed!");
  if ((size_t)(Q.rows()) != XDoF_ || (size_t)(Q.cols()) != XDoF_)
    throw std::runtime_error("Weight matrix Q size is ill-formed!");
  if ((size_t)(R.rows()) != UDoF_ || (size_t)(R.cols()) != UDoF_)
    throw std::runtime_error("Weight matrix R size is ill-formed!");
  if ((size_t)(N.rows()) != XDoF_ || (size_t)(N.cols()) != UDoF_)
    throw std::runtime_error("Weight matrix N size is ill-formed!");
  // convert to Eigen types
  this->X_.setZero();
  this->X_des_.setZero();
  this->U_.setZero();
  this->U_act_.setZero();
  this->Y_.setZero();
  auto cinvTmp = this->C_.completeOrthogonalDecomposition();
  this->C_inv_ = cinvTmp.pseudoInverse();
  if (discretize) {
    // Apply Tustin transformation to discretize A_.
    this->A_ = (Eigen::MatrixXd::Identity(XDoF, XDoF) + 0.5 * dt * this->A_) *
               (Eigen::MatrixXd::Identity(XDoF, XDoF) - 0.5 * dt * this->A_)
                   .colPivHouseholderQr()
                   .solve(Eigen::MatrixXd::Identity(XDoF, XDoF));
    this->B_ *= dt;
  }
  // Create LQR object
  this->optimal_controller = new dLQR(this->A_, this->B_, this->Q_, this->R_, this->N_);
  // std::cout << "LQR Controller Gains: \n" << this->optimal_controller.getK()
  // << std::endl;
}

//==================================================================================

LqgControl::LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
                       const bool &u_feedback, const double &dt, const std::vector<double> &A,
                       const std::vector<double> &B, const std::vector<double> &C,
                       const std::vector<double> &D, const std::vector<double> &Q,
                       const std::vector<double> &R, const std::vector<double> &N,
                       const std::vector<double> &Sd, const std::vector<double> &Sn,
                       const std::vector<double> &P0)
    : LqrControl(XDoF, UDoF, YDoF, discretize, u_feedback, dt, A, B, C, D, Q, R, N),
      sigmaDisturbance_(Eigen::Map<const Eigen::MatrixXd>(Sd.data(), XDoF, XDoF)),
      sigmaMeasurements_(Eigen::Map<const Eigen::MatrixXd>(Sn.data(), YDoF, YDoF)) {
  if (Sd.size() != XDoF_ * XDoF_)
    throw std::runtime_error("Disturbance covariance matrix Sd size is ill-formed!");
  if (Sn.size() != YDoF_ * YDoF_)
    throw std::runtime_error("Measurement covariance matrix Sn size is ill-formed!");
  if (P0.size() != XDoF_ * XDoF_) {
    this->optimal_state_estimate =
        new KalmanFilter(dt_, this->A_, this->B_, this->C_, this->sigmaDisturbance_,
                         this->sigmaMeasurements_, Eigen::MatrixXd::Constant(XDoF_, XDoF_, 1e-12));
  } else {
    Eigen::MatrixXd p0Mat = Eigen::Map<const Eigen::MatrixXd>(P0.data(), XDoF, XDoF);
    this->optimal_state_estimate =
        new KalmanFilter(dt_, this->A_, this->B_, this->C_, this->sigmaDisturbance_,
                         this->sigmaMeasurements_, p0Mat);
  }
}

LqgControl::LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
                       const bool &u_feedback, const double &dt, const Eigen::MatrixXd &A,
                       const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
                       const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &N,
                       const Eigen::MatrixXd &Sd, const Eigen::MatrixXd &Sn,
                       const Eigen::MatrixXd &P0)
    : LqrControl(XDoF, UDoF, YDoF, discretize, u_feedback, dt, A, B, C, D, Q, R, N),
      sigmaDisturbance_(Sd),
      sigmaMeasurements_(Sn) {
  if ((size_t)(Sd.rows()) != XDoF_ || (size_t)(Sd.cols()) != XDoF_)
    throw std::runtime_error("Disturbance covariance matrix Sd size is ill-formed!");
  if ((size_t)(Sn.rows()) != YDoF_ || (size_t)(Sn.cols()) != YDoF_)
    throw std::runtime_error("Measurement covariance matrix Sn size is ill-formed!");
  if ((size_t)(P0.rows()) != XDoF_ || (size_t)(P0.cols()) != XDoF_) {
    this->optimal_state_estimate =
        new KalmanFilter(dt_, this->A_, this->B_, this->C_, this->sigmaDisturbance_,
                         this->sigmaMeasurements_, Eigen::MatrixXd::Constant(XDoF_, XDoF_, 1e-12));
  } else {
    this->optimal_state_estimate = new KalmanFilter(
        dt_, this->A_, this->B_, this->C_, this->sigmaDisturbance_, this->sigmaMeasurements_, P0);
  }
}

void LqgControl::initializeStates(std::vector<double> &X0) {
  rclcpp::Time tNow = rclcpp::Clock().now();
  Eigen::VectorXd x0_;
  x0_ = Eigen::Map<const Eigen::VectorXd>(X0.data(), XDoF_);
  this->optimal_state_estimate->init(
      static_cast<double>(tNow.seconds()) + static_cast<double>(tNow.nanoseconds()) * 1.e-9, x0_);
}

void LqgControl::initializeStates(const Eigen::VectorXd &X0) {
  rclcpp::Time tNow = rclcpp::Clock().now();
  this->optimal_state_estimate->init(
      static_cast<double>(tNow.seconds()) + static_cast<double>(tNow.nanoseconds()) * 1.e-9, X0);
}

void LqgControl::initializeStates(std::vector<double> &X0, std::vector<double> &P0) {
  rclcpp::Time tNow = rclcpp::Clock().now();
  Eigen::VectorXd x0_;
  x0_ = Eigen::Map<const Eigen::VectorXd>(X0.data(), XDoF_);
  Eigen::MatrixXd stateCov;
  if (P0.size() == XDoF_ * XDoF_) {
    stateCov = Eigen::Map<const Eigen::MatrixXd>(P0.data(), XDoF_, XDoF_);
  } else {
    stateCov = Eigen::MatrixXd::Ones(XDoF_, XDoF_) * 1e-12;
  }
  this->optimal_state_estimate->init(
      static_cast<double>(tNow.seconds()) + static_cast<double>(tNow.nanoseconds()) * 1.e-9, x0_,
      stateCov);
}

void LqgControl::initializeStates(const Eigen::VectorXd &X0, const Eigen::MatrixXd &P0) {
  rclcpp::Time tNow = rclcpp::Clock().now();
  this->optimal_state_estimate->init(
      static_cast<double>(tNow.seconds()) + static_cast<double>(tNow.nanoseconds()) * 1.e-9, X0,
      P0);
}

void LqgControl::updateMeasurementCov(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  // Assuming all elements in the msg array are row-major valid readings.
  if (msg->data.size() == (this->YDoF_ + 1) * this->YDoF_ / 2) {
    // Upper triangular matrix notation
    size_t ind = 0;
    for (size_t i = 0; i < this->YDoF_; i++) {
      for (size_t j = i; j < this->YDoF_; j++) {
        this->sigmaMeasurements_(i, j) = msg->data[ind];
        ind++;
      }
      for (size_t j = 0; j < i; j++) {
        this->sigmaMeasurements_(i, j) = this->sigmaMeasurements_(j, i);
      }
    }
    return;
  } else if (msg->data.size() != this->YDoF_ * this->YDoF_)
    return;

  // Full matrix notation
  if (msg->data.size() == this->YDoF_ * this->YDoF_)
    this->sigmaMeasurements_ = Eigen::Map<const Eigen::MatrixXd>(msg->data.data(), YDoF_, YDoF_);
}

void LqgControl::controlCallback(const rclcpp::Logger &logger) {
  if (this->optimal_state_estimate == NULL) {
    LqrControl::controlCallback(logger);
    return;
  }
  
  rclcpp::Time control_time = rclcpp::Clock().now();
  rclcpp::Duration time_diff = control_time - this->state_update_time_;
  double state_dt = static_cast<double>(time_diff.seconds()) +
                    static_cast<double>(time_diff.nanoseconds()) * 1e-9;
  time_diff = control_time - this->last_control_time_;
  double control_dt = static_cast<double>(time_diff.seconds()) +
                      static_cast<double>(time_diff.nanoseconds()) * 1.e-9;
  this->last_control_time_ = control_time;

  if (!this->optimal_state_estimate->isInitialized()) {
    this->setCmdToZeros();
    rclcpp::Clock clock;
    RCLCPP_WARN_THROTTLE(logger, clock, 2000U,
                         "Controller still waiting for initial state to initialize...\n");
    return;
  }

  this->predicted_ = false;

  if (state_dt >= this->dt_) {
    this->getPrediction();
    this->optimal_state_estimate->update_time_variant_R(this->Y_, this->U_act_,
                                                        this->sigmaMeasurements_, control_dt);
    this->setCmdToZeros();
    RCLCPP_WARN(logger, "State observation age: %f seconds is too stale! (timeout = %f s)\n",
                state_dt, this->dt_);
    return;
  }

  // summarize asynchronous updates of the state measurements.
  this->optimal_state_estimate->update_time_variant_R(this->Y_, this->U_act_,
                                                      this->sigmaMeasurements_, control_dt);
  this->U_ = this->optimal_controller->calculateControl(this->optimal_state_estimate->state(),
                                                        this->X_des_);

  // std::cout << "Estimated State: \n" << this->x_est_.transpose() << "\n";
  /* <<
                "Desired State: \n" << this->x_desired_.transpose() << "\n" <<
                "Observations: \n" << this->Y_.transpose() << "\n" <<
                "Calculated Best Command: \n" << this->u_raw_.transpose() <<
     "\n" <<
                "\n----------------------------------------\n";
                */

  if (!this->u_feedback_) this->U_act_ = this->U_;
}

}  // namespace control
