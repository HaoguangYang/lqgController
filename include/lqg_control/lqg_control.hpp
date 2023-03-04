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

#ifndef LQG_CONTROL_HPP
#define LQG_CONTROL_HPP

#include <eigen3/Eigen/Dense>
#include <functional>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "lqg_control/Kalman.hpp"
#include "lqg_control/LQR.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace control {

class LqrControl {
 public:
  LqrControl() = delete;

  // std::vector version
  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   * @param[in] N specifies the error-effort cross penalty. [NOT YET IMPLEMENTED]
   */
  LqrControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const std::vector<double> &A,
             const std::vector<double> &B, const std::vector<double> &C,
             const std::vector<double> &D, const std::vector<double> &Q,
             const std::vector<double> &R, const std::vector<double> &N);

  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   */
  LqrControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const std::vector<double> &A,
             const std::vector<double> &B, const std::vector<double> &C,
             const std::vector<double> &D, const std::vector<double> &Q,
             const std::vector<double> &R)
      : LqrControl(XDoF, UDoF, YDoF, discretize, u_feedback, dt, A, B, C, D, Q, R,
                   std::vector<double>(XDoF * UDoF, 0.0)){};

  // Eigen::MatrixXd version
  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   * @param[in] N specifies the error-effort cross penalty. [NOT YET IMPLEMENTED]
   */
  LqrControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const Eigen::MatrixXd &A,
             const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
             const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &N);

  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   */
  LqrControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const Eigen::MatrixXd &A,
             const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
             const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R)
      : LqrControl(XDoF, UDoF, YDoF, discretize, u_feedback, dt, A, B, C, D, Q, R,
                   Eigen::MatrixXd::Zero(XDoF, UDoF)){};

  virtual ~LqrControl() { delete (this->optimal_controller); };

  /*
  int vectorPack(std::vector<double> &in, Eigen::VectorXd &out) {
    // conversion from std types to Eigen types
    if ((size_t)(out.size()) != in.size()) return -1;
    out = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(in.data(), in.size());
    return 0;
  };

  int matrixPack(std::vector<double> &in, Eigen::MatrixXd &out) {
    // conversion from std types to Eigen types
    if ((size_t)(out.size()) != in.size()) return -1;
    size_t ind = 0;
    for (size_t i = 0; i < (size_t)(out.rows()); i++) {
      for (size_t j = 0; j < (size_t)(out.cols()); j++) {
        out(i, j) = in[ind];
        ind++;
      }
    }
    return 0;
  };

  int matrixUnpack(const Eigen::MatrixXd &in, std::vector<double> &out) {
    // conversion from std types to Eigen types
    if ((size_t)(out.size()) != static_cast<long unsigned int>(in.size())) return -1;
    size_t ind = 0;
    for (size_t i = 0; i < (size_t)(in.rows()); i++) {
      for (size_t j = 0; j < (size_t)(in.cols()); j++) {
        out[ind] = in(i, j);
        ind++;
      }
    }
    return 0;
  };
  */

  void setCmdToZeros() {
    this->U_.setZero();
    if (!this->u_feedback_) this->U_act_ = this->U_;
  };

  Eigen::MatrixXd getStateFeedbackMatrix() const { return this->optimal_controller->getK(); };

  Eigen::VectorXd getControl() const { return this->U_; };

  void updateActualControl(const Eigen::VectorXd &u) {
    if ((size_t)(u.size()) != this->UDoF_) return;
    this->U_act_ = u;
  };

  void updateMeasurement(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    // Assuming all elements in the msg array are valid readings.
    // size checking
    if (msg->data.size() != this->YDoF_) return;
    this->state_update_time_ = rclcpp::Clock().now();
    this->Y_ = Eigen::Map<const Eigen::VectorXd>(msg->data.data(), YDoF_);
  };

  void updateMeasurement(const Eigen::VectorXd &Y) {
    if ((size_t)(Y.size()) != this->YDoF_) return;
    this->state_update_time_ = rclcpp::Clock().now();
    this->Y_ = Y;
  };

  Eigen::VectorXd getDesiredState() const { return this->X_des_; };

  Eigen::VectorXd currentError() const { return this->optimal_controller->currentError(); };

  void updateDesiredState(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    // Assuming all elements in the msg array are valid readings.
    // size checking
    if (msg->data.size() != this->XDoF_) return;
    this->X_des_ = Eigen::Map<const Eigen::VectorXd>(msg->data.data(), XDoF_);
  };

  // FIXME: Not inlining these functions cause SIGSEV due to memory optimizations
  void updateDesiredState(const Eigen::VectorXd &X_des) {
    if ((size_t)(X_des.size()) != this->XDoF_) return;
    this->X_des_ = X_des;
  };

  void controlCallback(const rclcpp::Logger &logger) {
    this->X_ = this->C_inv_ * (this->Y_ - this->D_ * this->U_);
    this->U_ = this->optimal_controller->calculateControl(this->X_, this->X_des_);
  };

 protected:
  Eigen::MatrixXd A_, B_, C_, C_inv_, D_, Q_, R_, N_;
  Eigen::VectorXd X_, X_des_, U_, U_act_, Y_;
  size_t XDoF_, UDoF_, YDoF_;
  double dt_;
  bool u_feedback_, discretize_;

  rclcpp::Time state_update_time_, last_control_time_;

  control::dLQR *optimal_controller;

};  // end of class

class LqgControl : public LqrControl {
 public:
  using LqrControl::LqrControl;

  // std::vector version
  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   * @param[in] N specifies the error-effort cross penalty. [NOT YET IMPLEMENTED]
   * @param[in] Sd Covariance of disturbance matrix.
   * @param[in] Sn Covariance of noise matrix
   * @param[in] P0 Initial state covariance matrix
   */
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const std::vector<double> &A,
             const std::vector<double> &B, const std::vector<double> &C,
             const std::vector<double> &D, const std::vector<double> &Q,
             const std::vector<double> &R, const std::vector<double> &N,
             const std::vector<double> &Sd, const std::vector<double> &Sn,
             const std::vector<double> &P0);

  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   * @param[in] Sd Covariance of disturbance matrix.
   * @param[in] Sn Covariance of noise matrix
   * @param[in] P0 Initial state covariance matrix
   */
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const std::vector<double> &A,
             const std::vector<double> &B, const std::vector<double> &C,
             const std::vector<double> &D, const std::vector<double> &Q,
             const std::vector<double> &R, const std::vector<double> &Sd,
             const std::vector<double> &Sn, const std::vector<double> &P0)
      : LqgControl(XDoF, UDoF, YDoF, discretize, u_feedback, dt, A, B, C, D, Q, R,
                   std::vector<double>(XDoF * UDoF, 0.0), Sd, Sn, P0){};

  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   * @param[in] Sd Covariance of disturbance matrix.
   * @param[in] Sn Covariance of noise matrix
   */
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const std::vector<double> &A,
             const std::vector<double> &B, const std::vector<double> &C,
             const std::vector<double> &D, const std::vector<double> &Q,
             const std::vector<double> &R, const std::vector<double> &Sd,
             const std::vector<double> &Sn)
      : LqgControl(XDoF, UDoF, YDoF, discretize, u_feedback, dt, A, B, C, D, Q, R,
                   std::vector<double>(XDoF * UDoF, 0.0), Sd, Sn,
                   std::vector<double>(XDoF * XDoF, 1e-12)){};

  // Eigen::MatrixXd version
  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   * @param[in] N specifies the error-effort cross penalty. [NOT YET IMPLEMENTED]
   * @param[in] Sd Covariance of disturbance matrix.
   * @param[in] Sn Covariance of noise matrix
   * @param[in] P0 Initial state covariance matrix
   */
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const Eigen::MatrixXd &A,
             const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
             const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &N,
             const Eigen::MatrixXd &Sd, const Eigen::MatrixXd &Sn, const Eigen::MatrixXd &P0);

  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   * @param[in] Sd Covariance of disturbance matrix.
   * @param[in] Sn Covariance of noise matrix
   * @param[in] P0 Initial state covariance matrix
   */
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const Eigen::MatrixXd &A,
             const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
             const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &Sd,
             const Eigen::MatrixXd &Sn, const Eigen::MatrixXd &P0)
      : LqgControl(XDoF, UDoF, YDoF, discretize, u_feedback, dt, A, B, C, D, Q, R,
                   Eigen::MatrixXd::Zero(XDoF, UDoF), Sd, Sn, P0){};

  /**
   * @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
   * @param[in] A specifies the discrete-time state space equation.
   * @param[in] B specifies the discrete-time state space equation.
   * @param[in] C specifies the discrete-time state space equation.
   * @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
   * @param[in] Q specifies the state space error penalty.
   * @param[in] R specifies the control effort penalty.
   * @param[in] Sd Covariance of disturbance matrix.
   * @param[in] Sn Covariance of noise matrix
   */
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const bool &u_feedback, const double &dt, const Eigen::MatrixXd &A,
             const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D,
             const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &Sd,
             const Eigen::MatrixXd &Sn)
      : LqgControl(XDoF, UDoF, YDoF, discretize, u_feedback, dt, A, B, C, D, Q, R,
                   Eigen::MatrixXd::Zero(XDoF, UDoF), Sd, Sn,
                   Eigen::MatrixXd::Constant(XDoF, XDoF, 1e-12)){};

  virtual ~LqgControl() { delete (this->optimal_state_estimate); };

  void initializeStates(std::vector<double> &X0);

  void initializeStates(const Eigen::VectorXd &X0);

  void initializeStates(std::vector<double> &X0, std::vector<double> &P0);

  void initializeStates(const Eigen::VectorXd &X0, const Eigen::MatrixXd &P0);

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> getPrediction() {
    if (this->predicted_) return std::make_pair(this->Y_, this->sigmaMeasurements_);
    this->predicted_ = true;
    std::tie(this->Y_, this->sigmaMeasurements_) =
        this->optimal_state_estimate->predict(this->U_act_);
    return std::make_pair(this->Y_, this->sigmaMeasurements_);
  };

  void updateMeasurementCov(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  void updateMeasurementCov(Eigen::MatrixXd &cov) {
    if ((size_t)(cov.cols()) != this->YDoF_) return;
    if ((size_t)(cov.rows()) != this->YDoF_) return;
    this->sigmaMeasurements_ = cov;
  };

  bool gaussianEstimatorInitialized() const {
    return this->optimal_state_estimate->isInitialized();
  };

  void controlCallback(const rclcpp::Logger &logger);

 protected:
  Eigen::MatrixXd sigmaDisturbance_, sigmaMeasurements_;
  bool predicted_;
  control::KalmanFilter *optimal_state_estimate = NULL;

};  // end of class

}  // namespace control

#endif
