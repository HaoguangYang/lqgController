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
#include <iostream>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "lqg_control/Kalman.hpp"
#include "lqg_control/LQR.hpp"
#include "lqg_control/lqr_control.hpp"
// #include "rclcpp/rclcpp.hpp"
// #include "std_msgs/msg/float64_multi_array.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace control {

class LqgControl : public LqrControl {
 public:
  using LqrControl::LqrControl;
  using LqrControl::updateMeasurement;

  // vector version
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
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &u_feedback,
             const bool &discretize, const double &dt, const vector<double> &A,
             const vector<double> &B, const vector<double> &C, const vector<double> &D,
             const vector<double> &Q, const vector<double> &R, const vector<double> &N,
             const vector<double> &Sd, const vector<double> &Sn, const vector<double> &P0);

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
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &u_feedback,
             const bool &discretize, const double &dt, const vector<double> &A,
             const vector<double> &B, const vector<double> &C, const vector<double> &D,
             const vector<double> &Q, const vector<double> &R, const vector<double> &Sd,
             const vector<double> &Sn, const vector<double> &P0)
      : LqgControl(XDoF, UDoF, YDoF, u_feedback, discretize, dt, A, B, C, D, Q, R,
                   vector<double>(XDoF * UDoF, 0.0), Sd, Sn, P0){};

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
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &u_feedback,
             const bool &discretize, const double &dt, const vector<double> &A,
             const vector<double> &B, const vector<double> &C, const vector<double> &D,
             const vector<double> &Q, const vector<double> &R, const vector<double> &Sd,
             const vector<double> &Sn)
      : LqgControl(XDoF, UDoF, YDoF, u_feedback, discretize, dt, A, B, C, D, Q, R,
                   vector<double>(XDoF * UDoF, 0.0), Sd, Sn, vector<double>(XDoF * XDoF, 1e-12)){};

  // MatrixXd version
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
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &u_feedback,
             const bool &discretize, const double &dt, const MatrixXd &A, const MatrixXd &B,
             const MatrixXd &C, const MatrixXd &D, const MatrixXd &Q, const MatrixXd &R,
             const MatrixXd &N, const MatrixXd &Sd, const MatrixXd &Sn, const MatrixXd &P0);

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
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &u_feedback,
             const bool &discretize, const double &dt, const MatrixXd &A, const MatrixXd &B,
             const MatrixXd &C, const MatrixXd &D, const MatrixXd &Q, const MatrixXd &R,
             const MatrixXd &Sd, const MatrixXd &Sn, const MatrixXd &P0)
      : LqgControl(XDoF, UDoF, YDoF, u_feedback, discretize, dt, A, B, C, D, Q, R,
                   MatrixXd::Zero(XDoF, UDoF), Sd, Sn, P0){};

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
  LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &u_feedback,
             const bool &discretize, const double &dt, const MatrixXd &A, const MatrixXd &B,
             const MatrixXd &C, const MatrixXd &D, const MatrixXd &Q, const MatrixXd &R,
             const MatrixXd &Sd, const MatrixXd &Sn)
      : LqgControl(XDoF, UDoF, YDoF, u_feedback, discretize, dt, A, B, C, D, Q, R,
                   MatrixXd::Zero(XDoF, UDoF), Sd, Sn, MatrixXd::Constant(XDoF, XDoF, 1e-12)){};

  virtual ~LqgControl() { delete (this->optimal_state_estimate); };

  bool isInitialized() override {
    if (this->optimal_controller == NULL) return false;
    if (this->optimal_state_estimate == NULL) return false;
    return this->optimal_controller->isInitialized() &&
           (this->optimal_state_estimate->isInitialized());
  };

  void setCmdToZeros() override {
    this->U_.setZero();
    if (!this->u_feedback_) this->U_act_ = this->U_;
  };

  void initializeStates(const vector<double> &X0) {
    VectorXd x0_;
    x0_ = Eigen::Map<const VectorXd>(X0.data(), XDoF_);
    this->optimal_state_estimate->init(x0_);
  };

  void initializeStates(const VectorXd &X0) { this->optimal_state_estimate->init(X0); };

  void initializeStates(const vector<double> &X0, const vector<double> &P0);

  void initializeStates(const VectorXd &X0, const MatrixXd &P0) {
    this->optimal_state_estimate->init(X0, P0);
  };

  void initializeStatesFromObs(const vector<double> &X0, const vector<double> &P0);

  void initializeStatesFromObs(const VectorXd &Y0, const MatrixXd &P0) {
    VectorXd x_est_ = C_inv_ * Y0;
    this->optimal_state_estimate->init(x_est_, P0);
  };

  std::tuple<VectorXd, MatrixXd, VectorXd> getPrediction() {
    VectorXd X_est;
    if (!this->optimal_state_estimate->isPredicted()) {
      std::tie(this->Y_, this->sigmaMeasurements_, X_est) =
          this->optimal_state_estimate->predict(this->U_act_);
      return {this->Y_, this->sigmaMeasurements_, X_est};
    } else
      return this->optimal_state_estimate->predict(this->U_act_);
  };

  void updateActualControl(const VectorXd &u) {
    if ((size_t)(u.size()) != this->UDoF_) return;
    this->U_act_ = u;
  };

  void updateMeasurementCov(const vector<double> &msg);

  void updateMeasurementCov(const vector<double> &msg, const vector<bool> &mask);

  void updateMeasurementCov(const MatrixXd &cov) {
    if ((size_t)(cov.cols()) != this->YDoF_) return;
    if ((size_t)(cov.rows()) != this->YDoF_) return;
    this->sigmaMeasurements_ = cov;
  };

  void updateMeasurementCov(const MatrixXd &cov, const vector<bool> &mask);

  void updateMeasurement(const vector<double> &meas, const vector<double> &cov) {
    updateMeasurement(meas);
    updateMeasurementCov(cov);
  }

  void updateMeasurement(const VectorXd &meas, const MatrixXd &cov) {
    updateMeasurement(meas);
    updateMeasurementCov(cov);
  }

  void updateMeasurement(const vector<double> &meas, const vector<double> &cov,
                         const vector<bool> &mask) {
    updateMeasurement(meas, mask);
    updateMeasurementCov(cov, mask);
  }

  void updateMeasurement(const VectorXd &meas, const MatrixXd &cov, const vector<bool> &mask) {
    updateMeasurement(meas, mask);
    updateMeasurementCov(cov, mask);
  }

  bool gaussianEstimatorInitialized() const {
    return this->optimal_state_estimate->isInitialized();
  };

  void controlCallback(const bool &fullInternalUpdate = false);

 protected:
  MatrixXd sigmaDisturbance_, sigmaMeasurements_;
  VectorXd U_act_;
  bool u_feedback_, predicted_;
  control::KalmanFilter *optimal_state_estimate = NULL;

};  // end of class

}  // namespace control

#endif
