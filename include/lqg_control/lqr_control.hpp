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

#ifndef LQR_CONTROL_HPP
#define LQR_CONTROL_HPP

#include <eigen3/Eigen/Dense>
#include <functional>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
// #include <iostream>

#include "lqg_control/LQR.hpp"

using control::pseudoInverse;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace control {

class LqrControl {
 public:
  LqrControl() = delete;

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
   */
  LqrControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const double &dt, const vector<double> &A, const vector<double> &B,
             const vector<double> &C, const vector<double> &D, const vector<double> &Q,
             const vector<double> &R, const vector<double> &N)
      : A_(Eigen::Map<const MatrixXd>(A.data(), XDoF, XDoF)),
        B_(Eigen::Map<const MatrixXd>(B.data(), XDoF, UDoF)),
        C_(Eigen::Map<const MatrixXd>(C.data(), YDoF, XDoF)),
        C_inv_(pseudoInverse(this->C_)),
        D_(Eigen::Map<const MatrixXd>(D.data(), YDoF, UDoF)),
        Q_(Eigen::Map<const MatrixXd>(Q.data(), XDoF, XDoF)),
        R_(Eigen::Map<const MatrixXd>(R.data(), UDoF, UDoF)),
        N_(Eigen::Map<const MatrixXd>(N.data(), XDoF, UDoF)),
        X_(VectorXd::Zero(XDoF)),
        X_des_(VectorXd::Zero(XDoF)),
        U_(VectorXd::Zero(UDoF)),
        Y_(VectorXd::Zero(YDoF)),
        XDoF_(XDoF),
        UDoF_(UDoF),
        YDoF_(YDoF) {
    // clean vector parameters (check !NULL and check size)
    if (A.size() != XDoF_ * XDoF_) throw std::runtime_error("State matrix A size is ill-formed!");
    if (B.size() != XDoF_ * UDoF_) throw std::runtime_error("State matrix B size is ill-formed!");
    if (C.size() != YDoF_ * XDoF_) throw std::runtime_error("State matrix C size is ill-formed!");
    if (D.size() != YDoF_ * UDoF_) throw std::runtime_error("State matrix D size is ill-formed!");
    if (Q.size() != XDoF_ * XDoF_) throw std::runtime_error("Weight matrix Q size is ill-formed!");
    if (R.size() != UDoF_ * UDoF_) throw std::runtime_error("Weight matrix R size is ill-formed!");
    if (N.size() != XDoF_ * UDoF_) throw std::runtime_error("Weight matrix N size is ill-formed!");
    if (discretize) {
      // Apply Tustin transformation to discretize A_.
      this->A_ = (Eigen::MatrixXd::Identity(this->XDoF_, this->XDoF_) + 0.5 * dt * this->A_) *
                 (Eigen::MatrixXd::Identity(this->XDoF_, this->XDoF_) - 0.5 * dt * this->A_)
                     .colPivHouseholderQr()
                     .solve(Eigen::MatrixXd::Identity(this->XDoF_, this->XDoF_));
      this->B_ *= dt;
    }
    // Create LQR object
    this->optimal_controller = new dLQR(this->A_, this->B_, this->Q_, this->R_, this->N_);
    // std::cout << "LQR Controller Gains: \n" << this->optimal_controller.getK()
    // << std::endl;
  };

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
             const double &dt, const vector<double> &A, const vector<double> &B,
             const vector<double> &C, const vector<double> &D, const vector<double> &Q,
             const vector<double> &R)
      : LqrControl(XDoF, UDoF, YDoF, discretize, dt, A, B, C, D, Q, R,
                   vector<double>(XDoF * UDoF, 0.0)){};

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
   */
  LqrControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &discretize,
             const double &dt, const MatrixXd &A, const MatrixXd &B, const MatrixXd &C,
             const MatrixXd &D, const MatrixXd &Q, const MatrixXd &R, const MatrixXd &N)
      : A_(A),
        B_(B),
        C_(C),
        C_inv_(pseudoInverse(this->C_)),
        D_(D),
        Q_(Q),
        R_(R),
        N_(N),
        X_(VectorXd::Zero(XDoF)),
        X_des_(VectorXd::Zero(XDoF)),
        U_(VectorXd::Zero(UDoF)),
        Y_(VectorXd::Zero(YDoF)),
        XDoF_(XDoF),
        UDoF_(UDoF),
        YDoF_(YDoF) {
    // clean vector parameters (check !NULL and check size)
    if ((size_t)(A.rows()) != XDoF_ || (size_t)(A.cols()) != XDoF_)
      throw std::runtime_error("State matrix A size is ill-formed!");
    if ((size_t)(B.rows()) != XDoF_ || (size_t)(B.cols()) != UDoF_)
      throw std::runtime_error("State matrix B size is ill-formed!");
    if ((size_t)(C.rows()) != YDoF_ || (size_t)(C.cols()) != XDoF_)
      throw std::runtime_error("State matrix C size is ill-formed!");
    if ((size_t)(D.rows()) != YDoF_ || (size_t)(D.cols()) != UDoF_)
      throw std::runtime_error("State matrix D size is ill-formed!");
    if ((size_t)(Q.rows()) != XDoF_ || (size_t)(Q.cols()) != XDoF_)
      throw std::runtime_error("Weight matrix Q size is ill-formed!");
    if ((size_t)(R.rows()) != UDoF_ || (size_t)(R.cols()) != UDoF_)
      throw std::runtime_error("Weight matrix R size is ill-formed!");
    if ((size_t)(N.rows()) != XDoF_ || (size_t)(N.cols()) != UDoF_)
      throw std::runtime_error("Weight matrix N size is ill-formed!");
    if (discretize) {
      // Apply Tustin transformation to discretize A_.
      this->A_ = (Eigen::MatrixXd::Identity(this->XDoF_, this->XDoF_) + 0.5 * dt * this->A_) *
                 (Eigen::MatrixXd::Identity(this->XDoF_, this->XDoF_) - 0.5 * dt * this->A_)
                     .colPivHouseholderQr()
                     .solve(Eigen::MatrixXd::Identity(this->XDoF_, this->XDoF_));
      this->B_ *= dt;
    }
    // Create LQR object
    this->optimal_controller = new dLQR(this->A_, this->B_, this->Q_, this->R_, this->N_);
  }

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
             const double &dt, const MatrixXd &A, const MatrixXd &B, const MatrixXd &C,
             const MatrixXd &D, const MatrixXd &Q, const MatrixXd &R)
      : LqrControl(XDoF, UDoF, YDoF, discretize, dt, A, B, C, D, Q, R,
                   MatrixXd::Zero(XDoF, UDoF)){};

  virtual ~LqrControl() { delete (this->optimal_controller); };

  virtual bool isInitialized() const {
    if (this->optimal_controller == NULL) return false;
    return this->optimal_controller->isInitialized();
  };

  virtual void setCmdToZeros() { this->U_.setZero(); };

  MatrixXd getStateFeedbackMatrix() const { return this->optimal_controller->getK(); };

  VectorXd getControl() const { return this->U_; };

  void updateMeasurement(const vector<double> &msg) {
    // Assuming all elements in the msg array are valid readings.
    // size checking
    if (msg.size() != this->YDoF_) return;
    this->Y_ = Eigen::Map<const VectorXd>(msg.data(), YDoF_);
  };

  void updateMeasurement(const vector<double> &msg, const vector<bool> &mask) {
    if (msg.size() != this->YDoF_) return;
    if (mask.size() != this->YDoF_) return;
    for (size_t i = 0; i < this->YDoF_; i++) {
      if (mask[i]) this->Y_(i) = msg[i];
    }
  };

  void updateMeasurement(const VectorXd &Y) {
    if ((size_t)(Y.size()) != this->YDoF_) return;
    this->Y_ = Y;
  };

  void updateMeasurement(const VectorXd &Y, const vector<bool> &mask) {
    if ((size_t)(Y.size()) != this->YDoF_) return;
    if (mask.size() != this->YDoF_) return;
    for (size_t i = 0; i < this->YDoF_; i++) {
      if (mask[i]) this->Y_(i) = Y(i);
    }
  };

  VectorXd getDesiredState() const { return this->X_des_; };

  MatrixXd getCinv() const { return this->C_inv_; };

  VectorXd currentError() const { return this->optimal_controller->currentError(); };

  void updateDesiredState(const vector<double> &msg) {
    // Assuming all elements in the msg array are valid readings.
    // size checking
    if (msg.size() != this->XDoF_) return;
    this->X_des_ = Eigen::Map<const VectorXd>(msg.data(), XDoF_);
  };

  void updateDesiredState(const VectorXd &X_des) {
    if ((size_t)(X_des.size()) != this->XDoF_) return;
    this->X_des_ = X_des;
  };

  void controlCallback() {
    this->X_ = this->C_inv_ * (this->Y_ - this->D_ * this->U_);
    this->U_ = this->optimal_controller->calculateControl(this->X_, this->X_des_);
  };

 protected:
  MatrixXd A_, B_, C_, C_inv_, D_, Q_, R_, N_;
  VectorXd X_, X_des_, U_, Y_;
  size_t XDoF_, UDoF_, YDoF_;
  bool discretize_ = false;

  control::dLQR *optimal_controller = NULL;

};  // end of class

}  // namespace control

#endif
