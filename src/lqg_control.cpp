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

LqgControl::LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &u_feedback,
                       const bool &discretize, const double &dt, const vector<double> &A,
                       const vector<double> &B, const vector<double> &C, const vector<double> &D,
                       const vector<double> &Q, const vector<double> &R, const vector<double> &N,
                       const vector<double> &Sd, const vector<double> &Sn, const vector<double> &P0)
    : LqrControl(XDoF, UDoF, YDoF, discretize, dt, A, B, C, D, Q, R, N),
      sigmaDisturbance_(Eigen::Map<const MatrixXd>(Sd.data(), XDoF, XDoF)),
      sigmaMeasurements_(Eigen::Map<const MatrixXd>(Sn.data(), YDoF, YDoF)),
      U_act_(UDoF),
      u_feedback_(u_feedback) {
  this->U_act_.setZero();
  if (Sd.size() != XDoF_ * XDoF_)
    throw std::runtime_error("Disturbance covariance matrix Sd size is ill-formed!");
  if (Sn.size() != YDoF_ * YDoF_)
    throw std::runtime_error("Measurement covariance matrix Sn size is ill-formed!");
  if (P0.size() != XDoF_ * XDoF_) {
    this->optimal_state_estimate =
        new KalmanFilter(this->A_, this->B_, this->C_, this->sigmaDisturbance_,
                         this->sigmaMeasurements_, MatrixXd::Constant(XDoF_, XDoF_, 1e-12));
  } else {
    MatrixXd p0Mat = Eigen::Map<const MatrixXd>(P0.data(), XDoF, XDoF);
    this->optimal_state_estimate = new KalmanFilter(
        this->A_, this->B_, this->C_, this->sigmaDisturbance_, this->sigmaMeasurements_, p0Mat);
  }
}

LqgControl::LqgControl(const int &XDoF, const int &UDoF, const int &YDoF, const bool &u_feedback,
                       const bool &discretize, const double &dt, const MatrixXd &A,
                       const MatrixXd &B, const MatrixXd &C, const MatrixXd &D, const MatrixXd &Q,
                       const MatrixXd &R, const MatrixXd &N, const MatrixXd &Sd, const MatrixXd &Sn,
                       const MatrixXd &P0)
    : LqrControl(XDoF, UDoF, YDoF, discretize, dt, A, B, C, D, Q, R, N),
      sigmaDisturbance_(Sd),
      sigmaMeasurements_(Sn),
      U_act_(UDoF),
      u_feedback_(u_feedback) {
  this->U_act_.setZero();
  if ((size_t)(Sd.rows()) != XDoF_ || (size_t)(Sd.cols()) != XDoF_)
    throw std::runtime_error("Disturbance covariance matrix Sd size is ill-formed!");
  if ((size_t)(Sn.rows()) != YDoF_ || (size_t)(Sn.cols()) != YDoF_)
    throw std::runtime_error("Measurement covariance matrix Sn size is ill-formed!");
  if ((size_t)(P0.rows()) != XDoF_ || (size_t)(P0.cols()) != XDoF_) {
    this->optimal_state_estimate =
        new KalmanFilter(this->A_, this->B_, this->C_, this->sigmaDisturbance_,
                         this->sigmaMeasurements_, MatrixXd::Constant(XDoF_, XDoF_, 1e-12));
  } else {
    this->optimal_state_estimate = new KalmanFilter(
        this->A_, this->B_, this->C_, this->sigmaDisturbance_, this->sigmaMeasurements_, P0);
  }
}

void LqgControl::initializeStates(const vector<double> &X0, const vector<double> &P0) {
  VectorXd x0_;
  x0_ = Eigen::Map<const VectorXd>(X0.data(), XDoF_);
  MatrixXd stateCov;
  if (P0.size() == XDoF_ * XDoF_) {
    stateCov = Eigen::Map<const MatrixXd>(P0.data(), XDoF_, XDoF_);
  } else {
    stateCov = MatrixXd::Ones(XDoF_, XDoF_) * 1e-12;
  }
  this->optimal_state_estimate->init(x0_, stateCov);
}

void LqgControl::initializeStatesFromObs(const vector<double> &Y0, const vector<double> &P0) {
  VectorXd y0_;
  y0_ = Eigen::Map<const VectorXd>(Y0.data(), YDoF_);
  VectorXd x_est_ = C_inv_ * y0_;
  MatrixXd stateCov;
  if (P0.size() == XDoF_ * XDoF_) {
    stateCov = Eigen::Map<const MatrixXd>(P0.data(), XDoF_, XDoF_);
  } else {
    stateCov = MatrixXd::Ones(XDoF_, XDoF_) * 1e-12;
  }
  this->optimal_state_estimate->init(x_est_, stateCov);
}

void LqgControl::updateMeasurementCov(const vector<double> &msg) {
  // Assuming all elements in the msg array are row-major valid readings.
  if (msg.size() == (this->YDoF_ + 1) * this->YDoF_ / 2) {
    // Upper triangular matrix notation
    size_t ind = 0;
    for (size_t i = 0; i < this->YDoF_; i++) {
      for (size_t j = i; j < this->YDoF_; j++) {
        this->sigmaMeasurements_(i, j) = msg[ind];
        ind++;
      }
      for (size_t j = 0; j < i; j++) {
        this->sigmaMeasurements_(i, j) = this->sigmaMeasurements_(j, i);
      }
    }
    return;
  } else if (msg.size() != this->YDoF_ * this->YDoF_)
    return;

  // Full matrix notation
  this->sigmaMeasurements_ = Eigen::Map<const MatrixXd>(msg.data(), YDoF_, YDoF_);
}

void LqgControl::updateMeasurementCov(const vector<double> &msg, const vector<bool> &mask) {
  if (mask.size() != this->YDoF_) return;
  // Assuming all elements in the msg array are row-major valid readings.
  if (msg.size() == (this->YDoF_ + 1) * this->YDoF_ / 2) {
    // Upper triangular matrix notation
    size_t ind = 0;
    for (size_t i = 0; i < this->YDoF_; i++) {
      for (size_t j = i; j < this->YDoF_; j++) {
        if (mask[i] && mask[j]) this->sigmaMeasurements_(i, j) = msg[ind];
        ind++;
      }
      for (size_t j = 0; j < i; j++) {
        this->sigmaMeasurements_(i, j) = this->sigmaMeasurements_(j, i);
      }
    }
    return;
  } else if (msg.size() != this->YDoF_ * this->YDoF_)
    return;

  // Full matrix notation
  size_t ind = 0;
  for (size_t i = 0; i < this->YDoF_; i++) {
    for (size_t j = 0; j < this->YDoF_; j++) {
      if (mask[i] && mask[j]) this->sigmaMeasurements_(i, j) = msg[ind];
      ind++;
    }
  }
}

void LqgControl::updateMeasurementCov(const MatrixXd &cov, const vector<bool> &mask) {
  if ((size_t)(cov.cols()) != this->YDoF_) return;
  if ((size_t)(cov.rows()) != this->YDoF_) return;
  if (mask.size() != this->YDoF_) return;
  for (size_t i = 0; i < this->YDoF_; i++) {
    for (size_t j = 0; j < i; j++) {
      if (mask[i] && mask[j]) {
        this->sigmaMeasurements_(i, j) = cov(i, j);
        this->sigmaMeasurements_(j, i) = cov(j, i);
      }
    }
  }
  for (size_t i = 0; i < this->YDoF_; i++) {
    if (mask[i]) this->sigmaMeasurements_(i, i) = cov(i, i);
  }
}

void LqgControl::controlCallback(const bool &fullInternalUpdate) {
  if (this->optimal_state_estimate == NULL) {
    LqrControl::controlCallback();
    return;
  }

  if (!this->optimal_state_estimate->isInitialized()) {
    this->setCmdToZeros();
    return;
  }

  if (fullInternalUpdate) {
    VectorXd x_est_tmp;
    std::tie(this->Y_, this->sigmaMeasurements_, x_est_tmp) =
        this->optimal_state_estimate->predict(this->U_act_);
    this->optimal_state_estimate->update_time_variant_R(this->Y_, this->U_act_,
                                                        this->sigmaMeasurements_);
    this->setCmdToZeros();
    // RCLCPP_WARN(logger, "State observation age: %f seconds is too stale! (timeout = %f s)\n",
    //             state_dt, this->dt_);
    return;
  }

  // summarize asynchronous updates of the state measurements.
  this->optimal_state_estimate->update_time_variant_R(this->Y_, this->U_act_,
                                                      this->sigmaMeasurements_);
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
