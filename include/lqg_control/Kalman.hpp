/**
 * Kalman filter implementation using Eigen. Based on the following
 * introductory paper:
 *
 *     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 * @author: Haoguang Yang
 * @date: 2021.10.01
 */

#ifndef _KALMAN_FILTER_H_
#define _KALMAN_FILTER_H_

#include <eigen3/Eigen/Dense>
#include <cassert>

// #include <iostream>
// #include <stdexcept>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace control {

class KalmanFilter {
 public:
  /**
   * Create a blank estimator.
   */
  KalmanFilter() = delete;

  ~KalmanFilter() = default;

  /**
   * Create a Kalman filter with the specified matrices.
   *   A - System dynamics matrix
   *   B - Control input matrix
   *   C - Output matrix
   *   Q - Process noise covariance
   *   R - Measurement noise covariance
   *   P - Estimate error covariance
   */

  KalmanFilter(const MatrixXd &A, const MatrixXd &C)
      : KalmanFilter(A, MatrixXd(), C, MatrixXd::Zero(A.rows(), A.rows()),
                     MatrixXd::Zero(C.rows(), C.rows()), MatrixXd::Zero(A.cols(), A.cols())){};

  KalmanFilter(const MatrixXd &A, const MatrixXd &B, const MatrixXd &C)
      : KalmanFilter(A, B, C, MatrixXd::Zero(A.rows(), A.rows()),
                     MatrixXd::Zero(C.rows(), C.rows()), MatrixXd::Zero(A.cols(), A.cols())){};

  KalmanFilter(const MatrixXd &A, const MatrixXd &C, const MatrixXd &Q, const MatrixXd &R,
               const MatrixXd &P)
      : KalmanFilter(A, MatrixXd(), C, Q, R, P){};

  KalmanFilter(const MatrixXd &A, const MatrixXd &B, const MatrixXd &C, const MatrixXd &Q,
               const MatrixXd &R, const MatrixXd &P)
      : XDoF_(A.rows()),
        YDoF_(C.rows()),
        UDoF_(B.cols()),
        initialized(false),
        A(A),
        B(B),
        C(C),
        Q(Q),
        R(R),
        P(P),
        K(XDoF_, YDoF_),
        y_pred_cov_(YDoF_, YDoF_),
        x_hat(XDoF_),
        x_hat_new(XDoF_),
        y_pred_(YDoF_){};

  /**
   * Initialize the filter with initial states as zero.
   */
  bool init() {
    this->x_hat.setZero();
    this->initialized = true;
    return true;
  };

  /**
   * Initialize the filter with a guess for initial states.
   */
  bool init(const VectorXd &x0) {
    if (x0.size() != XDoF_) return false;
    this->x_hat = x0;
    this->initialized = true;
    return true;
  };

  bool init(const VectorXd &x0, const MatrixXd &P0) {
    if (x0.size() != XDoF_) return false;
    if (P0.rows() != XDoF_ || P0.cols() != XDoF_) return false;
    this->x_hat = x0;
    this->P = P0;
    this->initialized = true;
    return true;
  };

  bool isInitialized() const { return initialized; };

  bool isPredicted() const { return predicted_; };

  void setProcessDisturbanceCov(MatrixXd &Q) { this->Q = Q; };

  void setMeasurementNoiseCov(MatrixXd &R) { this->R = R; };

  std::tuple<VectorXd, MatrixXd, VectorXd> predict(const VectorXd &u);
  std::tuple<VectorXd, MatrixXd, VectorXd> predict();

  /**
   * Update the estimated state based on measured values. The
   * time step is assumed to remain constant.
   */
  void update(const VectorXd &y, const VectorXd &u);
  void update(const VectorXd &y);

  void update_no_cov(const VectorXd &y) {
    this->x_hat = this->C.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
    if (this->x_hat.hasNaN()) this->x_hat.setZero();
    this->P.setZero();
  };

  /**
   * Update the estimated state based on measured values,
   * using the given time step and dynamics matrix.
   */
  void update_time_variant_A(const VectorXd &y, const VectorXd &u, const MatrixXd &A) {
    this->A = A;
    update(y, u);
  };

  void update_time_variant_A(const VectorXd &y, const MatrixXd &A) {
    this->A = A;
    update(y);
  };

  void update_time_variant_R(const VectorXd &y, const VectorXd &u, const MatrixXd &R) {
    this->R = R;
    update(y, u);
  };

  void update_time_variant_R(const VectorXd &y, const MatrixXd &R) {
    this->R = R;
    update(y);
  };

  void update_time_variant_both_A_and_R(const VectorXd &y, const VectorXd &u, const MatrixXd &A,
                                        const MatrixXd &R) {
    this->A = A;
    this->R = R;
    update(y, u);
  };

  void update_time_variant_both_A_and_R(const VectorXd &y, const MatrixXd &A, const MatrixXd &R) {
    this->A = A;
    this->R = R;
    update(y);
  };

  /**
   * Return the current state and time.
   */
  VectorXd state() const { return x_hat; };

 private:
  // System dimensions
  int XDoF_, YDoF_, UDoF_;

  // Is the filter initialized?
  bool initialized = false;

  // Matrices for computation
  MatrixXd A, B, C, Q, R, P, K;

  bool predicted_;

  // n-size identity
  MatrixXd y_pred_cov_;

  // Estimated states
  VectorXd x_hat, x_hat_new, y_pred_;
};
}  // namespace control

#endif
