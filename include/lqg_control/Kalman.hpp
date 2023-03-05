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

// #include <iostream>
// #include <stdexcept>

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

  KalmanFilter(const Eigen::MatrixXd &A, const Eigen::MatrixXd &C)
      : KalmanFilter(A, Eigen::MatrixXd(), C, Eigen::MatrixXd::Zero(A.rows(), A.rows()),
                     Eigen::MatrixXd::Zero(C.rows(), C.rows()),
                     Eigen::MatrixXd::Zero(A.cols(), A.cols())){};

  KalmanFilter(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C)
      : KalmanFilter(A, B, C, Eigen::MatrixXd::Zero(A.rows(), A.rows()),
                     Eigen::MatrixXd::Zero(C.rows(), C.rows()),
                     Eigen::MatrixXd::Zero(A.cols(), A.cols())){};

  KalmanFilter(const Eigen::MatrixXd &A, const Eigen::MatrixXd &C, const Eigen::MatrixXd &Q,
               const Eigen::MatrixXd &R, const Eigen::MatrixXd &P)
      : KalmanFilter(A, Eigen::MatrixXd(), C, Q, R, P){};

  KalmanFilter(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C,
               const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &P)
      : A(A),
        B(B),
        C(C),
        Q(Q),
        R(R),
        P(P),
        YDoF_(C.rows()),
        XDoF_(A.rows()),
        UDoF_(B.cols()),
        initialized(false),
        x_hat(XDoF_),
        x_hat_new(XDoF_){};

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
  bool init(const Eigen::VectorXd &x0) {
    if (x0.size() != XDoF_) return false;
    this->x_hat = x0;
    this->initialized = true;
    return true;
  };

  bool init(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0) {
    if (x0.size() != XDoF_) return false;
    if (P0.rows() != XDoF_ || P0.cols() != XDoF_) return false;
    this->x_hat = x0;
    this->P = P0;
    this->initialized = true;
    return true;
  };

  bool isInitialized() const { return initialized; };

  void setProcessDisturbanceCov(Eigen::MatrixXd &Q) { this->Q = Q; };

  void setMeasurementNoiseCov(Eigen::MatrixXd &R) { this->R = R; };

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(const Eigen::VectorXd &u);
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict();

  /**
   * Update the estimated state based on measured values. The
   * time step is assumed to remain constant.
   */
  void update(const Eigen::VectorXd &y, const Eigen::VectorXd &u);
  void update(const Eigen::VectorXd &y);

  void updateNoCov(const Eigen::VectorXd &y) {
    this->x_hat = this->C.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
    if (this->x_hat.hasNaN()) this->x_hat.setZero();
    this->P.setZero();
  };

  /**
   * Update the estimated state based on measured values,
   * using the given time step and dynamics matrix.
   */
  void update_time_variant_A(const Eigen::VectorXd &y, const Eigen::VectorXd &u,
                             const Eigen::MatrixXd &A) {
    this->A = A;
    update(y, u);
  };

  void update_time_variant_A(const Eigen::VectorXd &y, const Eigen::MatrixXd &A) {
    this->A = A;
    update(y);
  };

  void update_time_variant_R(const Eigen::VectorXd &y, const Eigen::VectorXd &u,
                             const Eigen::MatrixXd &R) {
    this->R = R;
    update(y, u);
  };

  void update_time_variant_R(const Eigen::VectorXd &y, const Eigen::MatrixXd &R) {
    this->R = R;
    update(y);
  };

  void update_time_variant_both_A_and_R(const Eigen::VectorXd &y, const Eigen::VectorXd &u,
                                        const Eigen::MatrixXd &A, const Eigen::MatrixXd &R) {
    this->A = A;
    this->R = R;
    update(y, u);
  };

  void update_time_variant_both_A_and_R(const Eigen::VectorXd &y, const Eigen::MatrixXd &A,
                                        const Eigen::MatrixXd &R) {
    this->A = A;
    this->R = R;
    update(y);
  };

  /**
   * Return the current state and time.
   */
  Eigen::VectorXd state() const { return x_hat; };

 private:
  // Matrices for computation
  Eigen::MatrixXd A, B, C, Q, R, P, K, P0;

  // System dimensions
  int YDoF_, XDoF_, UDoF_;

  // Is the filter initialized?
  bool initialized = false;

  bool predicted_;

  // n-size identity
  Eigen::MatrixXd y_pred_cov_;

  // Estimated states
  Eigen::VectorXd x_hat, x_hat_new, y_pred_;
};
}  // namespace control

#endif
