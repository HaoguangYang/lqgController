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

  KalmanFilter(const double &dt, const Eigen::MatrixXd &A, const Eigen::MatrixXd &C);

  KalmanFilter(const double &dt, const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
               const Eigen::MatrixXd &C);

  KalmanFilter(const double &dt, const Eigen::MatrixXd &A, const Eigen::MatrixXd &C,
               const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::MatrixXd &P);

  KalmanFilter(const double &dt, const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
               const Eigen::MatrixXd &C, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
               const Eigen::MatrixXd &P);

  /**
   * Initialize the filter with initial states as zero.
   */
  void init();

  /**
   * Initialize the filter with a guess for initial states.
   */
  void init(const double &t0, const Eigen::VectorXd &x0);

  void init(const double &t0, const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0);

  bool isInitialized() const { return initialized; };

  void setProcessDisturbanceCov(Eigen::MatrixXd &Q) { this->Q = Q; };

  void setMeasurementNoiseCov(Eigen::MatrixXd &R) { this->R = R; };

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(const Eigen::VectorXd &u);
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict();

  /**
   * Update the estimated state based on measured values. The
   * time step is assumed to remain constant.
   */
  inline void update(const Eigen::VectorXd &y, const Eigen::VectorXd &u);
  inline void update(const Eigen::VectorXd &y);

  void updateNoCov(const Eigen::VectorXd &y);

  /**
   * Update the estimated state based on measured values,
   * using the given time step and dynamics matrix.
   */
  void update_time_variant_A(const Eigen::VectorXd &y, const Eigen::VectorXd &u,
                             const Eigen::MatrixXd &A);
  void update_time_variant_A(const Eigen::VectorXd &y, const Eigen::VectorXd &u,
                             const Eigen::MatrixXd &A, const double &dt);
  void update_time_variant_A(const Eigen::VectorXd &y, const Eigen::MatrixXd &A);
  void update_time_variant_A(const Eigen::VectorXd &y, const Eigen::MatrixXd &A, const double &dt);
  void update_time_variant_R(const Eigen::VectorXd &y, const Eigen::VectorXd &u,
                             const Eigen::MatrixXd &R);
  void update_time_variant_R(const Eigen::VectorXd &y, const Eigen::VectorXd &u,
                             const Eigen::MatrixXd &R, const double &dt);
  void update_time_variant_R(const Eigen::VectorXd &y, const Eigen::MatrixXd &R);
  void update_time_variant_R(const Eigen::VectorXd &y, const Eigen::MatrixXd &R, const double &dt);
  void update_time_variant_both_A_and_R(const Eigen::VectorXd &y, const Eigen::VectorXd &u,
                                        const Eigen::MatrixXd &A, const Eigen::MatrixXd &R);
  void update_time_variant_both_A_and_R(const Eigen::VectorXd &y, const Eigen::VectorXd &u,
                                        const Eigen::MatrixXd &A, const Eigen::MatrixXd &R,
                                        const double &dt);
  void update_time_variant_both_A_and_R(const Eigen::VectorXd &y, const Eigen::MatrixXd &A,
                                        const Eigen::MatrixXd &R);
  void update_time_variant_both_A_and_R(const Eigen::VectorXd &y, const Eigen::MatrixXd &A,
                                        const Eigen::MatrixXd &R, const double &dt);

  /**
   * Return the current state and time.
   */
  Eigen::VectorXd state() const { return x_hat; };
  double time() const { return t; };

 private:
  // Matrices for computation
  Eigen::MatrixXd A, B, C, Q, R, P, K, P0;

  // System dimensions
  int YDoF_, XDoF_, UDoF_;

  // Initial and current time
  double t;

  // Discrete time step
  double dt;

  // Is the filter initialized?
  bool initialized;

  bool predicted_;

  // n-size identity
  Eigen::MatrixXd y_pred_cov_;

  // Estimated states
  Eigen::VectorXd x_hat, x_hat_new, y_pred_;
};
}  // namespace control

#endif
