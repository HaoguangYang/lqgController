/**
 * Kalman filter implementation using Eigen. Based on the following
 * introductory paper:
 *
 *     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 */

#ifndef _KALMAN_FILTER_H_
#define _KALMAN_FILTER_H_

#include <eigen3/Eigen/Dense>

namespace control
{

  class KalmanFilter {

  public:

    /**
    * Create a Kalman filter with the specified matrices.
    *   A - System dynamics matrix
    *   B - Control input matrix
    *   C - Output matrix
    *   Q - Process noise covariance
    *   R - Measurement noise covariance
    *   P - Estimate error covariance
    */
    KalmanFilter(
        double dt,
        const Eigen::MatrixXd& A,
        const Eigen::MatrixXd& B,
        const Eigen::MatrixXd& C,
        const Eigen::MatrixXd& Q,
        const Eigen::MatrixXd& R,
        const Eigen::MatrixXd& P
    );
    KalmanFilter(
        double dt,
        const Eigen::MatrixXd& A,
        const Eigen::MatrixXd& C,
        const Eigen::MatrixXd& Q,
        const Eigen::MatrixXd& R,
        const Eigen::MatrixXd& P
    );

    /**
    * Create a blank estimator.
    */
    KalmanFilter();

    /**
    * Initialize the filter with initial states as zero.
    */
    void init();

    /**
    * Initialize the filter with a guess for initial states.
    */
    void init(double t0, const Eigen::VectorXd& x0);

    inline bool isInitialized() const { return initialized; };

    /**
    * Update the estimated state based on measured values. The
    * time step is assumed to remain constant.
    */
    void update(const Eigen::VectorXd& y, const Eigen::VectorXd& u);
    void update(const Eigen::VectorXd& y);

    /**
    * Update the estimated state based on measured values,
    * using the given time step and dynamics matrix.
    */
    void update_time_variant_A(const Eigen::VectorXd& y, const Eigen::VectorXd& u, const Eigen::MatrixXd& A, double dt);
    void update_time_variant_A(const Eigen::VectorXd& y, const Eigen::MatrixXd& A, double dt);
    void update_time_variant_R(const Eigen::VectorXd& y, const Eigen::VectorXd& u, const Eigen::MatrixXd& R, double dt);
    void update_time_variant_R(const Eigen::VectorXd& y, const Eigen::MatrixXd& R, double dt);
    void update_time_variant_both_A_and_R
      (const Eigen::VectorXd& y, const Eigen::VectorXd& u, const Eigen::MatrixXd& A, const Eigen::MatrixXd& R, double dt);
    void update_time_variant_both_A_and_R
      (const Eigen::VectorXd& y, const Eigen::MatrixXd& A, const Eigen::MatrixXd& R, double dt);

    /**
    * Return the current state and time.
    */
    inline const Eigen::VectorXd& state() const { return x_hat; };
    inline double time() const { return t; };

  private:

    // Matrices for computation
    Eigen::MatrixXd A, B, C, Q, R, P, K, P0;

    // System dimensions
    int m, n;

    // Initial and current time
    double t0, t;

    // Discrete time step
    double dt;

    // Is the filter initialized?
    bool initialized;

    // n-size identity
    Eigen::MatrixXd I;

    // Estimated states
    Eigen::VectorXd x_hat, x_hat_new;
  };
}

#endif
