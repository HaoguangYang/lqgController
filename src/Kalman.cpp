/**
 * Implementation of KalmanFilter class.
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 * 
 * Adaptation made for the LQG controller
 * @author: Haoguang Yang
 * @date: 2021.10.17
 *
 * Copyright (c) 2021-2022, Haoguang Yang
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <stdexcept>

#include "Kalman.hpp"

namespace control
{
  KalmanFilter::KalmanFilter(
      double dt,
      const Eigen::MatrixXd& A,
      const Eigen::MatrixXd& B,
      const Eigen::MatrixXd& C,
      const Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& R,
      const Eigen::MatrixXd& P)
    : A(A), B(B), C(C), Q(Q), R(R), P0(P),
      m(C.rows()), n(A.rows()), dt(dt), initialized(false),
      I(n, n), x_hat(n), x_hat_new(n)
  {
    I.setIdentity();
  }

  KalmanFilter::KalmanFilter(
      double dt,
      const Eigen::MatrixXd& A,
      const Eigen::MatrixXd& C,
      const Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& R,
      const Eigen::MatrixXd& P)
    : A(A), C(C), Q(Q), R(R), P0(P),
      m(C.rows()), n(A.rows()), dt(dt), initialized(false),
      I(n, n), x_hat(n), x_hat_new(n)
  {
    I.setIdentity();
  }

  KalmanFilter::KalmanFilter() {}

  void KalmanFilter::init(double t0, const Eigen::VectorXd& x0) {
    x_hat = x0;
    P = P0;
    this->t0 = t0;
    t = t0;
    initialized = true;
  }

  void KalmanFilter::init() {
    x_hat.setZero();
    P = P0;
    t0 = 0;
    t = t0;
    initialized = true;
  }

  /*-- The version without control input --*/
  void KalmanFilter::update(const Eigen::VectorXd& y) {
    assert(initialized && "Filter is not initialized!");
    x_hat_new = A * x_hat;
    P = A*P*A.transpose() + Q;
    K = P*C.transpose()*(C*P*C.transpose() + R).colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(m,m));
    if (K.hasNaN())
      K = Eigen::MatrixXd::Identity(K.rows(), K.cols());
    x_hat_new += K * (y - C*x_hat_new);
    P = (I - K*C)*P;
    x_hat = x_hat_new;
    if (x_hat.hasNaN())
      x_hat.setZero();
    t += dt;
  }

  /*-- Update done at the beginning of NEXT control cycle, 
   *   with previous control u published and new measurements y obtained. --*/
  void KalmanFilter::update(const Eigen::VectorXd& y, const Eigen::VectorXd& u) {
    assert(initialized && "Filter is not initialized!");
    assert(B.rows()==A.rows() && "Filter is not initialized with a proper B matrix!");
    x_hat_new = A * x_hat + B * u;
    P = A*P*A.transpose() + Q;
    K = P*C.transpose()*(C*P*C.transpose() + R).colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(m,m));
    if (K.hasNaN())
      K = Eigen::MatrixXd::Identity(K.rows(), K.cols());
    x_hat_new += K * (y - C*x_hat_new);
    P = (I - K*C)*P;
    x_hat = x_hat_new;
    if (x_hat.hasNaN())
      x_hat.setZero();
    t += dt;
  }

  void KalmanFilter::update_time_variant_A(const Eigen::VectorXd& y, const Eigen::VectorXd& u, const Eigen::MatrixXd& A, double dt) {
    this->A = A;
    this->dt = dt;
    update(y, u);
  }

  void KalmanFilter::update_time_variant_A(const Eigen::VectorXd& y, const Eigen::MatrixXd& A, double dt) {
    this->A = A;
    this->dt = dt;
    update(y);
  }

  void KalmanFilter::update_time_variant_R(const Eigen::VectorXd& y, const Eigen::VectorXd& u, const Eigen::MatrixXd& R, double dt) {
    this->R = R;
    this->dt = dt;
    update(y, u);
  }

  void KalmanFilter::update_time_variant_R(const Eigen::VectorXd& y, const Eigen::MatrixXd& R, double dt) {
    this->R = R;
    this->dt = dt;
    update(y);
  }

  void KalmanFilter::update_time_variant_both_A_and_R
    (const Eigen::VectorXd& y, const Eigen::VectorXd& u, const Eigen::MatrixXd& A, const Eigen::MatrixXd& R, double dt) {
    this->A = A;
    this->R = R;
    this->dt = dt;
    update(y, u);
  }

  void KalmanFilter::update_time_variant_both_A_and_R
    (const Eigen::VectorXd& y, const Eigen::MatrixXd& A, const Eigen::MatrixXd& R, double dt) {
    this->A = A;
    this->R = R;
    this->dt = dt;
    update(y);
  }
}
