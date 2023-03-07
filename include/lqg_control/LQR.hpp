/**
 * Discrete time LQR controller implementation
 * @author: Haoguang Yang
 * @date: 09-27-2021
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

#ifndef _DLQR_H_
#define _DLQR_H_

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/LU>
#include <utility>
// #include <armadillo>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace control {

// method for calculating the pseudo-Inverse as recommended by Eigen developers
template <typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a,
                            double epsilon = std::numeric_limits<double>::epsilon()) {
  Eigen::JacobiSVD<_Matrix_Type_> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0);
  _Matrix_Type_ ret = svd.matrixV() *
                      (svd.singularValues().array().abs() > tolerance)
                          .select(svd.singularValues().array().inverse(), 0)
                          .matrix()
                          .asDiagonal() *
                      svd.matrixU().adjoint();
  return ret;
}

/**
 * @class dLQR
 * @brief A discrete time LQR controller class. The LQR controller minimizes the H_inf control cost,
 * as defined by: J = \sum_t[(Xd(t)-X(t))'Q(Xd(t)-X(t)) + u(t)'Ru(t) + 2*(Xd(t)-X(t))'Nu(t)].
 */
class dLQR {
 public:
  /**
   *  @brief Constructor
   *  @param[in] K is the LQR gain matrix for u=K(Xd-X), which drives X to Xd.
   */
  dLQR(const MatrixXd &K);

  /**
   *  @brief Constructor using system x[n+1]=Ax[n]+Bu[n]
   *  @param[in] A specifies the discrete-time state space equation.
   *  @param[in] B specifies the discrete-time state space equation.
   *  @param[in] Q specifies the state space error penalty.
   *  @param[in] R specifies the control effort penalty.
   *  @param[in] N specifies the error-effort cross penalty.
   */
  dLQR(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
       const MatrixXd &N);

  dLQR() = delete;

  ~dLQR() { initialized = false; };

  /**
   * @brief Calculate the ARE solution given A, B, Q, R, N matrices. This algorithm is based on the
   * work of https://github.com/TakaHoribe/Riccati_Solver/, and
   * W. F. Arnold and A. J. Laub, "Generalized eigenproblem algorithms and software for algebraic
   * Riccati equations," in Proceedings of the IEEE, vol. 72, no. 12, pp. 1746-1754, Dec. 1984,
   * doi: 10.1109/PROC.1984.13083.
   */
  MatrixXd care(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
                const MatrixXd &N);

  /**
   * @brief The DARE solver is based on:
   * https://github.com/arunabh1904/LQR-ROS/blob/master/lqr_obsavoid/src/are_solver.cpp.
   */
  MatrixXd dare(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R);

  /**
   *  @brief Input the current states and desired states, output the control values
   *  @param X is the most recently state space
   *  @param Xd is the most recently desired state space
   *  @return the control value from the LQR algorithm
   */
  VectorXd calculateControl(const VectorXd &X, const VectorXd &Xd);

  std::pair<MatrixXd, VectorXd> balance_matrix(const MatrixXd &A);

  /**
   *  @brief Get the current LQR gain.
   *  @return The current K matrix.
   */
  MatrixXd getK() const { return K_; };

  /**
   *  @brief Get the current value of the control error
   */
  VectorXd currentError() const { return e_; };

  /**
   *  @brief Get the current control value (calculated based on most recent errors)
   *  @return the current control value
   */
  VectorXd currentControl() const { return u_; };
  bool isInitialized() const { return initialized; };

 private:
  /**
   *  @brief whether the controller parameters are initialized
   */
  bool initialized = false;

  /**
   *  @brief LQR gain.
   */
  MatrixXd K_;

  /**
   *  @brief Current control value
   */
  VectorXd u_;

  /**
   *  @brief Current state space error used in LQR control algorithm
   */
  VectorXd e_;

};  // End class dLQR

}  // End namespace control

#endif