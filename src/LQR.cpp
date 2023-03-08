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
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "lqg_control/LQR.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace control {
dLQR::dLQR(const MatrixXd &K) : K_(K), u_(K_.rows()), e_(K_.rows()) {
  u_.setZero();
  e_.setZero();
  this->initialized = true;
}

dLQR::dLQR(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
           const MatrixXd &N)
    : K_(B.cols(), A.rows()), u_(B.cols()), e_(B.cols()) {
  // by setting u1 = u + R^-1 * N^T * x we can account for the cross-over term N. Reference:
  // https://math.stackexchange.com/questions/1777348/lqr-problem-with-interaction-term-between-state-and-control
  MatrixXd Rinv = R.colPivHouseholderQr().solve(N.transpose()).eval();
  MatrixXd AA = A - B * Rinv;
  MatrixXd QQ = Q - N * Rinv;
  MatrixXd S = dare(AA, B, QQ, R);
  std::cout << "DARE SOLUTION: \n" << S << std::endl;
  MatrixXd lhs = R + B.transpose() * S * B;
  MatrixXd rhs = B.transpose() * S * AA;
  K_ = lhs.colPivHouseholderQr().solve(rhs).eval() + Rinv;
  u_.setZero();
  e_.setZero();
  this->initialized = true;
}

MatrixXd dLQR::dare(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q,
                    const MatrixXd &R) const {
  // TODO:
  // https://math.stackexchange.com/questions/1777348/lqr-problem-with-interaction-term-between-state-and-control
  const int dim_x = A.cols();
  const int dim_u = B.cols();
  MatrixXd balancedA;
  VectorXd balanceP;
  // balancedA = inv(balanceP) * A * balanceP
  std::tie(balancedA, balanceP) = balance_matrix(A);
  double cond = pseudoInverse(balancedA).norm() * A.norm();
  if (cond > 1.e6) {
    std::cout << "[WARNING] You are using an onboard-computed controller gain matrix with \n"
              << "System Condition Number = " << cond << "\nThe gain matrix may be problematic.\n"
              << "YOU HAVE BEEN WARNED." << std::endl;
  }
  MatrixXd AinvT = (balanceP.asDiagonal() *
                    balancedA.colPivHouseholderQr().solve(MatrixXd::Identity(dim_x, dim_x)) *
                    balanceP.cwiseInverse().asDiagonal())
                       .transpose();
  MatrixXd Rinv = R.ldlt().solve(MatrixXd::Identity(dim_u, dim_u));

  // set Sympletic matrix pencil
  MatrixXd tmp2 = -B * Rinv * B.transpose() * AinvT;
  MatrixXd tmp1 = A - tmp2 * Q;
  MatrixXd tmp3 = -AinvT * Q;
  MatrixXd Sym(2 * dim_x, 2 * dim_x);
  Sym << tmp1, tmp2, tmp3, AinvT;

  // Sym.block(0, dim_x, dim_x, dim_x) = tmp;
  // Sym.block(0, 0, dim_x, dim_x) = A - tmp * Q;
  // std::cout << Sym << std::endl;
  // Sym.block(dim_x, 0, dim_x, dim_x) = -AinvT * Q;
  // Sym.block(dim_x, dim_x, dim_x, dim_x) = AinvT;

  MatrixXd balancedSym;
  VectorXd balancePS;
  std::tie(balancedSym, balancePS) = balance_matrix(Sym);
  // calc eigenvalues and eigenvectors
  Eigen::ComplexEigenSolver<MatrixXd> Eigs(balancedSym);

  Eigen::MatrixXcd U_1(dim_x, dim_x);
  Eigen::MatrixXcd U_2(dim_x, dim_x);
  Eigen::VectorXcd eigenValues = Eigs.eigenvalues();
  Eigen::MatrixXcd eigenVects = balancePS.asDiagonal() * Eigs.eigenvectors();

  // extract eigenvectors within unit circle into U1, U2
  int u_col = 0;
  for (int eigInd = 0; eigInd < 2 * dim_x; eigInd++) {
    if (std::abs(eigenValues(eigInd)) < 1.0) {
      U_1.block(0, u_col, dim_x, 1) = eigenVects.block(0, eigInd, dim_x, 1);
      U_2.block(0, u_col, dim_x, 1) = eigenVects.block(dim_x, eigInd, dim_x, 1);
      u_col++;
    }
  }
  // calc P with stable eigen vector matrix
  assert(u_col == dim_x && "DARE ERROR: No Solution Found");
  cond = pseudoInverse(U_1).norm() * U_1.norm();
  // balancePS.tail(dim_x).asDiagonal()* ...
  // *balancePS.head(dim_x).asDiagonal().inverse()
  MatrixXd P = (U_1 * U_1.adjoint()).ldlt().solve(U_1 * U_2.adjoint()).real();
  if (cond > 1.e6) {
    std::cout << "[WARNING] You are using an onboard-computed controller "
                 "gain matrix "
                 "with \n"
              << "Condition Number = " << cond
              << "\nrecorded during generation. The gain matrix may be "
                 "problematic.\n"
              << "YOU HAVE BEEN WARNED." << std::endl;
  }

  MatrixXd ret = (P + P.transpose()) * 0.5;
  return ret;
}

/*
MatrixXd dLQR::dare(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
                    const MatrixXd &N) {
  const uint dim_x = A.rows();
  const uint dim_u = B.cols();

  // set Sympletic matrix pencil
  MatrixXd Sym = MatrixXd::Zero(2 * dim_x + dim_u, 2 * dim_x + dim_u);
  Sym.block(0, 0, dim_x, dim_x) = A;
  Sym.block(0, 2 * dim_x, dim_x, dim_u) = B;
  Sym.block(dim_x, 0, dim_x, dim_x) = -Q;
  Sym.block(dim_x, dim_x, dim_x, dim_x) = MatrixXd::Identity(dim_x, dim_x);
  Sym.block(dim_x, 2 * dim_x, dim_x, dim_u) = -N;
  Sym.block(2 * dim_x, 0, dim_u, dim_x) = N.transpose();
  Sym.block(2 * dim_x, 2 * dim_x, dim_u, dim_u) = R;
  MatrixXd J = MatrixXd::Zero(2 * dim_x + dim_u, 2 * dim_x + dim_u);
  J.block(0, 0, dim_x, dim_x) = MatrixXd::Identity(dim_x, dim_x);
  J.block(dim_x, dim_x, dim_x, dim_x) = A.transpose();
  J.block(2 * dim_x, dim_x, dim_u, dim_x) = -B.transpose();

  Eigen::HouseholderQR<MatrixXd> qr(Sym.rightCols(dim_u));
  MatrixXd Q1 = qr.householderQ();
  Eigen::MatrixXcd H = Q1.block(0, dim_u, 2 * dim_x + dim_u, 2 * dim_x).transpose() *
                       Sym.block(0, 0, 2 * dim_x + dim_u, 2 * dim_x);
  Eigen::MatrixXcd J1 = Q1.block(0, dim_u, 2 * dim_x + dim_u, 2 * dim_x).transpose() *
                        J.block(0, 0, 2 * dim_x + dim_u, 2 * dim_x);
  arma::cx_mat AA(2 * dim_x, 2 * dim_x);
  arma::cx_mat BB(2 * dim_x, 2 * dim_x);
  arma::cx_mat QQ(2 * dim_x, 2 * dim_x);
  arma::cx_mat Z(2 * dim_x, 2 * dim_x);
  arma::cx_mat HH = arma::cx_mat(H.data(), H.rows(), H.cols(), false, false);
  arma::cx_mat JJ1 = arma::cx_mat(J1.data(), J1.rows(), J1.cols(), false, false);
  char method[] = "iuc";
  arma::qz(AA, BB, QQ, Z, HH, JJ1, method);
  Eigen::MatrixXcd ZZ = Eigen::Map<Eigen::MatrixXcd>(Z.memptr(), Z.n_rows, Z.n_cols);
  std::cout << HH << std::endl;
  std::cout << Z << std::endl;

  // std::cout << u_col << std::endl;
  Eigen::MatrixXcd U00 = ZZ.block(0, 0, dim_x, dim_x);
  Eigen::MatrixXcd U10 = ZZ.block(dim_x, 0, dim_x, dim_x);
  // assert(u_col == dim_x && "DARE ERROR: No Solution Found");
  Eigen::PartialPivLU<Eigen::MatrixXcd> lu(U00);
  Eigen::MatrixXcd L = Eigen::MatrixXcd::Identity(dim_x, dim_x);
  L.triangularView<Eigen::StrictlyLower>() = lu.matrixLU().triangularView<Eigen::StrictlyLower>();
  Eigen::MatrixXcd U = lu.matrixLU().triangularView<Eigen::Upper>();
  Eigen::MatrixXcd X = (U.conjugate().transpose() * L.conjugate().transpose())
                           .colPivHouseholderQr()
                           .solve(U10.conjugate().transpose())
                           .conjugate()
                           .transpose() *
                       (lu.permutationP().transpose());
  return (X + X.conjugate().transpose()).real() * 0.5;
}
*/

MatrixXd dLQR::care(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R,
                    const MatrixXd &N) const {
  const int dim_x = A.rows();
  // const int dim_u = B.cols();

  // set Hamilton matrix
  MatrixXd Ham = MatrixXd::Zero(2 * dim_x, 2 * dim_x);
  MatrixXd Rinv = R.inverse();
  MatrixXd h1 = A - B * Rinv * N.transpose();
  MatrixXd h2 = -B * Rinv * B.transpose();
  MatrixXd h3 = N * Rinv * N.transpose() - Q;
  MatrixXd h4 = -(A - B * Rinv * N.transpose()).transpose();
  Ham << h1, h2, h3, h4;

  // calc eigenvalues and eigenvectors
  Eigen::EigenSolver<MatrixXd> Eigs(Ham);

  // check eigen values
  // std::cout << "eigen values：\n" << Eigs.eigenvalues() << std::endl;
  // std::cout << "eigen vectors：\n" << Eigs.eigenvectors() << std::endl;

  // extract stable eigenvectors into 'eigvec'
  Eigen::MatrixXcd eigvec = Eigen::MatrixXcd::Zero(2 * dim_x, dim_x);
  int j = 0;
  for (int i = 0; i < 2 * dim_x && j < dim_x; ++i) {
    if (Eigs.eigenvalues()[i].real() < 0.) {
      eigvec.col(j) = Eigs.eigenvectors().block(0, i, 2 * dim_x, 1);
      ++j;
    }
  }
  assert(j == dim_x && "CARE ERROR: No Solution Found");
  // calc P with stable eigen vector matrix
  Eigen::MatrixXcd Vs_1, Vs_2;
  Vs_1 = eigvec.block(0, 0, dim_x, dim_x);
  Vs_2 = eigvec.block(dim_x, 0, dim_x, dim_x);

  MatrixXd ret = (Vs_2 * Vs_1.inverse()).real();

  return ret;
}

std::pair<MatrixXd, VectorXd> dLQR::balance_matrix(const MatrixXd &A) const {
  // https://arxiv.org/pdf/1401.5766.pdf (Algorithm #3)
  const int p = 2;
  const double beta = 2.;  // Radix base (2)
  MatrixXd Aprime = A;
  VectorXd D = VectorXd::Ones(A.rows());
  bool converged = false;
  do {
    converged = true;
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
      double c = Aprime.col(i).lpNorm<p>();
      double r = Aprime.row(i).lpNorm<p>();
      double s = std::pow(c, p) + std::pow(r, p);
      double f = 1;
      while (c < r / beta) {
        c *= beta;
        r /= beta;
        f *= beta;
      }
      while (c >= r * beta) {
        c /= beta;
        r *= beta;
        f /= beta;
      }
      if (std::pow(c, p) + std::pow(r, p) < 0.95 * s) {
        converged = false;
        D(i) *= f;
        Aprime.col(i) *= f;
        Aprime.row(i) /= f;
      }
    }
  } while (!converged);
  return {Aprime, D};
}

VectorXd dLQR::calculateControl(const VectorXd &X, const VectorXd &Xd) {
  if (!initialized) throw std::runtime_error("LQR Controller is not initialized yet!");
  e_ = Xd - X;
  u_ = K_ * e_;
  return u_;
}

}  // End namespace control
