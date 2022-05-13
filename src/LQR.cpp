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

#include <cmath>
#include <algorithm>
#include <iostream>

#include "LQR.hpp"

namespace control
{
  dLQR::dLQR(const Eigen::MatrixXd& K)
  {
    K_ = K;
    u_ = Eigen::VectorXd::Zero(K_.rows());
    e_ = Eigen::VectorXd::Zero(K_.rows());
    this->initialized = true;
  }

  dLQR::dLQR(const Eigen::MatrixXd& A,
             const Eigen::MatrixXd& B,
             const Eigen::MatrixXd& Q,
             const Eigen::MatrixXd& R,
             const Eigen::MatrixXd& N)
  {
    Eigen::MatrixXd S = dare(A, B, Q, R, N);
    std::cout << "DARE SOLUTION: \n" << S << std::endl;
    K_ = (R+B.transpose()*S*B).colPivHouseholderQr().solve(B.transpose()*S*A + N.transpose());
    u_ = Eigen::VectorXd::Zero(K_.rows());
    e_ = Eigen::VectorXd::Zero(K_.rows());
    this->initialized = true;
  }


  Eigen::MatrixXd dLQR::dare(const Eigen::MatrixXd& A,
                                    const Eigen::MatrixXd& B,
                                    const Eigen::MatrixXd& Q,
                                    const Eigen::MatrixXd& R,
                                    const Eigen::MatrixXd& N)
  {
    //TODO: cross-prod term N is not used for now.
    const int dim_x = A.rows();
    const int dim_u = B.cols();
    Eigen::MatrixXd balancedA(dim_x, dim_x);
    Eigen::VectorXd balanceP(dim_x);
    //balancedA = inv(balanceP) * A * balanceP
    balance_matrix(A, balancedA, balanceP);
    double cond = balancedA.completeOrthogonalDecomposition().pseudoInverse().norm() * A.norm();
    if (cond>1.e6){
      std::cout << "[WARNING] You are using an onboard-computed controller gain matrix with \n" <<
                   "System Condition Number = " << cond <<
                   "\nThe gain matrix may be problematic.\n" <<
                   "YOU HAVE BEEN WARNED." << std::endl;
    }
    Eigen::MatrixXd AinvT = (balanceP.asDiagonal() * balancedA.colPivHouseholderQr().solve(
                              Eigen::MatrixXd::Identity(dim_x,dim_x)) * balanceP.cwiseInverse().asDiagonal()).transpose();
    Eigen::MatrixXd Rinv = R.ldlt().solve(Eigen::MatrixXd::Identity(dim_u,dim_u));

    // set Sympletic matrix pencil
    Eigen::MatrixXd Sym(2*dim_x,2*dim_x);
    Sym.block(0,dim_x,dim_x,dim_x) = -B*Rinv*B.transpose()*AinvT;
    Sym.block(0,0,dim_x,dim_x) = A-Sym.block(0,dim_x,dim_x,dim_x)*Q;
    Sym.block(dim_x,0,dim_x,dim_x) = -AinvT*Q;
    Sym.block(dim_x,dim_x,dim_x,dim_x) = AinvT;

    Eigen::MatrixXd balancedSym(2*dim_x,2*dim_x);
    Eigen::VectorXd balancePS(2*dim_x);
    balance_matrix(Sym, balancedSym, balancePS);
    // calc eigenvalues and eigenvectors
    Eigen::ComplexEigenSolver<Eigen::MatrixXd> Eigs(balancedSym);
    
    Eigen::MatrixXcd U_1(dim_x, dim_x);
    Eigen::MatrixXcd U_2(dim_x, dim_x);
    Eigen::VectorXcd eigenValues = Eigs.eigenvalues();
    Eigen::MatrixXcd eigenVects = balancePS.asDiagonal() * Eigs.eigenvectors();

    // extract eigenvectors within unit circle into U1, U2
    int u_col = 0;
    for(int eigInd=0; eigInd < 2*dim_x ; eigInd++ )
    {
        if( std::abs(eigenValues(eigInd)) < 1.0 )
        {
            U_1.block(0,u_col,dim_x,1) = eigenVects.block(0,eigInd,dim_x,1);
            U_2.block(0,u_col,dim_x,1) = eigenVects.block(dim_x,eigInd,dim_x,1);
            u_col ++;
        }
    }
    // calc P with stable eigen vector matrix
    assert(u_col == dim_x && "DARE ERROR: No Solution Found");
    cond = U_1.completeOrthogonalDecomposition().pseudoInverse().norm() * U_1.norm();
    //balancePS.tail(dim_x).asDiagonal()* ... *balancePS.head(dim_x).asDiagonal().inverse()
    Eigen::MatrixXd P = (U_1 * U_1.adjoint()).ldlt().solve(U_1 * U_2.adjoint()).real();
    if (cond>1.e6){
      std::cout << "[WARNING] You are using an onboard-computed controller gain matrix with \n" <<
                   "Condition Number = " << cond <<
                   "\nrecorded during generation. The gain matrix may be problematic.\n" <<
                   "YOU HAVE BEEN WARNED." << std::endl;
    }
    return (P + P.transpose())*0.5;
  }

/*
  Eigen::MatrixXd dLQR::dare(const Eigen::MatrixXd& A,
                             const Eigen::MatrixXd& B,
                             const Eigen::MatrixXd& Q,
                             const Eigen::MatrixXd& R,
                             const Eigen::MatrixXd& N)
  {
    const uint dim_x = A.rows();
    const uint dim_u = B.cols();

    // set Sympletic matrix pencil
    Eigen::MatrixXd Sym = Eigen::MatrixXd::Zero(2*dim_x+dim_u, 2*dim_x+dim_u);
    Sym.block(0,0,dim_x, dim_x) = A;
    Sym.block(0,2*dim_x, dim_x, dim_u) = B;
    Sym.block(dim_x, 0, dim_x, dim_x) = -Q;
    Sym.block(dim_x, dim_x, dim_x, dim_x) = Eigen::MatrixXd::Identity(dim_x,dim_x);
    Sym.block(dim_x, 2*dim_x, dim_x, dim_u) = -N;
    Sym.block(2*dim_x, 0, dim_u, dim_x) = N.transpose();
    Sym.block(2*dim_x, 2*dim_x, dim_u, dim_u) = R;
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2*dim_x+dim_u, 2*dim_x+dim_u);
    J.block(0,0,dim_x, dim_x) = Eigen::MatrixXd::Identity(dim_x, dim_x);
    J.block(dim_x, dim_x, dim_x, dim_x) = A.transpose();
    J.block(2*dim_x, dim_x, dim_u, dim_x) = -B.transpose();
    
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Sym.rightCols(dim_u));
    Eigen::MatrixXd Q1 = qr.householderQ();
    Eigen::MatrixXcd H = Q1.block(0,dim_u,2*dim_x+dim_u,2*dim_x).transpose() * Sym.block(0,0,2*dim_x+dim_u,2*dim_x);
    Eigen::MatrixXcd J1 = Q1.block(0,dim_u,2*dim_x+dim_u,2*dim_x).transpose() * J.block(0,0,2*dim_x+dim_u,2*dim_x);
    arma::cx_mat AA(2*dim_x, 2*dim_x);
    arma::cx_mat BB(2*dim_x, 2*dim_x);
    arma::cx_mat QQ(2*dim_x, 2*dim_x);
    arma::cx_mat Z(2*dim_x, 2*dim_x);
    arma::cx_mat HH = arma::cx_mat(H.data(), H.rows(), H.cols(), false, false);
    arma::cx_mat JJ1 = arma::cx_mat(J1.data(), J1.rows(), J1.cols(), false, false);
    char method[]="iuc";
    arma::qz(AA, BB, QQ, Z, HH, JJ1, method);
    Eigen::MatrixXcd ZZ = Eigen::Map<Eigen::MatrixXcd>(Z.memptr(), Z.n_rows, Z.n_cols);
    std::cout<< HH << std::endl;
    std::cout<< Z << std::endl;

    //std::cout << u_col << std::endl;
    Eigen::MatrixXcd U00 = ZZ.block(0,0,dim_x, dim_x);
    Eigen::MatrixXcd U10 = ZZ.block(dim_x, 0, dim_x, dim_x);
    //assert(u_col == dim_x && "DARE ERROR: No Solution Found");
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(U00);
    Eigen::MatrixXcd L = Eigen::MatrixXcd::Identity(dim_x, dim_x);
    L.triangularView<Eigen::StrictlyLower>() = lu.matrixLU().triangularView<Eigen::StrictlyLower>();
    Eigen::MatrixXcd U = lu.matrixLU().triangularView<Eigen::Upper>();
    Eigen::MatrixXcd X = (U.conjugate().transpose()*L.conjugate().transpose()).colPivHouseholderQr().solve(U10.conjugate().transpose()).conjugate().transpose()*(lu.permutationP().transpose());
    return (X+X.conjugate().transpose()).real()*0.5;
  }
*/

  Eigen::MatrixXd dLQR::care(const Eigen::MatrixXd& A,
                                    const Eigen::MatrixXd& B,
                                    const Eigen::MatrixXd& Q,
                                    const Eigen::MatrixXd& R,
                                    const Eigen::MatrixXd& N)
  {
    const int dim_x = A.rows();
    //const int dim_u = B.cols();

    // set Hamilton matrix
    Eigen::MatrixXd Ham = Eigen::MatrixXd::Zero(2 * dim_x, 2 * dim_x);
    Eigen::MatrixXd Rinv = R.inverse();
    Ham << A-B*Rinv*N.transpose(), -B*Rinv*B.transpose(), N*Rinv*N.transpose()-Q, -(A-B*Rinv*N.transpose()).transpose();

    // calc eigenvalues and eigenvectors
    Eigen::EigenSolver<Eigen::MatrixXd> Eigs(Ham);

    // check eigen values
    //std::cout << "eigen values：\n" << Eigs.eigenvalues() << std::endl;
    //std::cout << "eigen vectors：\n" << Eigs.eigenvectors() << std::endl;

    // extract stable eigenvectors into 'eigvec'
    Eigen::MatrixXcd eigvec = Eigen::MatrixXcd::Zero(2 * dim_x, dim_x);
    int j = 0;
    for (int i = 0; i < 2 * dim_x && j < dim_x ; ++i) {
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
    
    return (Vs_2 * Vs_1.inverse()).real();
  }

  void dLQR::balance_matrix(const Eigen::MatrixXd &A, Eigen::MatrixXd &Aprime, Eigen::VectorXd &D) {
    // https://arxiv.org/pdf/1401.5766.pdf (Algorithm #3)
    const int p = 2;
    const double beta = 2.; // Radix base (2)
    Aprime = A;
    D = Eigen::VectorXd::Ones(A.rows());
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
            while (c >= r*beta) {
                c /= beta;
                r *= beta;
                f /= beta;
            }
            if (std::pow(c, p) + std::pow(r, p) < 0.95*s) {
                converged = false;
                D(i) *= f;
                Aprime.col(i) *= f;
                Aprime.row(i) /= f;
            }
        }
    } while (!converged);
  }

  const Eigen::VectorXd& dLQR::calculateControl(const Eigen::VectorXd& X, const Eigen::VectorXd& Xd){
    if (!initialized)
      throw std::runtime_error("LQR Controller is not initialized yet!");
    e_ = Xd - X;
    u_ = K_*e_;
    return u_;
  }

} // End namespace control