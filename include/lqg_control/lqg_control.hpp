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

#ifndef LQG_CONTROL_HPP
#define LQG_CONTROL_HPP

#include <thread>
#include <mutex>
#include <vector>
#include <utility>
#include <functional>
#include <tuple>
#include <eigen3/Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "lqg_control/LQR.hpp"
#include "lqg_control/Kalman.hpp"

namespace control
{

class LqgControl
{
  public:
    LqgControl() { this->_mtx = new std::mutex(); };

    //std::vector version

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R);

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    /// @param[in] N specifies the error-effort cross penalty. [NOT YET IMPLEMENTED]
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& N);

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    /// @param[in] Sd Covariance of disturbance matrix. The input combinations without 
    /// @param[in] Sn Covariance of noise matrix
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& Sd, std::vector<double>& Sn);

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    /// @param[in] Sd Covariance of disturbance matrix. The input combinations without 
    /// @param[in] Sn Covariance of noise matrix
    /// @param[in] P0 Initial state covariance matrix
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& Sd, std::vector<double>& Sn,
                std::vector<double>& P0);

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    /// @param[in] N specifies the error-effort cross penalty. [NOT YET IMPLEMENTED]
    /// @param[in] Sd Covariance of disturbance matrix. The input combinations without 
    /// @param[in] Sn Covariance of noise matrix
    /// @param[in] P0 Initial state covariance matrix
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& N,
                std::vector<double>& Sd, std::vector<double>& Sn,
                std::vector<double>& P0);

    //Eigen::MatrixXd version

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    /// @param[in] N specifies the error-effort cross penalty. [NOT YET IMPLEMENTED]
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& N);

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    /// @param[in] Sd Covariance of disturbance matrix. The input combinations without 
    /// @param[in] Sn Covariance of noise matrix
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn);

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    /// @param[in] Sd Covariance of disturbance matrix. The input combinations without 
    /// @param[in] Sn Covariance of noise matrix
    /// @param[in] P0 Initial state covariance matrix
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn,
                const Eigen::MatrixXd& P0);

    /// @brief Constructor using system x[n+1]=Ax[n]+Bu[n]; y[n]=Cx[n]+Du[n], with t[n+1]-t[n]=dt.
    /// @param[in] A specifies the discrete-time state space equation.
    /// @param[in] B specifies the discrete-time state space equation.
    /// @param[in] C specifies the discrete-time state space equation.
    /// @param[in] D specifies the discrete-time state space equation. [NOT YET IMPLEMENTED]
    /// @param[in] Q specifies the state space error penalty.
    /// @param[in] R specifies the control effort penalty.
    /// @param[in] N specifies the error-effort cross penalty. [NOT YET IMPLEMENTED]
    /// @param[in] Sd Covariance of disturbance matrix. The input combinations without 
    /// @param[in] Sn Covariance of noise matrix
    /// @param[in] P0 Initial state covariance matrix
    LqgControl(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& N,
                const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn,
                const Eigen::MatrixXd& P0);

    void initializeCovariances(std::vector<double>& Sd, std::vector<double>& Sn);

    void initializeCovariances(std::vector<double>& Sd, std::vector<double>& Sn,
                              std::vector<double>& P0);

    void initializeCovariances(const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn);

    void initializeCovariances(const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn,
                              const Eigen::MatrixXd& P0);

    void initializeStates(std::vector<double>& X0);

    void initializeStates(const Eigen::VectorXd& X0);

    void initializeStates(std::vector<double>& X0, std::vector<double>& P0);

    void initializeStates(const Eigen::VectorXd& X0, const Eigen::MatrixXd& P0);

    inline int vectorPack(std::vector<double>& in, Eigen::VectorXd& out) {
      // conversion from std types to Eigen types
      if ((size_t)(out.size()) != in.size())
        return -1;
      out = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(in.data(), in.size());
      return 0;
    };

    inline int matrixPack(std::vector<double>& in, Eigen::MatrixXd& out){
      // conversion from std types to Eigen types
      if ((size_t)(out.size()) != in.size())
        return -1;
      size_t ind = 0;
      for (size_t i = 0; i < (size_t)(out.rows()); i++)
      {
        for (size_t j = 0; j < (size_t)(out.cols()); j++)
        {
          out(i,j) = in[ind];
          ind ++;
        }
      }
      return 0;
    };

    inline int matrixUnpack(const Eigen::MatrixXd& in, std::vector<double>& out){
      // conversion from std types to Eigen types
      if ((size_t)(out.size()) != static_cast<long unsigned int>(in.size()))
        return -1;
      size_t ind = 0;
      for (size_t i = 0; i < (size_t)(in.rows()); i++)
      {
        for (size_t j = 0; j < (size_t)(in.cols()); j++)
        {
          out[ind] = in(i,j);
          ind ++;
        }
      }
      return 0;
    };

    inline void setCmdToZeros() {
      std::lock_guard<std::mutex> l(*_mtx);
      this->U_.setZero();
      if (!this->u_feedback_)
        this->U_act_ = this->U_;
    };

    inline const Eigen::MatrixXd& getStateFeedbackMatrix() { return this->optimal_controller.getK(); };

    inline const Eigen::VectorXd& getControl() { return this->U_; };

    inline void updateActualControl(const Eigen::VectorXd& u) {
      std::lock_guard<std::mutex> l(*_mtx);
      if ((size_t)(u.size()) != this->UDoF_)
        return;
      this->U_act_ = u;
    };

    inline std::pair<Eigen::VectorXd, Eigen::MatrixXd> getPrediction() {
      std::lock_guard<std::mutex> l(*_mtx);
      if (this->predicted_)
        return std::make_pair(this->Y_, this->sigmaMeasurements_);
      this->predicted_ = true;
      std::tie(this->Y_, this->sigmaMeasurements_) = this->optimal_state_estimate.predict(this->U_act_);
      return std::make_pair(this->Y_, this->sigmaMeasurements_);
    };
    
    void updateMeasurement(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void updateMeasurement(Eigen::VectorXd& Y);

    void updateMeasurementCov(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void updateMeasurementCov(Eigen::MatrixXd& cov);

    inline const Eigen::VectorXd& getDesiredState() { return this->X_des_; };

    inline const Eigen::VectorXd& currentError() { return this->optimal_controller.CurrentError(); };

    void updateDesiredState(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void updateDesiredState(Eigen::VectorXd& X_des);

    void controlCallback(const rclcpp::Logger& logger);

  protected:
    Eigen::MatrixXd A_;
    Eigen::MatrixXd B_;
    Eigen::MatrixXd C_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd N_;
    Eigen::VectorXd X_;
    Eigen::VectorXd X_des_;
    Eigen::VectorXd U_;
    Eigen::VectorXd U_act_;
    Eigen::VectorXd Y_;

    rclcpp::Time state_update_time_, last_control_time_;

    bool discretize_, u_feedback_, predicted_;
    double dt_;
    size_t XDoF_, UDoF_, YDoF_;

    control::dLQR optimal_controller;
    Eigen::MatrixXd sigmaMeasurements_;
    Eigen::MatrixXd sigmaDisturbance_;
    control::KalmanFilter optimal_state_estimate;

    mutable std::mutex *_mtx;
  
  private:
    inline void _LqrControlFull_(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& N);
    inline void _LqgControlFull_(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& N,
                std::vector<double>& Sd, std::vector<double>& Sn,
                std::vector<double>& P0);
    inline void _LqrControlFull_(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& N);
    inline void _LqgControlFull_(const int& XDoF, const int& UDoF, const int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                const Eigen::MatrixXd& C, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                const Eigen::MatrixXd& N,
                const Eigen::MatrixXd& Sd, const Eigen::MatrixXd& Sn,
                const Eigen::MatrixXd& P0);
}; // end of class

} // end of namespace

#endif
