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

#ifndef LQR_CONTROL_HPP
#define LQR_CONTROL_HPP

#include <thread>
#include <mutex>
#include <vector>
#include <eigen3/Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "LQR.hpp"

namespace control
{

class LqrControl :
{
  public:
    LqrControl();

    //LqrControl(Eigen::MatrixXd& K_predef);
    
    LqrControl(int& XDoF, int& UDoF, int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& N = NULL);
    
    
    int matrixPack(std::vector<double>& in, Eigen::MatrixXd& out);

    inline void setCmdToZeros();

    inline Eigen::VectorXd getControl();

    void updateActualControl(const Eigen::VectorXd& u);

    Eigen::VectorXd getPrediction();

    void updateMeasurement(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void updateMeasurement(Eigen::VectorXd& Y);

    inline Eigen::VectorXd getDesiredState();

    void updateDesiredState(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void updateDesiredState(Eigen::VectorXd& X_des);

    void controlCallback();

  private:
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

    control::dLQR optimal_controller;

    mutable std::mutex _mtx;
}; // end of class

} // end of namespace

#endif
