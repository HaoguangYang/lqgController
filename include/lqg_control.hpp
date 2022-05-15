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
#include "lqr_control.hpp"
#include "LQR.hpp"
#include "Kalman.hpp"

namespace control
{

class LqgControl : public control::LqrControl
{
  public:
    LqgControl();

    LqgControl(int& XDoF, int& UDoF, int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& Sd, std::vector<double>& Sn);

    LqgControl(int& XDoF, int& UDoF, int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& N,
                std::vector<double>& Sd, std::vector<double>& Sn);

    LqgControl(int& XDoF, int& UDoF, int& YDoF,
                bool& discretize, bool& u_feedback, double& dt,
                std::vector<double>& A, std::vector<double>& B,
                std::vector<double>& C, std::vector<double>& D,
                std::vector<double>& Q, std::vector<double>& R,
                std::vector<double>& N,
                std::vector<double>& Sd, std::vector<double>& Sn,
                std::vector<double>& P0);


    std::pair<Eigen::VectorXd, Eigen::MatrixXd> getPrediction();

    void updateMeasurementCov(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void updateMeasurementCov(Eigen::MatrixXd& cov);

    void controlCallback(rclcpp::Logger& logger);

  protected:
    Eigen::MatrixXd sigmaMeasurements_;
    Eigen::MatrixXd sigmaDisturbance_;
    control::KalmanFilter optimal_state_estimate;
}; // end of class

} // end of namespace

#endif
