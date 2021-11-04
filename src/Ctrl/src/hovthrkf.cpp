#include "hovthrkf.h"
#include <iostream>
#include <uav_utils/utils.h>

HovThrKF::HovThrKF(Parameter_t& param_) : param(param_) {}

void HovThrKF::init() {
    double mass = param.mass;
    double max_force = param.full_thrust;

    // to ensure init this after param is inited.
    ROS_ASSERT_MSG(mass > 0.1 && max_force > 9.8, "mass=%f max_force=%f", mass, max_force);

    x = Eigen::Vector2d(0.0, 0.0);
    P = Eigen::Matrix<double, 2, 2>();
    P << 0.5 * 0.5, 0, 0, 1 * 1;
    Q = Eigen::Matrix<double, 2, 2>();
    Q << 0.1 * 0.1, 0, 0, 1 * 1;
    F = Eigen::Matrix<double, 2, 2>();
    F << 1, 0, -max_force / mass, 0;
    B = Eigen::Matrix<double, 2, 1>();
    B << 0, max_force / mass;
    H = Eigen::Matrix<double, 1, 2>();
    H << 0, 1;
    R = Eigen::Matrix<double, 1, 1>();
    R << 0.001 * 0.001;
}

void HovThrKF::process(double u) {
    x = F * x + B * u;
    P = F * P * F.transpose() + Q * 0;
}

void HovThrKF::update(double a) {
    Eigen::Matrix<double, 1, 1> z;
    z << a - 9.8;
    Eigen::MatrixXd y = z - H * x;
    Eigen::Vector2d k = P * H.transpose() * (H * P * H.transpose() + R).inverse();
    x = x + k * y;
    P = (Eigen::Matrix2d::Identity() - k * H) * P;
    uav_utils::limit_range(x(0), param.hover.percent_lower_limit, param.hover.percent_higher_limit);
}

double HovThrKF::get_hov_thr() {
    return x(0);
}

void HovThrKF::set_hov_thr(double hov) {
    x(0) = hov;
}

void HovThrKF::simple_update(Eigen::Quaterniond q, double u, Eigen::Vector3d acc) {
    Eigen::Matrix3d bRw = q.toRotationMatrix().transpose();
    Eigen::Vector3d acc_body = acc - bRw * Eigen::Vector3d(0, 0, param.gra);
    static double acc_body2_filter = 0;
    acc_body2_filter = 0.8 * acc_body2_filter + 0.2 * acc_body(2);
    Vector3d acc_des = Eigen::Vector3d(0, 0, u * param.full_thrust / param.mass) - 
                       bRw * Eigen::Vector3d(0, 0, param.gra);
    double compensate = (acc_des(2) - acc_body2_filter) * 0.001;
    x(0) = x(0) + compensate;
    uav_utils::limit_range(x(0), param.hover.percent_lower_limit, param.hover.percent_higher_limit);
}

void HovThrKF::simple_update(Eigen::Vector3d des_v, Eigen::Vector3d odom_v) {
    double compensate = (des_v(2) - odom_v(2)) * 0.001;
    x(0) = x(0) + compensate;
    uav_utils::limit_range(x(0), param.hover.percent_lower_limit, param.hover.percent_higher_limit);
}