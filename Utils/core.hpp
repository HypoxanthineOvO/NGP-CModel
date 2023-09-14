#ifndef CORE_HPP_
#define CORE_HPP_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
#include <memory> // It's for Ubuntu and other Linux OS using GCC
#include <iostream>

#include "nlohmann/json.hpp"

using Vec2f = Eigen::Vector2f;
using Vec2i = Eigen::Vector2i;

using Vec3f = Eigen::Vector3f;
using Vec3i = Eigen::Vector3i;

using Vec4f = Eigen::Vector4f;
using Vec4i = Eigen::Vector4i;

using Mat3f = Eigen::Matrix3f;
using Mat3i = Eigen::Matrix3i;
using Mat4f = Eigen::Matrix4f;
using Mat4i = Eigen::Matrix4i;

using VecXf = Eigen::VectorXf;
using MatXf = Eigen::MatrixXf;


// Basic Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float INV_PI = 0.31830988618379067154f;
constexpr float EPS = 1e-5f;
constexpr float RAY_DEFAULT_MIN = 1e-5;
constexpr float RAY_DEFAULT_MAX = 1e7;

// Constants for Instant NGP
constexpr float SQRT_3 = 1.73205080756887729352f;
constexpr int NGP_STEPs = 1024;
constexpr float NGP_STEP_SIZE = SQRT_3 / NGP_STEPs;

inline Eigen::VectorXf stdvectorToEigenVector(std::vector<float>& stdv){
    return Eigen::Map<Eigen::VectorXf>(stdv.data(), stdv.size());
}



#endif // CORE_HPP_