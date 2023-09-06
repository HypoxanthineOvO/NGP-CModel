#include "camera.hpp"
#include "utils.hpp"
#include <iostream>


Mat4f nerf_matrix_to_ngp(MatXf pose, float scale = 0.33, Vec3f offset = Vec3f(0.5, 0.5, 0.5)){
    Mat4f out_mat;
    out_mat << pose(1, 0) , -pose(1, 1) , -pose(1, 2) , pose(1, 3) * scale + offset(0) , \
        pose(2, 0) , -pose(2, 1) , -pose(2, 2) , pose(2, 3) * scale + offset(1) , \
        pose(0, 0) , -pose(0, 1) , -pose(0, 2) , pose(0, 3) * scale + offset(2) ,\
        0 , 0 , 0 , 1;
    return out_mat;
}

Ray Camera::generateRay(float dx, float dy){
    Vec3f ray_o(position(0), position(1), position(2));
    Vec3f ray_d(
        (((dx + 0.5) / static_cast<float>(img_h)) - 0.5) * static_cast<float>(img_h) / focal_length,
        (((dy + 0.5) / static_cast<float>(img_w)) - 0.5) * static_cast<float>(img_w) / focal_length,
        1.0
    );
    ray_d = camera_to_world * ray_d;
    ray_d = ray_d / ray_d.norm();
    return Ray(ray_o, ray_d, 0.6f, 2.0f);
}


void Camera::initFromCameraMatrix(const float (&matrix)[12]){
    // Get Focal Length
    focal_length = 0.5 * img_w / tan(0.5 * camera_angle_x);
    // Init position, and direction from matrix
    Eigen::MatrixXf mat(3, 4);
    
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 4; ++j){
            mat(i, j) = matrix[i * 4 + j];
        }
    }

    auto ngp_mat = nerf_matrix_to_ngp(mat);
    position = Vec3f(ngp_mat.col(3)(0), ngp_mat.col(3)(1), ngp_mat.col(3)(2));
    camera_to_world = ngp_mat.block(0, 0, 3, 3);
}