#include "camera.hpp"
#include "utils.hpp"
#include <iostream>

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

