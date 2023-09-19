#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include "core.hpp"
#include "image.hpp"
#include "ray.hpp"

static const float SCALE = 0.33;

class Camera {
public:
    Camera():img_w(800), img_h(800){}; //
    Camera(const nlohmann::json& config, std::shared_ptr<Image>& image, int Test_ID = 0):
        image(image), img_w(image->getResolution().x()), img_h(image->getResolution().y()){
            config["camera_angle_x"].get_to(camera_angle_x);
            auto matrix = config["frames"][Test_ID]["transform_matrix"];
            // Get Focal Length
            focal_length = 0.5 * static_cast<float>(img_w) / std::tan(0.5 * camera_angle_x);
            // Init position, and direction from matrix
            Eigen::MatrixXf mat(3, 4);
            
            for(int i = 0; i < 3; ++i){
                for(int j = 0; j < 4; ++j){
                    matrix[i][j].get_to(mat(i, j));
                }
            }

            auto ngp_mat = utils::nerf_matrix_to_ngp(mat);
            position = Vec3f(ngp_mat.col(3)(0), ngp_mat.col(3)(1), ngp_mat.col(3)(2));
            camera_to_world = ngp_mat.block(0, 0, 3, 3);
        };

    Ray generateRay(float dx, float dy){
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
    


    // Getters
    Vec2i getResolution(){
        return Vec2i(img_w, img_h);
    }
    Vec3f getPosition(){
        return this->position;
    }
    std::shared_ptr<Image>& getImage(){
        return this->image;
    }
private:
    // Image Parameters
    int img_w, img_h;
    std::shared_ptr<Image> image;
    // Camera Parameters
    Vec3f position;
    Mat3f camera_to_world;
    float camera_angle_x, focal_length;
};

#endif // CAMERA_HPP_