#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include "core.hpp"
#include "image.hpp"
#include "ray.hpp"

static const float SCALE = 0.33;

class Camera {
public:
    Camera():img_w(800), img_h(800){}; //
    Camera(const nlohmann::json& config, std::shared_ptr<Image>& image):
        image(image), camera_angle_x(config["camera_angle_x"]),
        img_w(image->getResolution().x()), img_h(image->getResolution().y()){
            auto matrix = config["frames"][0]["transform_matrix"];
            // Get Focal Length
            focal_length = 0.5 * img_w / tan(0.5 * camera_angle_x);
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

    Ray generateRay(float x, float y);
    


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