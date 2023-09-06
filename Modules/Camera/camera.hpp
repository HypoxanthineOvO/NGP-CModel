#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include "core.hpp"
#include "image.hpp"
#include "ray.hpp"
#include "config.hpp"

static const float SCALE = 0.33;

class Camera {
public:
    Camera():img_w(800), img_h(800){}; //
    Camera(const Config::Snapshot::Camera& config, std::shared_ptr<Image>& image):
        image(image), camera_angle_x(config.camera_angle_x),
        img_w(image->getResolution().x()), img_h(image->getResolution().y()){
            initFromCameraMatrix(config.matrix);
        };

    Ray generateRay(float x, float y);
    
    void initFromCameraMatrix(const float(&matrix)[12]);


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