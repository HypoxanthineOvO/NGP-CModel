#ifndef NGP_RUNNER_HPP_
#define NGP_RUNNER_HPP_

#include "core.hpp"
#include "camera.hpp"
#include "mlp.hpp"
#include "hash.hpp"
#include "sh.hpp"
#include <vector>
#include <string>

class OccupancyGrid{
public:
    OccupancyGrid(int resolution, float aabb_l, float aabb_r):
        resolution(resolution), aabb_l_f(aabb_l), aabb_r_f(aabb_r),
        aabb_l_vec(Vec3f(aabb_l, aabb_l, aabb_l)),
        aabb_r_vec(Vec3f(aabb_r, aabb_r, aabb_r)),
        size(aabb_r - aabb_l), grid(std::vector<int>(resolution * resolution * resolution)){
        };
    void loadParams(std::string file){
        std::ifstream f;
        f.open(file);
        int v;
        for(int i = 0; i < resolution * resolution * resolution; i++){
            f >> v;
            grid[i] = v;
        }
        f.close();
    }
    int isOccupy(Vec3f point){
        
        for(int i = 0; i < 3; i++){
            if (point(i) < 0.0f || point(i) > 1.0f){
                return 0;
            }
        }
        Vec3f loc_vec = point * 128;
        int index = static_cast<int>(std::floor(loc_vec.x()) * resolution * resolution +
            std::floor(loc_vec.y()) * resolution + std::floor(loc_vec.z()));
        return grid[index];
    }

private:
std::vector<int> grid;
    int resolution;
    float aabb_l_f, aabb_r_f;
    Vec3f aabb_l_vec, aabb_r_vec;
    float size;

};


class NGP_Runner{
public:
    using Color = Eigen::Vector3f;
    using Feature = Eigen::VectorXf;
    
    NGP_Runner(
        std::shared_ptr<Camera> camera,
        std::shared_ptr<OccupancyGrid> occupancy_grid,
        std::shared_ptr<MLP> sig_mlp, std::shared_ptr<MLP> color_mlp,
        std::shared_ptr<HashEncoding> hash_encoding, std::shared_ptr<SHEncoding> sh_encoding
    ):
    camera(camera), occupancy_grid(occupancy_grid),
    sig_mlp(sig_mlp), color_mlp(color_mlp),
    hash_encoding(hash_encoding), sh_encoding(sh_encoding)
    {};

    Color ray_marching(const Ray& ray, float rand_offset = 0.0f);
    void run();
    void writeImage(){
        camera->getImage()->writeImgToFile("./Output.png");
    };
    

private:
    std::shared_ptr<Camera> camera;
    std::shared_ptr<OccupancyGrid> occupancy_grid;
    std::shared_ptr<MLP> sig_mlp, color_mlp;
    std::shared_ptr<HashEncoding> hash_encoding;
    std::shared_ptr<SHEncoding> sh_encoding;
};

#endif // NGP_RUNNER_HPP_