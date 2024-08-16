#ifndef NGP_RUNNER_HPP_
#define NGP_RUNNER_HPP_

#include "core.hpp"
#include "camera.hpp"
#include "mlp.hpp"
#include "hash.hpp"
#include "sh.hpp"
#include <vector>
#include <string>

#include <tuple>

class OccupancyGrid{
public:
    OccupancyGrid(int resolution, float aabb_l, float aabb_r):
        resolution(resolution), aabb_l_f(aabb_l), aabb_r_f(aabb_r),
        aabb_l_vec(Vec3f(aabb_l, aabb_l, aabb_l)),
        aabb_r_vec(Vec3f(aabb_r, aabb_r, aabb_r)),
        size(aabb_r - aabb_l), num_of_params(resolution * resolution * resolution),
        grid(std::vector<int>(resolution * resolution * resolution)){
        };
    void loadParameters(const std::vector<int>& params){
        for(int i = 0; i < num_of_params; i++){
            grid[i] = params[i];
        }
    }

    void loadParametersFromFile(std::string file){
        std::ifstream f;
        f.open(file);
        std::vector<int> params(grid.size());
        for(int i = 0; i < num_of_params; i++){
            f >> params[i];
        }
        f.close();
        loadParameters(params);
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

    int getNumParams(){
        return num_of_params;
    }
    int getResolution(){
        return resolution;
    }

private:
    std::vector<int> grid;
    int resolution;
    int num_of_params;
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

    void loadParameters(std::string path);

    using Ray_Marching_Result = std::tuple<Color, float, float, int>;

    Ray_Marching_Result ray_marching(const Ray& ray, float rand_offset = 0.0f);
    // Utils for ray marching
    float get_initial_t(const Ray& ray);
    void run();
    void writeImage(std::string name = "./Output.png"){
        camera->getImage()->writeImgToFile(name);
    };

    void save_near_and_far() {
        // Write The data to 2 txt files
        std::ofstream near_file, far_file;
        near_file.open("./near.txt");
        far_file.open("./far.txt");
        for(int i = 0; i < camera->getResolution().x(); i++){
            for(int j = 0; j < camera->getResolution().y(); j++){
                near_file << near_depth->getPixel(i,j)(0) << " ";
                far_file << sp_far->getPixel(i,j)(0) << " ";
            }
            near_file << std::endl;
            far_file << std::endl;
        }
    }

    void save_counter() {
        std::ofstream counter_file;
        counter_file.open("./counter.txt");
        for(int i = 0; i < camera->getResolution().x(); i++){
            for(int j = 0; j < camera->getResolution().y(); j++){
                counter_file << point_counter->getPixel(i,j)(0) << " ";
            }
            counter_file << std::endl;
        }
    }
    
    /* Profiler */
    void Profiling(){
        int TotalPoints = camera->getResolution().x() * camera->getResolution().y() * NGP_STEPs;
        int Max_Valid_Points = 0, NonZeroRays = 0;
        for(int i = 0;i < 30; i++)printf("=");
        puts("");
        puts("Profiling Results");
        int coarse_sample_point = 0, fine_sample_point = 0;
        for(auto cp: coarse_sample_points){
            coarse_sample_point += cp;
        }
        for(auto fp: fine_sample_points){
            fine_sample_point += fp;
        }
        printf("Coarse Sample Points: %d, Percentage: %.4f%\n", coarse_sample_point, (100.0f * coarse_sample_point) / TotalPoints);
        printf("Fine Sample Points: %d, Percentage: %.4f%\n", fine_sample_point, (100.0f * fine_sample_point) / TotalPoints);
        puts("Time breakdown");
        printf("Coarse Sample Points Time: %.4f sec\n", coarse_sp_time);
        printf("Fine Sample Points Time: %.4f sec\n", fine_sp_time);

        // Write Sample Points to File
        std::ofstream sp_file("./sample_points_stat.txt");
        sp_file << coarse_sample_point << " " <<  fine_sample_point << std::endl;
        
    }

private:
    std::shared_ptr<Camera> camera;
    std::shared_ptr<OccupancyGrid> occupancy_grid;
    std::shared_ptr<MLP> sig_mlp, color_mlp;
    std::shared_ptr<HashEncoding> hash_encoding;
    std::shared_ptr<SHEncoding> sh_encoding;

    /* Utils for Profiler */
    std::vector<int> coarse_sample_points, fine_sample_points;
    double coarse_sp_time, fine_sp_time;
    std::shared_ptr<Image> near_depth = std::make_shared<Image>(800, 800);
    std::shared_ptr<Image> sp_far = std::make_shared<Image>(800, 800);
    int sample_points_counter = 0;
    std::shared_ptr<Image> point_counter = std::make_shared<Image>(800, 800);
};




#endif // NGP_RUNNER_HPP_