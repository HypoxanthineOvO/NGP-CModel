#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include "image.hpp"
#include "ngp_runner.hpp"
#include "omp.h"

std::string PATH = "./config/base.json";

int RESOLITION = 800;
std::string NAME = "lego";
std::string DATA_PATH;
int ID = 0;

int main(int argc, char* argv[]){
    if (argc > 1){
        NAME = argv[1];
    }
    if (argc > 2){
        RESOLITION = std::stoi(argv[2]);
    }
    if (argc > 3){
        ID = std::stoi(argv[3]);
    }
    Eigen::initParallel();
    omp_set_num_threads(24);

    std::cout << "Running Scene " << NAME << std::endl;

    nlohmann::json configs, camera_configs;
    
    std::ifstream fin;
    fin.open(PATH);
    std::cout << "Read Config From " << PATH << std::endl;
    fin >> configs;
    fin.close();

    DATA_PATH = "./data/nerf_synthetic/" + NAME + "/" + "transforms_test.json";
    fin.open(DATA_PATH);
    fin >> camera_configs;
    fin.close();

    
    /* Generate Camera and Image */
    std::cout << "Initializing..." << std::endl;
    std::shared_ptr<Image> img = 
        std::make_shared<Image>(RESOLITION, RESOLITION);
    std::shared_ptr<Camera> camera =
        std::make_shared<Camera>(
            camera_configs, img, ID
        );

    /* Generate NGP Runner */
    std::shared_ptr<OccupancyGrid> ocgrid =
        std::make_shared<OccupancyGrid>(
            128, -0.5, 1.5
        );
    // Two MLP
    std::shared_ptr<MLP> sigma_mlp =
        std::make_shared<MLP>(
            32, 16, configs.at("network")
        );
    std::shared_ptr<MLP> color_mlp =
        std::make_shared<MLP>(
            32, 16, configs.at("rgb_network")
        );
    // Hash Encoding
    std::shared_ptr<HashEncoding> hashenc =
        std::make_shared<HashEncoding>(
            configs.at("encoding")
        );
    // SH Encoding
    std::shared_ptr<SHEncoding> shenc =
        std::make_shared<SHEncoding>(
            configs.at("dir_encoding").at("nested")[0]
        );
    // NGP Runner
    NGP_Runner ngp_runner(
        camera, ocgrid, sigma_mlp, color_mlp, hashenc, shenc
    );
    ngp_runner.loadParameters("./snapshots/Hash19_Float/" + NAME + ".msgpack");
    //std::cout << "Load msgpack from " << "./snapshots/BaseData/" + NAME + ".msgpack" << std::endl;

    std::cout << "Running..." << std::endl;
    /* Run Instant NGP */
    //auto begin = std::chrono::high_resolution_clock::now();
    ngp_runner.run();
    //auto end = std::chrono::high_resolution_clock::now();
    //auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    //printf("Time Used: %.4f sec\n", elapsed.count() * 1e-9);

    /* Profiler */
    //ngp_runner.Profiling();

    /* Write Image */
    ngp_runner.writeImage();
    // ngp_runner.save_near_and_far();
    // ngp_runner.Profiling();
    
    ngp_runner.save_counter();
    std::cout << "Done!" << std::endl;
}