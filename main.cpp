#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include "image.hpp"
#include "ngp_runner.hpp"


std::string PATH = "./config/base.json";

const int RESOLITION = 800;
std::string NAME = "lego";
std::string DATA_PATH;

int main(int argc, char* argv[]){
    if (argc > 1){
        NAME = argv[1];
    }
    DATA_PATH = "./data/nerf_synthetic/" + NAME + "/" + "transforms_test.json";

    nlohmann::json configs, camera_configs;
    
    std::ifstream fin;
    fin.open(PATH);
    std::cout << "Read Config From " << PATH << std::endl;
    fin >> configs;
    fin.close();


    fin.open(DATA_PATH);
    std::cout << "Reading Camera Config From " << std::endl;
    fin >> camera_configs;
    fin.close();

    
    /* Generate Camera and Image */
    std::shared_ptr<Image> img = 
        std::make_shared<Image>(RESOLITION, RESOLITION);
    std::shared_ptr<Camera> camera =
        std::make_shared<Camera>(
            camera_configs, img
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
    ngp_runner.loadParameters("./snapshots/BigData/" + NAME + ".msgpack");

    /* Run Instant NGP */
    ngp_runner.run();
    /* Write Image */
    ngp_runner.writeImage();
}