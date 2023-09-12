#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include "image.hpp"
#include "config.hpp"
#include "config_io.hpp"
#include "ngp_runner.hpp"


std::string PATH = "./config/lego.json";

std::string NAME = "lego";

int main(){
    /* Load Config */
    Config config;
    std::ifstream fin;
    // Open the config file
    fin.open(PATH);
    if(!fin.is_open()){
        std::cerr << "Failed to open config file: " << PATH << std::endl;
        exit(0);
    }
    // Try to load config into config object and handle exceptions
    try{
        nlohmann::json j;
        fin >> j;
        nlohmann::from_json(j, config);
        fin.close();
    }
    catch(nlohmann::json::exception& ex){
        fin.close();
        std::cerr << "Error: " << ex.what() << std::endl;
        exit(-1);
    }
    std::cout << "Successfully loaded config file: " << PATH << std::endl;
    // ShowConfigInfo(config);
    
    /* Generate Camera and Image */
    std::shared_ptr<Image> img = 
        std::make_shared<Image>(800, 800);
    std::shared_ptr<Camera> camera =
        std::make_shared<Camera>(
            config.snapshot.camera, img
        );

    
    /* Generate NGP Runner */
    std::shared_ptr<OccupancyGrid> ocgrid =
        std::make_shared<OccupancyGrid>(
            config.snapshot.density_grid_size,
            -0.5, 1.5
        );

    // Two MLP
    std::shared_ptr<MLP> sigma_mlp =
        std::make_shared<MLP>(
            32, // Input size
            16, // Output size
            config.sig_network.n_hidden_layers, // Number of hidden layers
            config.sig_network.n_neurons // Number of neurons
        );
    std::shared_ptr<MLP> color_mlp =
        std::make_shared<MLP>(
            32, // Input size
            16, // Output size
            config.rgb_network.n_hidden_layers, // Number of layers
            config.rgb_network.n_neurons // Number of neurons
        );
    // Hash Encoding
    std::shared_ptr<HashEncoding> hashenc =
        std::make_shared<HashEncoding>(
            config.pos_encoding
        );
    // SH Encoding
    std::shared_ptr<SHEncoding> shenc =
        std::make_shared<SHEncoding>(
            config.dir_encoding
        );

    // hashenc->loadParametersFromFile("./data/" + NAME + "/params_hash.txt");
    
    // color_mlp->loadParametersFromFile("./data/" + NAME + "/params_7168.txt");
    
    // sigma_mlp->loadParametersFromFile("./data/" + NAME + "/params_3072.txt");

    // ocgrid->loadParametersFromFile("./data/" + NAME + "/OccupancyGrid.txt");
    
    // Final NGP Runner
    NGP_Runner ngp_runner(
        camera, ocgrid, sigma_mlp, color_mlp, hashenc, shenc
    );
    ngp_runner.loadParameters("./data/lego.msgpack");

    /* Run Instant NGP */
    ngp_runner.run();
    /* Write Image */
    ngp_runner.writeImage();
}