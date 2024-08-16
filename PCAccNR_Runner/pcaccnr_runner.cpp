#include "pcaccnr_runner.hpp"
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include "omp.h"
#include <chrono>
#include <thread>



void PCAccNR_Runner::run() {
    Vec2i resolution = camera->getImage()->getResolution();
    int cnt = 0, tot_pixel = resolution.x() * resolution.y();
    auto begin = std::chrono::high_resolution_clock::now();
    for(int dy = 0; dy < resolution.y(); dy++){
        for(int dx = 0; dx < resolution.x(); dx++){
            ++cnt;
            printf("\r%.02f%%",  100 * static_cast<float>(cnt) / static_cast<float>(tot_pixel));

            
            PCAccNR_Runner::Color color(0, 0, 0);
            float dx_f = static_cast<float>(dx), dy_f = static_cast<float>(dy);
            Ray ray = camera->generateRay(dx_f, dy_f);

            auto [c, cnt] = ray_marching(ray, dx, dy);
            color += c;
            camera->getImage()->setPixel(dx, resolution.y() - 1 - dy, color);
            
        }
    }
    puts("");
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Time Used: %.4f sec\n", elapsed.count() * 1e-9);
}

PCAccNR_Runner::Ray_Marching_Result PCAccNR_Runner::ray_marching(const Ray& ray, int dx, int dy) {
    /* Ray Marching for each ray */
    Color color(0, 0, 0);
    float opacity = 0.0f;

    int cnt = 0;

    float t = near_depth[dx + dy * camera->getImage()->getResolution().x()],
          t1 = far_depth[dx + dy * camera->getImage()->getResolution().x()];
    //if(t < 1e-3) return std::make_tuple(color, cnt);
    
    while(t < t1){
        Vec3f pos = ray(t);
        t += NGP_STEP_SIZE;

        if (occupancy_grid->isOccupy(pos)){
            Feature hash_encoding_mat = hash_encoding->encode(pos);
            
            Feature sh_encoding_mat = sh_encoding->encode((ray.getDirection() + Vec3f(1, 1, 1)) / 2);
            
            Feature hash_feature_after_mlp = sig_mlp->inference(hash_encoding_mat);


            int hash_size = hash_feature_after_mlp.size() , sh_size = sh_encoding_mat.size();
            Feature feature(hash_size + sh_size);
            for(int i = 0; i < hash_size + sh_size; i++){
                if(i < hash_size){
                    feature(i) = hash_feature_after_mlp(i);
                }
                else{
                    feature(i) = sh_encoding_mat(i - hash_size);
                }
            }

            Vec3f color_raw = color_mlp->inference(feature);
            float alpha_raw = hash_feature_after_mlp(0);

            float T = 1 - opacity;
            float alpha = 1 - std::exp(-std::exp(alpha_raw) * NGP_STEP_SIZE);
            float weight = T * alpha;
            
            opacity += weight;
            color += utils::sigmoid(color_raw) * weight;

            cnt++;
            if(opacity > 0.99) break;
        }
    }
    return std::make_tuple(color, cnt);
}

/* Data Loader */
void PCAccNR_Runner::loadParameters(std::string path){
    using namespace nlohmann;
    std::ifstream input_msgpack_file(path, std::ios::in | std::ios::binary);
    json data = json::from_msgpack(input_msgpack_file);

    json::binary_t params = data["snapshot"]["params_binary"];
    
    int size_hashnet = sig_mlp->getNumParams(), size_rgbnet = color_mlp->getNumParams(),
        size_hashgrid = hash_encoding->getNumParams();
    std::vector<float> sig_mlp_params(size_hashnet), color_mlp_params(size_rgbnet),
        hashgrid_params(size_hashgrid);
    int num_of_params = params.size();

    if (num_of_params / 2 != (size_hashgrid + size_hashnet + size_rgbnet)){
        std::cout << "Mismatched Snapshot and Config!" << std::endl;
        exit(1);
    }
    
    for(int i = 0; i < num_of_params; i += 2){
        uint32_t value = params[i] | (params[i + 1] << 8);
        int index = i / 2;
        float value_float = utils::from_int_to_float16(value);
        if(index < size_hashnet) {
            sig_mlp_params[index] = value_float;
        }
        else if(index < size_rgbnet + size_hashnet) {
            color_mlp_params[index - size_hashnet] = value_float;
        }
        else {
            hashgrid_params[index - size_hashnet - size_rgbnet] = value_float;
        }
    }
    hash_encoding->loadParameters(hashgrid_params);
    sig_mlp->loadParameters(sig_mlp_params);
    color_mlp->loadParameters(color_mlp_params);
    
    
    json::binary_t density_grid_params = data["snapshot"]["density_grid_binary"];

    int num_of_params_ocgrid = density_grid_params.size();
    int size_ocgrid = occupancy_grid->getNumParams(), resolution = occupancy_grid->getResolution();

    std::vector<int> oc_params(size_ocgrid, 0);
    for(int i = 0; i < num_of_params_ocgrid; i += 2){
        uint32_t value = density_grid_params[i] | (density_grid_params[i + 1] << 8);
        float value_float = utils::from_int_to_float16(value);
        int index = utils::inv_morton(i / 2, resolution);
        if(value_float > 0.01) oc_params[index] = 1;
        else oc_params[index] = 0;
    }
    occupancy_grid->loadParameters(oc_params);
}