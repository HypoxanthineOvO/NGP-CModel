#include "ngp_runner.hpp"
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include "omp.h"
#include <chrono>
#include <thread>



void NGP_Runner::run() {
    Vec2i resolution = camera->getImage()->getResolution();
    int cnt = 0, tot_pixel = resolution.x() * resolution.y();

    auto begin = std::chrono::high_resolution_clock::now();

    

    for(int dy = 0; dy < resolution.y(); dy++){
        for(int dx = 0; dx < resolution.x(); dx++){
            ++cnt;
            printf("\r%.02f%%",  100 * static_cast<float>(cnt) / static_cast<float>(tot_pixel));
            
            //printf("Count: %d, Total Count: %d\n", cnt, tot_pixel);
            //std::chrono::milliseconds wait_time(100);
            //std::this_thread::sleep_for(wait_time);
            
            NGP_Runner::Color color(0, 0, 0);
            float dx_f = static_cast<float>(dx), dy_f = static_cast<float>(dy);
            Ray ray = camera->generateRay(dx_f, dy_f);
            //Vec3f c;
            //float t_min, t_max;
            auto [c, t_min, t_max, cnt] = ray_marching(ray);
            color += c;
            camera->getImage()->setPixel(dx, resolution.y() - 1 - dy, color);
            
            // For Profiling
            near_depth->setPixel(dx, dy, Vec3f(t_min, t_min, t_min));
            sp_far->setPixel(dx, dy, Vec3f(t_max, t_max, t_max));
            sample_points_counter += cnt;
            point_counter->setPixel(dx, dy, Vec3f(cnt, cnt, cnt));
        }
    }
    puts("");
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("Time Used: %.4f sec\n", elapsed.count() * 1e-9);
    printf("Total Sample Points: %d\n", sample_points_counter);
    printf("FPS under 100 MHZ: %.4f\n", (1e8) / (sample_points_counter));
}

NGP_Runner::Ray_Marching_Result NGP_Runner::ray_marching(const Ray& ray, float rand_offset) {
    /* Ray Marching for each ray */
    Color color(0, 0, 0);
    float opacity = 0.0f;
    float t0 = ray.getTMin() + rand_offset, t1 = ray.getTMax();

    float t = t0;

    int cnt = 0;

    float t_min = 1e3, t_max = 0;

    int CoarseSamplingPointsCounter = 0,FineSamplingPointsCounter = 0;
    
    while(t < t1){
        // auto coarse_sp_begin = std::chrono::high_resolution_clock::now();
        Vec3f pos = ray(t);
        t += NGP_STEP_SIZE;
        CoarseSamplingPointsCounter++;

        if (occupancy_grid->isOccupy(pos)){
            if(t < t_min) t_min = t;
            if(t > t_max) t_max = t;
            // auto coarse_sp_end = std::chrono::high_resolution_clock::now();
            // coarse_sp_time += std::chrono::duration_cast<std::chrono::nanoseconds>(coarse_sp_end - coarse_sp_begin).count() * 1e-9;
            // auto fine_sp_begin = std::chrono::high_resolution_clock::now();
            FineSamplingPointsCounter++;
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

            // auto fine_sp_end = std::chrono::high_resolution_clock::now();
            // fine_sp_time += std::chrono::duration_cast<std::chrono::nanoseconds>(fine_sp_end - fine_sp_begin).count() * 1e-9;
            cnt++;
            if(opacity > 0.99 || cnt > 128) break;
        }
    }
    coarse_sample_points.push_back(CoarseSamplingPointsCounter);
    fine_sample_points.push_back(FineSamplingPointsCounter);
    return std::make_tuple(color, t_min, t_max, cnt);
}

float NGP_Runner::get_initial_t(const Ray& r){
    // Occupancy Grid: (-0.5)^3 ~ (1.5)^3
    float tx1 = (1.0 - r.getOrigin().x()) / r.getDirection().x(),
        tx2 = (0.0 - r.getOrigin().x()) / r.getDirection().x(),
        ty1 = (1.0 - r.getOrigin().y()) / r.getDirection().y(),
        ty2 = (0.0 - r.getOrigin().y()) / r.getDirection().y(),
        tz1 = (1.0 - r.getOrigin().z()) / r.getDirection().z(),
        tz2 = (0.0 - r.getOrigin().z()) / r.getDirection().z();
    float tx_min = utils::min(tx1, tx2), tx_max = utils::max(tx1, tx2),
    ty_min = utils::min(ty1, ty2), ty_max = utils::max(ty1, ty2),
    tz_min = utils::min(tz1, tz2), tz_max = utils::max(tz1, tz2);

    float t_enter = utils::max(utils::max(tx_min, ty_min), tz_min),
        t_exit = utils::min(utils::min(tx_max, ty_max), tz_max);
    if ((t_enter < t_exit) && (t_exit >= 0)) return t_enter;
    else return -10086.0f;
}

/* Data Loader */
void NGP_Runner::loadParameters(std::string path){
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

    // Save Sigma MLP Parameters To txt file
    // std::cout << size_hashnet << std::endl;
    // std::ofstream f;
    // f.open("Sigma_MLP_Params.txt");
    // for(int i = 0; i < size_hashnet; i++) {
    //     f << sig_mlp_params[i] << " ";
    // }
    // std::cout << "Save Sigma MLP" << std::endl;
    // f.close();

    // Test Sigma MLP
    //VecXf input_vec = VecXf::Ones(32);
    //std::cout << input_vec << std::endl;
    //std::cout << sig_mlp->inference(input_vec) << std::endl;
    
    
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