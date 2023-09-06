#include "ngp_runner.hpp"
#include <iostream>
#include <fstream>


NGP_Runner::Color NGP_Runner::ray_marching(const Ray& ray, float rand_offset) {
    /* Ray Marching for each ray */
    Color color(0, 0, 0);
    float opacity = 0.0f;
    float t0 = ray.getTMin() + rand_offset, t1 = ray.getTMax();
    
    float t = t0;

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

            if(opacity > 0.99) break;
        }
    }
    return color;
}

void NGP_Runner::run() {
    Vec2i resolution = camera->getImage()->getResolution();
    int cnt = 0, tot_pixel = resolution.x() * resolution.y();
    #pragma omp parallel for shared(cnt) num_threads(24)
    for(int dy = 0; dy < resolution.y(); dy++){
        
        for(int dx = 0; dx < resolution.x(); dx++){
            #pragma omp atomic
            printf("\r%.02f%%",  static_cast<float>(100 * cnt / tot_pixel));
            ++cnt;
            NGP_Runner::Color color(0, 0, 0);
            float dx_f = static_cast<float>(dx), dy_f = static_cast<float>(dy);
            Ray ray = camera->generateRay(dx_f, dy_f);
            color += ray_marching(ray);
            camera->getImage()->setPixel(dx, resolution.y() - 1 - dy, color);
        }
    }
    puts("");
}