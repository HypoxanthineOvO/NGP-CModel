#include "hash.hpp"

void HashEncoding::loadParametersFromFile(std::string file){
    std::ifstream f(file);
    int idx = 0;
    for (int level = 0; level < n_levels; level++){
        for(int num_feature_pairs = 0; num_feature_pairs < sizes[level]; num_feature_pairs++){
            VecXf feat(n_feature_per_level);
            for(int feat_cnt = 0; feat_cnt < n_feature_per_level; feat_cnt++){
                float value;
                f >> value;
                feat(feat_cnt) = value;
                idx++;
            }
            layers[level]->loadParameters(
                num_feature_pairs, feat
            );
        }
    }
}

void HashEncoding::loadParameters(const std::vector<float>& params){
    int idx = 0;
    for (int level = 0; level < n_levels; level++){
        for(int num_feature_pairs = 0; num_feature_pairs < sizes[level]; num_feature_pairs++){
            VecXf feat(n_feature_per_level);
            for(int feat_cnt = 0; feat_cnt < n_feature_per_level; feat_cnt++){
                float p = params[idx];
                //if(std::abs(p) < 1e-2) p = 0.0f;
                feat(feat_cnt) = p;
                idx++;
            }
            layers[level]->loadParameters(
                num_feature_pairs, feat
            );
        }
    }
}

VecXf HashEncoding::encode(Vec3f point){
    VecXf out_feature(n_feature_per_level * n_levels);
    for(int level = 0; level < n_levels; level++){
        auto scale = scales[level];
        float resolution = (std::ceil(scale)) + 1; 
        // Judge Resolution
        if (sizes[level] >= (1 << log2_hashtable_size)) resolution = 0.0f;

        float x = point.x(), y = point.y(), z = point.z();
        float x_scale = x * scale + 0.5,
            y_scale = y * scale + 0.5,
            z_scale = z * scale + 0.5;
        int x_grid = static_cast<int>(std::floor(x_scale)),
            y_grid = static_cast<int>(std::floor(y_scale)),
            z_grid = static_cast<int>(std::floor(z_scale));
        float dx = x_scale - x_grid,
            dy = y_scale - y_grid,
            dz = z_scale - z_grid;

        
        Vec3i v_000(x_grid, y_grid, z_grid),
            v_001(x_grid, y_grid, z_grid + 1),
            v_010(x_grid, y_grid + 1, z_grid),
            v_011(x_grid, y_grid + 1, z_grid + 1),
            v_100(x_grid + 1, y_grid, z_grid),
            v_101(x_grid + 1, y_grid, z_grid + 1),
            v_110(x_grid + 1, y_grid + 1, z_grid),
            v_111(x_grid + 1, y_grid + 1, z_grid + 1);
        // InterPolation

        float w_000 = (1 - dx) * (1 - dy) * (1 - dz),
            w_001 = (1 - dx) * (1 - dy) * dz,
            w_010 = (1 - dx) * dy * (1 - dz),
            w_011 = (1 - dx) * dy * dz,
            w_100 = dx * (1 - dy) * (1 - dz),
            w_101 = dx * (1 - dy) * dz,
            w_110 = dx * dy * (1 - dz),
            w_111 = dx * dy * dz;
        
        VecXf f_000, f_001, f_010, f_011, f_100, f_101, f_110, f_111;        
        f_000 = layers[level]->getFeature(v_000, resolution),
        f_001 = layers[level]->getFeature(v_001, resolution),
        f_010 = layers[level]->getFeature(v_010, resolution),
        f_011 = layers[level]->getFeature(v_011, resolution),
        f_100 = layers[level]->getFeature(v_100, resolution),
        f_101 = layers[level]->getFeature(v_101, resolution),
        f_110 = layers[level]->getFeature(v_110, resolution),
        f_111 = layers[level]->getFeature(v_111, resolution);
        

            
        auto lev_feat = f_000 * w_000 + f_001 * w_001 + f_010 * w_010 + f_011 * w_011 +
            f_100 * w_100 + f_101 * w_101 + f_110 * w_110 + f_111 * w_111;
        for(int j = 0; j < n_feature_per_level; j++){
            out_feature(level * n_feature_per_level + j) = lev_feat(j);
        }
    }
    return out_feature;
}