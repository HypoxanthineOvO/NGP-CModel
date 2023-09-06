#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include "core.hpp"
#include <string>
#include <vector>
#include <map>

struct Config{
    struct Pos_encoding{
        int base_resolution;
        int log2_hashmap_size;
        int n_features_per_level;
        int n_levels;
    };
    struct Dir_encoding{
        int degree;
        int n_dims_to_encode;
    };
    struct Sig_network{
        std::string activation;
        int n_hidden_layers;
        int n_neurons;
    };
    struct RGB_network{
        std::string activation;
        int n_hidden_layers;
        int n_neurons;
    };
    struct Snapshot{
        // Declare
        struct AABB{
            float max[3];
            float min[3];
        };
        int background_color[4];
        struct Camera{
            float matrix[12];
            float camera_angle_x;
            int up_dir[3];
        };
        
        int density_grid_size;
        AABB aabb;
        Camera camera;
    };

    Pos_encoding pos_encoding;
    Dir_encoding dir_encoding;
    Sig_network sig_network;
    RGB_network rgb_network;
    Snapshot snapshot;
};
#endif // CONFIG_HPP_