#ifndef CONFIG_IO_HPP_
#define CONFIG_IO_HPP_

#define JSON_USE_IMPLICIT_CONVERSIONS 0

#include <nlohmann/json.hpp>
#include "config.hpp"

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Config::Pos_encoding, base_resolution, log2_hashmap_size, n_features_per_level, n_levels
);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Config::Dir_encoding, degree, n_dims_to_encode
);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Config::Sig_network, activation, n_hidden_layers, n_neurons
);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Config::RGB_network, activation, n_hidden_layers, n_neurons
);
// Snapshots
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Config::Snapshot::AABB, max, min
);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Config::Snapshot::Camera, matrix, camera_angle_x, up_dir
);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Config::Snapshot, aabb, background_color, camera, density_grid_size
);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    Config, pos_encoding, dir_encoding, sig_network, rgb_network, snapshot
);

#endif // CONFIG_IO_HPP_