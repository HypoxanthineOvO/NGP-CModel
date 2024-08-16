#ifndef HASHENCODING_HPP_
#define HASHENCODING_HPP_

#include <vector>
#include "core.hpp"
#include "utils.hpp"
#include <fstream>

// One Layer of Multi-Hash
class HashTable {
public:
    explicit HashTable(long long hashtable_size, int n_feature_per_level):
        size(hashtable_size), n_feature_per_level(n_feature_per_level), 
        table(std::vector<VecXf>(size)){};
    void loadParameters(int index_of_hash, VecXf value){
        table[index_of_hash] = value;
    }

    VecXf getFeature(Vec3i vertex, float non_hashing_resolution = 0.0){
        int x = vertex.x(), y = vertex.y(), z = vertex.z();
        
        int index;
        if(non_hashing_resolution == 0.0){
            // Do Index_Hash
            index = (((x * 1) ^ (y * 2654435761) ^ (z * 805459861)) % size + size) % size;      
        }
        else{
            int int_scale = static_cast<int>(non_hashing_resolution);
            index = (x + y * int_scale + z * int_scale * int_scale) % size;
        }
        return table[index];
    }
private:
    long long size;
    int n_feature_per_level;
    std::vector<VecXf> table;
};

class HashEncoding {
public:

    explicit HashEncoding(const nlohmann::json& configs):
    HashEncoding(
        utils::get_int_from_json(configs, "n_features_per_level"), 
        utils::get_int_from_json(configs, "base_resolution"), 
        utils::get_int_from_json(configs, "log2_hashmap_size"),
        utils::get_int_from_json(configs, "n_levels")){}
    explicit HashEncoding(
        int n_feature_per_level, int base_resolution, int log2_hashtable_size, 
            int n_levels, float per_level_scale = 1.38191288):
        n_feature_per_level(n_feature_per_level), base_resolution(base_resolution),
            log2_hashtable_size(log2_hashtable_size), n_levels(n_levels), per_level_scale(per_level_scale){
            /* Compute Each Layer's Size*/
            long long total_features = 0;
            for(int i = 0; i < n_levels; i++){
                auto scale_raw = std::pow(2.0, i * std::log2(per_level_scale)) * base_resolution - 1.0;
                
                long long resolution = static_cast<long long>(std::ceil(scale_raw)) + 1;
                long long num_of_features_raw = static_cast<long long>(std::ceil(
                    std::pow(resolution, 3) / 8
                ) * 8);
                long long THREDHOLD = static_cast<long long>(
                    std::pow(2, log2_hashtable_size)
                );
                long long num_of_features = std::min(num_of_features_raw, THREDHOLD);
                
                sizes.push_back(num_of_features);
                scales.push_back(scale_raw);
                layers.push_back(std::make_shared<HashTable>(num_of_features, n_feature_per_level));
                total_features += num_of_features;
            }
            total_parameters = static_cast<int>(total_features * n_feature_per_level);
        };
    void loadParametersFromFile(std::string file);
    void loadParameters(const std::vector<float>& params);

    VecXf encode(Vec3f point);

    int getNumParams(){
        return total_parameters;
    }

private:
    int n_feature_per_level;
    int base_resolution;
    int log2_hashtable_size;
    int n_levels;
    int total_parameters;
    float per_level_scale;
    std::vector<std::shared_ptr<HashTable>> layers;
    std::vector<int> sizes;
    std::vector<float> scales;
};
#endif // HASHENCODING_HPP_