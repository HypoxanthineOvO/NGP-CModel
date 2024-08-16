#include "mlp.hpp"
#include <iostream>
#include <fstream>

void MLP::loadParametersFromFile(std::string path){
    std::ifstream nfin(path);
    std::vector<float> params(num_of_params);
    for(int i = 0; i < num_of_params; i++){
        nfin >> params[i];
    }
    loadParameters(params);
}

void MLP::loadParameters(const std::vector<float>& params){
    int idx = 0;
    for(int c = 0; c < layers[0].cols(); c++){
        for(int r = 0; r < layers[0].rows(); r++){
            float p = params[idx++];
            //if(std::abs(p) < 0.032) p = 0.0f;
            layers[0](r, c) = p;
        }
    }
    for(int j = 1; j < depth; j++){
        for(int c = 0; c < layers[j].cols(); c++){
            for(int r = 0; r < layers[j].rows(); r++){
            float p = params[idx++];
            //if(std::abs(p) < 0.032) p = 0.0f;
            layers[j](r, c) = p;
            }
        }
    }
    for(int c = 0; c < layers[depth].cols(); c++){
        for(int r = 0; r < layers[depth].rows(); r++){      
            float p = params[idx++];
            //if(std::abs(p) < 0.032) p = 0.0f;  
            layers[depth](r, c) = p;
        }
    }
}

MLP::Output MLP::inference(MLP::Input vec){
    Eigen::MatrixXf midvec = vec;
    for(auto& layer: layers){
        midvec = layer.transpose() * midvec;
        if (&layer == &layers.back()) break;
        for(int i = 0; i < midvec.size(); i++) {
            midvec(i) = ReLU(midvec(i));
        }
    }
    return midvec;
}