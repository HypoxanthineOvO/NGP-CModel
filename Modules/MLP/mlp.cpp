#include "mlp.hpp"
#include<iostream>

void MLP::load_params(float* params){
    int idx = 0;
    // Input-Hidden
    for(int c = 0; c < layers[0].cols(); c++){
        for(int r = 0; r < layers[0].rows(); r++){
            layers[0](r, c) = params[idx++];
        }
    }

    for(int j = 1; j < depth; j++){
        for(int c = 0; c < layers[j].cols(); c++){
            for(int r = 0; r < layers[j].rows(); r++){
                layers[j](r, c) = params[idx++];
            }
        }
    }
    for(int c = 0; c < layers[depth].cols(); c++){
        for(int r = 0; r < layers[depth].rows(); r++){        
            layers[depth](r, c) = params[idx++];
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