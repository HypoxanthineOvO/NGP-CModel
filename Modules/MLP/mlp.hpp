#ifndef MLP_HPP_
#define MLP_HPP_

#include "core.hpp"

class MLP {
public:
    using Input = Eigen::VectorXf;
    using Output = Eigen::VectorXf;
    using Weight = Eigen::MatrixXf;
    using act_fn = float(*)(float);
    explicit MLP(int input_size, int output_size, 
        int num_of_hidden_layer, int width):
        input_size(input_size), output_size(output_size), 
        width(width), depth(num_of_hidden_layer){
            layers.push_back(Weight(input_size, width));
            for(int idx = 1; idx < num_of_hidden_layer; idx++){
                layers.push_back(Weight(width, width));
            }
            layers.push_back(Weight(width, output_size));
        }
    void load_params(float* params);
    Output inference(Input vec);

private:
    int input_size, output_size, width, depth;
    std::vector<Weight> layers;
    static float Sigmoid(float input){
        return 1.0 / (1.0 + std::exp(-input));
    }
    static float ReLU(float input){
        return std::max(0.0f, input);
    }
};

#endif // MLP_HPP_