#ifndef MLP_HPP_
#define MLP_HPP_

#include "core.hpp"
#include "utils.hpp"

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
            num_of_params += input_size * width;
            for(int idx = 1; idx < num_of_hidden_layer; idx++){
                layers.push_back(Weight(width, width));
                num_of_params += width * width;
            }
            layers.push_back(Weight(width, output_size));
            num_of_params += width * output_size;
        }
    explicit MLP(int input_size, int output_size, const nlohmann::json& configs):
        MLP(input_size, output_size, 
            utils::get_int_from_json(configs, "n_hidden_layers"),
            utils::get_int_from_json(configs, "n_neurons")){}
    
    void loadParameters(const std::vector<float>& params);
    void loadParametersFromFile(std::string path);

    Output inference(Input vec);

    int getNumParams(){
        return num_of_params;
    }

private:
    int input_size, output_size, width, depth;
    std::vector<Weight> layers;
    static float Sigmoid(float input){
        return 1.0 / (1.0 + std::exp(-input));
    }
    static float ReLU(float input){
        return std::max(0.0f, input);
    }
    int num_of_params = 0;
};

#endif // MLP_HPP_