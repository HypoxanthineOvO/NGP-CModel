#ifndef SHENCODING_HPP_
#define SHENCODING_HPP_

#include <vector>

#include "core.hpp"
#include "utils.hpp"

class SHEncoding{
public:
    using Direction = Eigen::Vector3f;
    using Feature = Eigen::VectorXf;
    SHEncoding() = delete;
    SHEncoding(const nlohmann::json& configs):
    SHEncoding(
        utils::get_int_from_json(configs, "degree"),
        utils::get_int_from_json(configs, "n_dims_to_encode")
    ){}
    SHEncoding(int degree, int n_dims_to_encode):
        degree(degree), n_dims_to_encode(n_dims_to_encode){};
    Feature encode(Direction dir);

    // Get Output Dimension
    int getOutDim() const{
        return degree * degree;
    }
private:
    int degree;
    int n_dims_to_encode;
};

#endif // SHENCODING_HPP_

