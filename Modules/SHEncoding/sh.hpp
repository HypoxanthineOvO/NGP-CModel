#ifndef SHENCODING_HPP_
#define SHENCODING_HPP_

#include <vector>

#include "core.hpp"
#include "config.hpp"

class SHEncoding{
public:
    using Direction = Eigen::Vector3f;
    using Feature = Eigen::VectorXf;
    SHEncoding() = delete;
    SHEncoding(int degree, int n_dims_to_encode):
        degree(degree), n_dims_to_encode(n_dims_to_encode){};
    SHEncoding(const Config::Dir_encoding& config):
        degree(config.degree), n_dims_to_encode(config.n_dims_to_encode){};
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

