#include "sh.hpp"

SHEncoding::Feature SHEncoding::encode(SHEncoding::Direction dir) {
    // Initialize sh_coeffs
    int num_of_feature = degree * degree;
    SHEncoding::Feature sh_coeffs(num_of_feature);
    // Basic Direction Data
    float x = dir.x() * 2 - 1, y = dir.y() * 2 - 1, z = dir.z() * 2 - 1;
    // Combining data
    float x2 = x*x, y2 = y*y, z2 = z*z,
        xy = x*y, xz = x*z, yz = y*z;
    // Encoding
    if(degree < 1 || degree > 4){
        std::cerr << "Invalid Degree for SH Encoding" << std::endl;
        exit(1);
    }
    // Reference from Junran's Jax-Instant-NGP
    if(degree >= 1){
        sh_coeffs[0] = 0.28209479177387814;
    }
    if(degree >= 2){
        sh_coeffs[1] = -0.48860251190291987 * y;
        sh_coeffs[2] = 0.48860251190291987 * z;
        sh_coeffs[3] = -0.48860251190291987 * x;        
    }
    if(degree >= 3){
        sh_coeffs[4] = 1.0925484305920792 * xy;
        sh_coeffs[5] = -1.0925484305920792 * yz;
        sh_coeffs[6] = 0.94617469575755997 * z2 - 0.31539156525251999;
        sh_coeffs[7] = -1.0925484305920792 * xz;
        sh_coeffs[8] = 0.54627421529603959 * x2 - 0.54627421529603959 * y2;        
    }
    if(degree >= 4){
        sh_coeffs[9] = 0.59004358992664352 * y * (-3.0 * x2 + y2);
        sh_coeffs[10] = 2.8906114426405538 * xy * z;
        sh_coeffs[11] = 0.45704579946446572 * y * (1.0 - 5.0 * z2);
        sh_coeffs[12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0);
        sh_coeffs[13] = 0.45704579946446572 * x * (1.0 - 5.0 * z2);
        sh_coeffs[14] = 1.4453057213202769 * z * (x2 - y2);
        sh_coeffs[15] = 0.59004358992664352 * x * (-x2 + 3.0 * y2);        
    }
    return sh_coeffs;
}