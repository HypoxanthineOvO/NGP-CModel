#ifndef UTILS_HPP_
#define UTILS_HPP_

#include "core.hpp"

#include <random>

namespace utils {

	static inline float clamp01(float v) {
		if (v > 1) v = 1;
		else if (v < 0) v = 0;
		return v;
	}

	static inline uint8_t gammaCorrection(float radiance) {
		return static_cast<uint8_t>(255.f * clamp01(powf(radiance, 1.f / 2.2f)));
	}

	static inline uint8_t trans(float radiance){
		return static_cast<uint8_t>(255.f * clamp01(radiance));
	}

	static inline float radians(float x) { return x * PI / 180; }

	static inline Vec3f deNan(const Vec3f& vec, float val) {
		Vec3f tmp = vec;
		if (vec.x() != vec.x()) tmp.x() = val;
		if (vec.y() != vec.y()) tmp.y() = val;
		if (vec.z() != vec.z()) tmp.z() = val;
		return tmp;
	}
	static VecXf sigmoid(VecXf input){
		VecXf output(input.size());
		for(int i = 0; i < input.size(); i++){
			output(i) = 1.0f / (1.0f + std::exp(-input(i)));
		}
		return output;
	}

	static inline uint32_t as_uint(const float x){
		return *(uint32_t*)(&x);
	}

	static inline float as_float(const uint32_t x) {
		return *(float*)&x;
	}

	static float from_int_to_float16(const uint32_t& x){
		const uint32_t exponent = (x & 0x7C00) >> 10,
			mantissa = (x & 0x03FF) << 13,
			v = as_uint((float)(mantissa)) >> 23;
			return as_float(
				(x&0x8000)<<16 | 
				(exponent != 0)*((exponent + 112) << 23 | mantissa) |
				((exponent == 0)&(mantissa != 0)) * ((v - 37) << 23|((mantissa << (150-v)) & 0x007FE000)));
	}

	static inline int inv_Part_1_By_2(int x){
			x = ((x >> 2) | x) & 0x030C30C3;
			x = ((x >> 4) | x) & 0x0300F00F;
			x = ((x >> 8) | x) & 0x030000FF;
			x = ((x >>16) | x) & 0x000003FF;
			return x;
	}


	static int inv_morton(int input, int resolution){
		int x = inv_Part_1_By_2(input &        0x09249249);
		int y = inv_Part_1_By_2((input >> 1) & 0x09249249);
		int z = inv_Part_1_By_2((input >> 2) & 0x09249249);
		
		return x * resolution * resolution + y * resolution + z;
	}

	static Mat4f nerf_matrix_to_ngp(MatXf pose, float scale = 0.33, Vec3f offset = Vec3f(0.5, 0.5, 0.5)){
		Mat4f out_mat;
		out_mat << pose(1, 0) , -pose(1, 1) , -pose(1, 2) , pose(1, 3) * scale + offset(0) , \
			pose(2, 0) , -pose(2, 1) , -pose(2, 2) , pose(2, 3) * scale + offset(1) , \
			pose(0, 0) , -pose(0, 1) , -pose(0, 2) , pose(0, 3) * scale + offset(2) ,\
			0 , 0 , 0 , 1;
		return out_mat;
	}
	static int get_int_from_json(const nlohmann::json& config, std::string name){
		int value;
		return config.at(name).get_to(value);
	}

	static inline float min(float v1, float v2){
		if (v1 < v2) return v1;
		else return v2;
	}
	static inline float max(float v1, float v2){
		if (v1 > v2) return v1;
		else return v2;
	}
}

class Sampler {
public:
	Sampler() = default;
	float get1D() { return dis(engine); }
	Vec2f get2D() { return { dis(engine), dis(engine) }; }
	void setSeed(int i) { engine.seed(i); }
private:
	std::default_random_engine engine;
	std::uniform_real_distribution<float> dis;
};

#endif //UTILS_HPP_
