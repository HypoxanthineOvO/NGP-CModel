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
