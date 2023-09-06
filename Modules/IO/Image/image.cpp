// It's for stb image write
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.hpp"

#include <stb_image_write.h>

void Image::writeImgToFile(const std::string& file_name){
    std::vector<uint8_t> rgb_data(resolution.x() * resolution.y() * 3);
	for (int i = 0; i < data.size(); i++) {
		rgb_data[3 * i] = utils::trans(data[i].x());//utils::gammaCorrection(data[i].x());
		rgb_data[3 * i + 1] = utils::trans(data[i].y());//utils::gammaCorrection(data[i].y());
		rgb_data[3 * i + 2] = utils::trans(data[i].z());//utils::gammaCorrection(data[i].z());
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_png(file_name.c_str(), resolution.x(), resolution.y(), 3, rgb_data.data(), 0);
}

