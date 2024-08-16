#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include "core.hpp"
#include "utils.hpp"
#include <vector>
#include <string>

class Image{
public:
    using Color = Vec3f;
    
    Image() = delete;
    Image(int w, int h): resolution(w, h){
        data.resize(w * h);
    }
    [[nodiscard]] float getAspectRatio() const{
        return static_cast<float>(resolution.x()) / static_cast<float>(resolution.y());
    };
    [[nodiscard]] Vec2i getResolution() const{
        return resolution;
    };
    [[nodiscard]] Color getPixel(int x, int y) const{
        return data[x + resolution.x() * y];
    }
    void setPixel(int x, int y, const Vec3f& value){
        data[x + resolution.x() * y] = value;
    }
    void writeImgToFile(const std::string& file_name);
private:
    std::vector<Color> data;
    Vec2i resolution;
};

#endif // IMAGE_HPP_