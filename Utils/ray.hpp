#ifndef RAY_HPP_
#define RAY_HPP_

#include "core.hpp"

class Ray{
public:
    using Point = Vec3f;
    using Direction = Vec3f;
    explicit Ray(const Point& origin, const Direction& direction, float t_min = RAY_DEFAULT_MIN, float t_max = RAY_DEFAULT_MIN):
        origin(origin), direction(direction), t_min(t_min), t_max(t_max){
            normalize();
        };

    void setOrigin(const Point& origin){
        this->origin = origin;
    }
    Point getOrigin() const{
        return origin;
    }
    void setDirection(const Direction& direction){
        this->direction = direction;
    }
    Direction getDirection() const{
        return direction;
    }

    float getTMin() const{
        return t_min;
    }
    float getTMax() const{
        return t_max;
    }

    Point at(float t) const{
        return origin + t * direction;
    }
    Point operator()(float t) const{
        return at(t);
    }
    void normalize(){
        direction = direction / direction.norm();
    }

private:
    Point origin;
    Direction direction;
    float t_min, t_max;
};

#endif // RAY_HPP_