#ifndef HITTABLEH
#define HITTABLEH

#include "primitives/ray.h"
#include "primitives/aabb.h"

class material;

struct hit_record {
    float t;
    float u;
    float v;
    point3 p;
    vec3 normal;
    material *mat_ptr;
};

class hittable {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        __device__ virtual aabb bounding_box() const = 0;
};

#endif