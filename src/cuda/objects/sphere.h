#ifndef SPHEREH
#define SPHEREH

#include "objects/hittable.h"
#include "primitives/ray.h"

class sphere: public hittable {
    public:
        ray center;
        float radius;
        material *mat_ptr;

        __device__ sphere() {}
        __device__ sphere(point3 cen, float r, material *m) : center(cen, vec3(0, 0, 0)), radius(r), mat_ptr(m) {}
        __device__ sphere(point3 cen1, point3 cen2, float r, material *m) : center(cen1,  cen2 - cen1), radius(r), mat_ptr(m) {}
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    point3 current_center = center.point_at_parameter(r.time());
    vec3 oc = r.origin() - current_center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - current_center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - current_center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

#endif