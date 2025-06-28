#ifndef SPHEREH
#define SPHEREH

#include <cmath>

#define M_PI 3.14159265358979323846f

#include "objects/hittable.h"
#include "primitives/ray.h"

__device__ void get_sphere_uv(const point3& p, float& u, float& v) {
    // p: a given point on the sphere of radius one, centered at the origin.
    // u: returned value [0,1] of angle around the Y axis from X=-1.
    // v: returned value [0,1] of angle from Y=-1 to Y=+1.
    //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
    //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
    //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

    float theta = acos(-p.y());
    float phi = atan2(-p.z(), p.x()) + M_PI;
    u = phi / (2 * M_PI);
    v = theta / M_PI;
}

class sphere: public hittable {
    public:
        ray center;
        float radius;
        material *mat_ptr;
        aabb bbox;

        __device__ sphere() {}
        __device__ sphere(point3 cen, float r, material *m) : center(cen, vec3(0, 0, 0)), radius(r), mat_ptr(m) {
            vec3 rvec(r, r, r);
            bbox = aabb(cen - rvec, cen + rvec);
        }
        __device__ sphere(point3 cen1, point3 cen2, float r, material *m) : center(cen1,  cen2 - cen1), radius(r), mat_ptr(m) {
            vec3 rvec(r, r, r);
            aabb box1(center.point_at_parameter(0) - rvec, center.point_at_parameter(0) + rvec);
            aabb box2(center.point_at_parameter(1) - rvec, center.point_at_parameter(1) + rvec);
            bbox = aabb(box1, box2);
        }
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual aabb bounding_box() const;
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
            get_sphere_uv(rec.normal, rec.u, rec.v);
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

__device__ aabb sphere::bounding_box() const {
    return bbox;
}

#endif