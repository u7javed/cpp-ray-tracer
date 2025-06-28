#ifndef AABBH
#define AABBH

#include "primitives/ray.h"

class aabb {
    public:
        float x_min, x_max, y_min, y_max, z_min, z_max;

        __device__ aabb() {}
        __device__ aabb(float x0, float x1, float y0, float y1, float z0, float z1) : x_min(x0), x_max(x1), y_min(y0), y_max(y1), z_min(z0), z_max(z1) {}
        __device__ aabb(const point3& a, const point3& b) {
            if (a.x() < b.x()) {
                x_min = a.x();
                x_max = b.x();
            } else {
                x_min = b.x();
                x_max = a.x();
            }

            if (a.y() < b.y()) {
                y_min = a.y();
                y_max = b.y();
            } else {
                y_min = b.y();
                y_max = a.y();
            }

            if (a.z() < b.z()) {
                z_min = a.z();
                z_max = b.z();
            } else {
                z_min = b.z();
                z_max = a.z();
            }
        }
        __device__ aabb(const aabb& box0, const aabb& box1) {
            x_min = box0.x_min < box1.x_min ? box0.x_min : box1.x_min;
            x_max = box0.x_max > box1.x_max ? box0.x_max : box1.x_max;
            y_min = box0.y_min < box1.y_min ? box0.y_min : box1.y_min;
            y_max = box0.y_max > box1.y_max ? box0.y_max : box1.y_max;
            z_min = box0.z_min < box1.z_min ? box0.z_min : box1.z_min;
            z_max = box0.z_max > box1.z_max ? box0.z_max : box1.z_max;
        }

        __device__ float axis_min(int n) const {
            return (n == 0) ? x_min : (n == 1) ? y_min : z_min;
        }

        __device__ float axis_max(int n) const {
            return (n == 0) ? x_max : (n == 1) ? y_max : z_max;
        }

        __device__ bool hit(const ray& r, float t_min, float t_max) const {
            point3 ray_origin = r.origin();
            vec3 ray_dir = r.direction();

            for(int axis = 0; axis < 3; axis++) {
                float axis_min = this->axis_min(axis);
                float axis_max = this->axis_max(axis);
                float adinv = 1.0 / ray_dir[axis];

                float t0 = (axis_min - ray_origin.e[axis]) * adinv;
                float t1 = (axis_max - ray_origin[axis]) * adinv;

                if (t0 < t1) {
                    if (t0 > t_min) t_min = t0;
                    if (t1 < t_max) t_max = t1;
                } else {
                    if (t1 > t_min) t_min = t1;
                    if (t0 < t_max) t_max = t0;
                }

                if (t_max <= t_min) return false; // No intersection
            }
            return true;
        }
};

#endif