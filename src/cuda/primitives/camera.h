#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>

#include "primitives/ray.h"
#include "objects/hittable.h"
#include "types/color.h"
#include "primitives/materials/material.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))
#define M_PI 3.14159265358979323846f

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0.0f) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class camera {
    public:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        float lens_radius;

        __device__ camera(
            vec3 lookfrom,
            vec3 lookat,
            vec3 vup,
            float vfov,
            float aspect_ratio,
            float aperture,
            float focus_dist
        ) {
            lens_radius = aperture / 2.0f;
            float theta = vfov * M_PI / 180.0f;
            float half_height = tan(theta / 2.0f);
            float half_width = aspect_ratio * half_height;
            origin = lookfrom;
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);
            lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
            horizontal = 2.0f * half_width * focus_dist * u;
            vertical = 2.0f * half_height * focus_dist * v;
        }

        __device__ ray get_ray(float s, float t, curandState *local_rand_state) {
            vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
            vec3 offset = u * rd.x() + v * rd.y();
            return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
        }

        __device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
            vec3 p;
            do {
                p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
            } while (p.squared_length() >= 1.0f);
            return p;
        }

        __device__ color ray_color(const ray& r, hittable **world, curandState *local_rand_state, int max_depth) {
            ray cur_ray = r;
            color cur_attenuation = color(1.0, 1.0, 1.0);
            for (int depth = 0; depth < max_depth; depth++) {
                hit_record rec;
                if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
                    ray scattered;
                    color attenuation;
                    if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                        cur_attenuation *= attenuation;
                        cur_ray = scattered;
                    } else {
                        return color(0.0, 0.0, 0.0);
                    }
                } else {
                    vec3 unit_direction = unit_vector(cur_ray.direction());
                    float t = 0.5f * (unit_direction.y() + 1.0f);
                    color c = (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
                    return cur_attenuation * c;
                }
            }
            return color(0.0, 0.0, 0.0); // exceeded max depth
        }
};

#endif