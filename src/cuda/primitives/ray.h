#ifndef RAYH
#define RAYH

#include "types/vec3.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { orig = a; dir = b; tm = 0.0f; }
        __device__ ray(const vec3& a, const vec3& b, float t) { orig = a; dir = b; tm = t; }
        __device__ point3 origin() const { return orig; }
        __device__ vec3 direction() const { return dir; }
        __device__ float time() const { return tm; }
        __device__ point3 point_at_parameter(float t) const { return orig + t*dir; }

        point3 orig;
        vec3 dir;
        float tm;
};

#endif