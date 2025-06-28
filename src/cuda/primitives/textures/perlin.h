#ifndef PERLINH
#define PERLINH

#include "types/vec3.h"

class perlin {
    public:
        __device__ perlin(curandState *local_rand_state) {
            for (int i = 0; i < point_count; i++) {
                randvec[i] = unit_vector(vec3(
                    curand_uniform(local_rand_state) * 2.0f - 1.0f,
                    curand_uniform(local_rand_state) * 2.0f - 1.0f,
                    curand_uniform(local_rand_state) * 2.0f - 1.0f
                ));
            }

            perlin_generate_perm(perm_x, local_rand_state);
            perlin_generate_perm(perm_y, local_rand_state);
            perlin_generate_perm(perm_z, local_rand_state);
        }

        __device__ float noise(const point3& p) const {
            float u = p.x() - std::floor(p.x());
            float v = p.y() - std::floor(p.y());
            float w = p.z() - std::floor(p.z());

            int i = int(std::floor(p.x()));
            int j = int(std::floor(p.y()));
            int k = int(std::floor(p.z()));

            vec3 c[2][2][2];
            for (int di = 0; di < 2; di++) {
                for (int dj = 0; dj < 2; dj++) {
                    for (int dk = 0; dk < 2; dk++) {
                        c[di][dj][dk] = randvec[
                            perm_x[(i + di) & 255] ^
                            perm_y[(j + dj) & 255] ^
                            perm_z[(k + dk) & 255]
                        ];
                    }
                }
            }
            return perlin_interp(c, u, v, w);
        }

        __device__ float turb(const point3& p, int depth) const {
            float accum = 0.0f;
            point3 temp_p = p;
            float weight = 1.0f;

            for (int i = 0; i < depth; i++) {
                accum += weight * noise(temp_p);
                weight *= 0.5f;
                temp_p *= 2.0f;
            }

            return std::fabs(accum);
        }
    
    private:
        static const int point_count = 256;
        vec3 randvec[point_count];
        int perm_x[point_count];
        int perm_y[point_count];
        int perm_z[point_count];

        __device__ static void perlin_generate_perm(int* p, curandState *local_rand_state) {
            for (int i = 0; i < point_count; i++) {
                p[i] = i;
            }

            permute(p, point_count, local_rand_state);
        }

        __device__ static void permute(int* p, int n, curandState *local_rand_state) {
            for (int i = n - 1; i > 0; i--) {
                int target = int(curand_uniform(local_rand_state) * (i + 1));
                int tmp = p[i];
                p[i] = p[target];
                p[target] = tmp;
            }
        }

        __device__ static float perlin_interp(const vec3 c[2][2][2], float u, float v, float w) {
            float uu = u * u * (3.0f - 2.0f * u);
            float vv = v * v * (3.0f - 2.0f * v);
            float ww = w * w * (3.0f - 2.0f * w);
            float accum = 0.0f;

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        vec3 weight_v(u - i, v - j, w - k);
                        accum += (i * uu + (1 - i) * (1 - uu))
                            * (j * vv + (1 - j) * (1 - vv))
                            * (k * ww + (1 - k) * (1 - ww))
                            * dot(c[i][j][k], weight_v);
                    }
                }
            }

            return accum;
        }
};

#endif