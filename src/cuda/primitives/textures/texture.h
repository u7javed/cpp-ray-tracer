#ifndef TEXTUREH
#define TEXTUREH

#include "types/vec3.h"
#include "types/color.h"
#include "primitives/textures/perlin.h"

class texture {
    public:
        __device__ virtual color value(float u, float v, const point3& p) const = 0;
};

class solid_color : public texture {
    public:
        color albedo;

        __device__ solid_color(const color& albedo) : albedo(albedo) {}

        __device__ solid_color(float red, float green, float blue) : solid_color(color(red, green, blue)) {}

        __device__ virtual color value(float u, float v, const point3& p) const override {
            return albedo;
        }
};

class checker_texture : public texture {
    public:
        float inv_scale;
        texture *even;
        texture *odd;

        __device__ checker_texture(float scale, texture *even, texture *odd) : inv_scale(1.0f / scale), even(even), odd(odd) {}

        __device__ checker_texture(float scale, const color& c1, const color& c2) : checker_texture(scale, new solid_color(c1), new solid_color(c2)) {}

        __device__ virtual color value(float u, float v, const point3& p) const override {
            int x = int(std::floor(inv_scale * p.x()));
            int y = int(std::floor(inv_scale * p.y()));
            int z = int(std::floor(inv_scale * p.z()));

            bool isEven = (x + y + z) % 2 == 0;
            return isEven ? even->value(u, v, p) : odd->value(u, v, p);
        }
};

class image_texture : public texture {
    public:
        const unsigned char *data;
        int width, height;

        __device__ image_texture(const unsigned char *data, int width, int height) : data(data), width(width), height(height) {}

        __device__ virtual color value(float u, float v, const point3& p) const override {
            if (width <= 0) return color(0, 1, 1);

            u = u < 0.0f ? 0.0f : u > 1.0f ? 1.0f : u;
            v = 1.0f - (v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v);

            int x = int(u * width);
            int y = int(v * height);
            x = x < 0 ? 0 : (x >= width ? width - 1 : x);
            y = y < 0 ? 0 : (y >= height ? height - 1 : y);
            int pixel_index = (y * width + x) * 3;
            const unsigned char *pixel = data + pixel_index;

            float color_scale = 1.0f / 255.0f;
            return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
        }

};

class noise_texture : public texture {
    public:
        perlin noise;
        float scale;

        __device__ noise_texture(float scale, curandState *local_rand_state) : scale(scale), noise(local_rand_state) {}

        __device__ virtual color value(float u, float v, const point3& p) const override {
            return color(.5f, .5f, .5f) * (1.0f + std::sin(scale * p.z() + 10.0f * noise.turb(p, 7)));
        }
};

#endif