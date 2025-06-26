#ifndef COLOR_H
#define COLOR_H

#include "rtweekend.h"
#include "vec3.h"
#include "utils/interval.h"

#include <vector>
#include <cstdint>

using color = vec3;

inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0) {
        return std::sqrt(linear_component);
    }
    return 0;
}

void write_color(std::vector<uint8_t> &image_data, int x, int y, int image_width, color pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Translate the [0,1] range to the [0,255] range.
    static const interval intensity(0.000, 0.999);
    uint8_t rbyte = uint8_t(256 * intensity.clamp(linear_to_gamma(r)));
    uint8_t gbyte = uint8_t(256 * intensity.clamp(linear_to_gamma(g)));
    uint8_t bbyte = uint8_t(256 * intensity.clamp(linear_to_gamma(b)));

    int index = (y * image_width + x) * 3;

    image_data[index] = rbyte;
    image_data[index + 1] = gbyte;
    image_data[index + 2] = bbyte;
}

#endif