#ifndef CAMERA_H
#define CAMERA_H

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include "objects/hittable.h"

#include <vector>
#include <cstdint>

class camera {
    public:
        double aspect_ratio = 1.0;
        int image_width = 100;

        void render(const hittable& world, const std::string& output_path) {
            initialize();

            std::vector<uint8_t> image_data(image_width * image_height * 3);

            for (int y = 0; y < image_height; y++) {
                std::clog << "\rScanlines remaining: " << (image_height - y) << ' ' << std::flush;
                for (int x = 0; x < image_width; x++) {
                    auto pixel_center = pixel00_loc + (x * pixel_delta_u) + (y * pixel_delta_v);
                    auto ray_direction = pixel_center - center;
                    ray r(center, ray_direction);

                    color pixel_color = ray_color(r, world);
                    write_color(image_data, x, y, image_width, pixel_color);
                }
            }

            std::string executable_dir = output_path.substr(0, output_path.find_last_of('/'));
            std::string image_path = executable_dir + "\\..\\..\\..\\image.png";
            std::cout << "\nWriting image to " << image_path << std::endl;
            stbi_write_png(image_path.c_str(), image_width, image_height, 3, image_data.data(), image_width * 3);
        }

    private:

        int image_height;
        point3 center;
        point3 pixel00_loc;
        vec3 pixel_delta_u;
        vec3 pixel_delta_v;

        void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            center = point3(0, 0, 0);

            // Determine viewport dimensions.
            auto focal_length = 1.0;
            auto viewport_height = 2.0;
            auto viewport_width = viewport_height * (double(image_width)/image_height);

            // Calculate the vectors across the horizontal and down the vertical viewport edges.
            auto viewport_u = vec3(viewport_width, 0, 0);
            auto viewport_v = vec3(0, -viewport_height, 0);

            // Calculate the horizontal and vertical delta vectors from pixel to pixel.
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            // Calculate the location of the upper left pixel.
            auto viewport_upper_left =
                center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        }

        color ray_color(const ray& r, const hittable& world) {
            hit_record rec;

            if (world.hit(r, interval(0, infinity), rec)) {
                return 0.5 * (rec.normal + color(1, 1, 1));
            }

            vec3 unit_direction = unit_vector(r.direction());
            auto a = 0.5 * (unit_direction.y() + 1.0);
            return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
        }
};

#endif