#ifndef CAMERA_H
#define CAMERA_H

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include "objects/hittable.h"
#include "primitives/materials/material.h"

#include <vector>
#include <cstdint>

class camera {
    public:
        double aspect_ratio = 1.0;
        int image_width = 100;
        int samples_per_pixel = 10; // Number of samples per pixel
        int max_depth = 5;

        double vfov = 90; // Vertical field of view in degrees
        point3 lookfrom = point3(0, 0, 0);
        point3 lookat = point3(0, 0, -1);
        vec3 vup = vec3(0, 1, 0);

        double defocus_angle = 0;
        double focus_dist = 10;

        void render(const hittable& world, const std::string& output_path) {
            initialize();

            std::vector<uint8_t> image_data(image_width * image_height * 3);

            for (int y = 0; y < image_height; y++) {
                std::clog << "\rScanlines remaining: " << (image_height - y) << ' ' << std::flush;
                for (int x = 0; x < image_width; x++) {
                    color pixel_color(0, 0, 0);
                    for (int sample = 0; sample < samples_per_pixel; ++sample) {
                        ray r = get_ray(x, y);
                        pixel_color += ray_color(r, world, max_depth);
                    }
                    write_color(image_data, x, y, image_width, pixel_samples_scale * pixel_color);
                }
            }

            std::string executable_dir = output_path.substr(0, output_path.find_last_of('/'));
            std::string image_path = executable_dir + "\\..\\..\\..\\image.png";
            std::cout << "\nWriting image to " << image_path << std::endl;
            stbi_write_png(image_path.c_str(), image_width, image_height, 3, image_data.data(), image_width * 3);
        }

    private:
        int image_height;
        double pixel_samples_scale;
        point3 center;
        point3 pixel00_loc;
        vec3 pixel_delta_u;
        vec3 pixel_delta_v;
        vec3 u, v, w;
        vec3 defocus_disk_u;
        vec3 defocus_disk_v;

        void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0 / samples_per_pixel;

            center = lookfrom;

            // Determine viewport dimensions.
            auto theta = degrees_to_radians(vfov);
            auto h = std::tan(theta/2);
            auto viewport_height = 2 * h * focus_dist;
            auto viewport_width = viewport_height * (double(image_width)/image_height);

            // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            // Calculate the vectors across the horizontal and down the vertical viewport edges.
            vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
            vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

            // Calculate the horizontal and vertical delta vectors from pixel to pixel.
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            // Calculate the location of the upper left pixel.
            auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

            // Calculate the camera defocus disk basis vectors.
            auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }

        ray get_ray(int x, int y) {
            // Construct a camera ray originating from the origin and directed at randomly
            // sampled point around the pixel location x, y

            auto offset = sample_square();
            auto pixel_sample = pixel00_loc + ((x + offset.x()) * pixel_delta_u) + ((y + offset.y()) * pixel_delta_v);
            auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
            auto ray_direction = pixel_sample - ray_origin;
            return ray(ray_origin, ray_direction);
        }

        vec3 sample_square() const {
            // Returns the vector at a random point in the [-.5, -.5] - [.5, .5] unit square.
            return vec3(random_double() - 0.5, random_double() - 0.5, 0);
        }

        point3 defocus_disk_sample() const {
            auto p = random_in_unit_disk();
            return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
        }

        color ray_color(const ray& r, const hittable& world, int depth) {
            // If we've exceeded the ray bounce limit, no more light is gathered.
            if (depth <= 0) {
                return color(0, 0, 0);
            }

            hit_record rec;

            if (world.hit(r, interval(0.01, infinity), rec)) {
                ray scattered;
                color attenuation;
                if (rec.mat->scatter(r, rec, attenuation, scattered)) {
                    return attenuation * ray_color(scattered, world, depth - 1);
                }
                return color(0, 0, 0);
            }

            vec3 unit_direction = unit_vector(r.direction());
            auto a = 0.5 * (unit_direction.y() + 1.0);
            return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
        }
};

#endif