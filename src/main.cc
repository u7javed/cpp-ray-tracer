#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cstdint>


int main(int argc, char** argv) {
    
    // Image

    int image_width = 256;
    int image_height = 256;

    // Render

    std::vector<uint8_t> image_data(image_width * image_height * 3);

    for (int y = 0; y < image_height; y++) {
        std::clog << "\rScanlines remaining: " << (image_height - y) << ' ' << std::flush;
        for (int x = 0; x < image_width; x++) {

            int index = (y * image_width + x) * 3;

            // Pixel color
            auto r = double(x) / (image_width-1);
            auto g = double(y) / (image_height-1);
            auto b = 0.0;

            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);

            image_data[index] = ir;
            image_data[index + 1] = ig;
            image_data[index + 2] = ib;

        }
    }

    // Write to PNG file
    // Get path of executable and place the image in the build folder
    std::string executable_path = argv[0];
    std::string executable_dir = executable_path.substr(0, executable_path.find_last_of('/'));
    std::string image_path = executable_dir + "\\..\\..\\..\\image.png";

    std::cout << "\nWriting image to " << image_path << std::endl;

    stbi_write_png(image_path.c_str(), image_width, image_height, 3, image_data.data(), image_width * 3);

    return 0;
}