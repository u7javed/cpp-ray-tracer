#ifndef RTW_STB_IMAGE_H
#define RTW_STB_IMAGE_H

#ifdef _MSC_VER
    #pragma warning(push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include "stb_image.h"

#include <cstdlib>
#include <iostream>

class rtw_image {
    public:
        rtw_image() {}

        rtw_image(const char *image_filename) {
            std::string filename = std::string(image_filename);
            const char* imagedir = getenv("RTW_IMAGES");

            // Hunt for the image file in some likely locations.
            if (imagedir != nullptr && load(std::string(imagedir) + "/" + image_filename)) return;
            if (load(filename)) return;
            if (load("images/" + filename)) return;
            if (load("../images/" + filename)) return;
            if (load("../../images/" + filename)) return;
            if (load("../../../images/" + filename)) return;
            if (load("../../../../images/" + filename)) return;
            if (load("../../../../../images/" + filename)) return;
            if (load("../../../../../../images/" + filename)) return;

            std::cerr << "ERROR: Could not load image file " << image_filename << std::endl;
        }

        ~rtw_image() {
            delete[] bdata;
            STBI_FREE(fdata);
        }

        bool load(const std::string& filename) {
            int n = bytes_per_pixel;
            fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
            if (fdata == nullptr) return false;

            bytes_per_scanline = image_width * bytes_per_pixel;
            convert_to_bytes();
            return true;
        }

        int width() const { return (fdata == nullptr) ? 0 : image_width; }
        int height() const { return (fdata == nullptr) ? 0 : image_height; }

        const unsigned char* pixel_data(int x, int y) const {
            static unsigned char magenta[] = { 255, 0, 255 };
            if (bdata == nullptr) return magenta;

            x = clamp(x, 0, image_width);
            y = clamp(y, 0, image_height);
            return bdata + y * bytes_per_scanline + x * bytes_per_pixel;
        }

        const unsigned char*** all_pixel_data() const {
            static unsigned char magenta[] = { 255, 0, 255 };
            unsigned char ***data = new unsigned char**[image_height];
            for (int i = 0; i < image_height; i++) {
                data[i] = new unsigned char*[image_width];
            }

            for (int y = 0; y < image_height; y++) {
                for (int x = 0; x < image_width; x++) {
                    int xInt = clamp(x, 0, image_width);
                    int yInt = clamp(y, 0, image_height);
                    
                    if (bdata == nullptr) {
                        data[y][x] = magenta;
                    } else {
                        data[y][x] = bdata + y * bytes_per_scanline + x * bytes_per_pixel;
                    }
                }
            }
            return (const unsigned char***) data;
        }

        const unsigned char* flat_pixel_data() const { return bdata; }
    
    private:
        const int bytes_per_pixel = 3;
        float *fdata = nullptr;
        unsigned char *bdata = nullptr;
        int image_width = 0;
        int image_height = 0;
        int bytes_per_scanline = 0;

        static int clamp(int x, int low, int high) {
            if (x < low) return low;
            if (x < high) return x;
            return high - 1;
        }

        static unsigned char float_to_byte(float value) {
            if (value <= 0.0f) return 0;
            if (1.0f <= value) return 255;
            return (unsigned char)(256.0f * value);
        }

        void convert_to_bytes() {
            int total_bytes = image_width * image_height * bytes_per_pixel;
            bdata = new unsigned char[total_bytes];

            unsigned char *bptr = bdata;
            float *fptr = fdata;

            for (int i = 0; i < total_bytes; i++, fptr++, bptr++) {
                *bptr = float_to_byte(*fptr);
            }
        }
};

// Restore MSVC compiler warnings
#ifdef _MSC_VER
    #pragma warning (pop)
#endif

#endif