#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include "types/vec3.h"
#include "types/color.h"
#include "primitives/ray.h"
#include "objects/hittable.h"
#include "objects/hittable_list.h"
#include "objects/sphere.h"
#include "primitives/camera.h"
#include "primitives/materials/material.h"

#include <iostream>
#include <time.h>
#include <curand_kernel.h>
#include <vector>
#include <string>

// Limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= max_x) || (y >= max_y)) return;
    int pixel_index = y * max_x + x;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(
    color *fb,
    int max_x,
    int max_y,
    int ns,
    camera **cam,
    hittable **world,
    curandState *rand_state
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= max_x) || (y >= max_y)) return;
    int pixel_index = y * max_x + x;
    curandState local_rand_state = rand_state[pixel_index];
    color col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(x + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(y + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += (*cam)->ray_color(r, world, &local_rand_state, 10); // hardcoded max depth
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_list[0] = new sphere(vec3(0,0,-1), 0.5,
                                new lambertian(vec3(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(vec3(0,-100.5,-1), 100,
                                new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1,0,-1), 0.5,
                                new metal(vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new sphere(vec3(-1,0,-1), 0.5,
                                 new dielectric(1.5));
        d_list[4] = new sphere(vec3(-1,0,-1), -0.45,
                                 new dielectric(1.5));
        *d_world  = new hittable_list(d_list,5);
        *d_camera = new camera();
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) {
    for (int i = 0; i < 5; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main(int argc, char *argv[]) {
    int nx = 1200;
    int ny = 600;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMallocManaged((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    // make our world of hitables
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 5*sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Time to render: " << timer_seconds << " seconds\n";

    // Output the image
    std::vector<uint8_t> image_data(nx * ny * 3);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int flipped_y = ny - y - 1;
            size_t pixel_index = flipped_y * nx + x;
            int ir = int(255.999 * fb[y * nx + x].r());
            int ig = int(255.999 * fb[y * nx + x].g());
            int ib = int(255.999 * fb[y * nx + x].b());

            image_data[3 * pixel_index + 0] = ir;
            image_data[3 * pixel_index + 1] = ig;
            image_data[3 * pixel_index + 2] = ib;
        }
    }

    std::string output_path = argv[0];
    std::string executable_dir = output_path.substr(0, output_path.find_last_of('/'));
    std::string image_path = executable_dir + "\\..\\..\\..\\image_cuda.png";
    std::cout << "\nWriting image to " << image_path << std::endl;
    stbi_write_png(image_path.c_str(), nx, ny, 3, image_data.data(), nx * 3);

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
    return 0;
}