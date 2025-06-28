#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include "types/vec3.h"
#include "types/color.h"
#include "primitives/ray.h"
#include "objects/hittable.h"
#include "objects/hittable_list.h"
#include "objects/sphere.h"
#include "objects/bvh.h"
#include "primitives/camera.h"
#include "primitives/materials/material.h"
#include "primitives/textures/texture.h"
#include "utils/rtw_stb_image.h"

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

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= max_x) || (y >= max_y)) return;
    int pixel_index = y * max_x + x;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
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
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += (*cam)->ray_color(r, world, &local_rand_state, 10); // hardcoded max depth
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_noise_sphere(
    hittable **d_list,
    hittable **d_world,
    camera **d_camera,
    int nx,
    int ny,
    curandState *rand_state
) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        curandState local_rand_state = *rand_state;
        texture *pertext = new noise_texture(4.0f, &local_rand_state);

        d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(pertext));
        d_list[1] = new sphere(vec3(0, 2, 0),  2, new lambertian(pertext));
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 2);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(
            lookfrom,
            lookat,
            vec3(0, 1, 0),
            20.0,
            float(nx)/float(ny),
            aperture,
            dist_to_focus
        );
    }
}

__global__ void create_earth(
    hittable **d_list,
    hittable **d_world,
    camera **d_camera,
    const unsigned char *earth_data,
    int earth_width,
    int earth_height,
    int nx,
    int ny,
    curandState *rand_state
) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        curandState local_rand_state = *rand_state;

        texture *earth_texture = new image_texture(earth_data, earth_width, earth_height);
        d_list[0] = new sphere(vec3(0, 0, 0), 2.0f, new lambertian(earth_texture));
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 1);

        vec3 lookfrom(0, 0, 12);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(
            lookfrom,
            lookat,
            vec3(0, 1, 0),
            20.0,
            float(nx)/float(ny),
            aperture,
            dist_to_focus
        );
    }
}

__global__ void create_checkered_spheres(
    hittable **d_list,
    hittable **d_world,
    camera **d_camera,
    int nx,
    int ny,
    curandState *rand_state
) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        curandState local_rand_state = *rand_state;
        texture *checker = new checker_texture(0.32f, color(.2, .3, .1), color(.9, .9, .9));
        d_list[0] = new sphere(vec3(0, -10, 0), 10.0f, new lambertian(checker));
        d_list[1] = new sphere(vec3(0, 10, 0),  10.0f, new lambertian(checker));
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 2);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(
            lookfrom,
            lookat,
            vec3(0, 1, 0),
            20.0,
            float(nx)/float(ny),
            aperture,
            dist_to_focus
        );
    }
}

__global__ void create_bouncing_spheres(
    hittable **d_list,
    hittable **d_world,
    camera **d_camera,
    int nx,
    int ny,
    curandState *rand_state
) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        curandState local_rand_state = *rand_state;
        texture *checker = new checker_texture(0.32f, color(.2, .3, .1), color(.9, .9, .9));
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(checker));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                vec3 center2 = center + vec3(0, RND*0.5f, 0);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, center2, 0.2f,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, center2, 0.2f,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, center2, 0.2f, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(
            lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx)/float(ny),
            aperture,
            dist_to_focus
        );
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera, int num_objects) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Delete the 2 spheres we created
        for(int i = 0; i < num_objects; i++) {
            sphere *s = (sphere *)d_list[i];
            delete s->mat_ptr;  // Delete the material
            delete s;           // Delete the sphere
        }
        
        // Delete the hittable_list and its contents
        hittable_list *world = (hittable_list *)*d_world;
        if (world->single_object != nullptr) {
            // Delete the BVH node (this will recursively delete its children)
            delete world->single_object;
        }
        delete world;
        
        // Delete the camera
        delete *d_camera;
    }
}

int main(int argc, char *argv[]) {
    int nx = 712;
    int ny = 400;
    int ns = 50;
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
    curandState *d_rand_state2;
    checkCudaErrors(cudaMallocManaged((void **)&d_rand_state2, 1*sizeof(curandState)));

    // We need that 2nd random state to be initialized for the world creation
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int cs = 3;

    // make our world of hitables
    hittable **d_list;
    int num_objects;
    switch (cs) {
        case 1:
            num_objects = 22*22+1+3;
            break;
        case 2:
            num_objects = 2;
            break;
        case 3:
            num_objects = 2;
            break;
        case 4:
            num_objects = 1;
            break;
    }
    checkCudaErrors(cudaMalloc((void **)&d_list, num_objects*sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    
    // GPU memory for earth image (only used in case 3)
    unsigned char *d_pixels = nullptr;
    
    switch (cs) {
        case 1:
            create_bouncing_spheres<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
            break;
        case 2:
            create_checkered_spheres<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
            break;
        case 3:
            create_noise_sphere<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
            break;
        case 4:
            rtw_image earth_image("earthmap.jpg");
            const unsigned char* earth_data = earth_image.flat_pixel_data();
            
            // Allocate GPU memory for the image data
            int earth_width = earth_image.width();
            int earth_height = earth_image.height();
            
            // Allocate the actual pixel data as a flat array
            checkCudaErrors(cudaMalloc((void **)&d_pixels, earth_width * earth_height * 3));
            
            // Copy pixel data from CPU to GPU
            checkCudaErrors(cudaMemcpy(d_pixels, earth_data, earth_width * earth_height * 3, cudaMemcpyHostToDevice));
            
            create_earth<<<1, 1>>>(d_list, d_world, d_camera, d_pixels, earth_width, earth_height, nx, ny, d_rand_state2);
            
            // Note: GPU memory will be cleaned up later
            break;
    }
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
    free_world<<<1,1>>>(d_list, d_world, d_camera, num_objects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));
    
    // Clean up earth image GPU memory if allocated
    if (d_pixels != nullptr) {
        checkCudaErrors(cudaFree(d_pixels));
    }

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
    return 0;
}