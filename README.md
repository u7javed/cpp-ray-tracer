# cpp-ray-tracer
A Ray Tracer Engine implemented in C++ and CUDA.

# Setup

This project uses CMake to compile the binaries for both CUDA and CPU ray tracing.

First build the project by running
```bat
.\build.bat
```

Then you can execute the ray tracer and render an image of the scene by running
```bat
.\run.bat
```

> By default, the `run.bat` runs on the cpu. To run it on the GPU, add the `--cuda` flag to the command as follows:

```bat
.\run.bat --gpu
```