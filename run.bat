@echo off

REM Check if --cuda flag is provided
if "%1"=="--gpu" (
    echo Running CUDA version...
    .\build\bin\Debug\cpp-ray-tracer-cuda.exe
) else (
    echo Running CPU version...
    .\build\bin\Debug\cpp-ray-tracer-cpu.exe
)
