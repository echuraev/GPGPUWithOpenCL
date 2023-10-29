#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include <random>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

std::string readKernel(const std::string& name);
int clBuildProgramWrapper(cl_program program,
                          cl_uint num_devices,
                          const cl_device_id* device_list,
                          const char* options = nullptr);

template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
std::vector<T> generateRandomVec(T min, T max, size_t size)
{
    std::vector<T> res(size);
    std::random_device rd;
    std::mt19937 engine(rd());

    using distribution_t = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>,
                                              std::uniform_real_distribution<T>>;
    distribution_t dist(min, max);
    for (size_t i = 0; i < size; ++i) {
        res[i] = dist(engine);
    }
    return res;
}

#endif  // COMMON_FUNCTIONS_H

