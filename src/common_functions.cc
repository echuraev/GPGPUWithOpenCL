#include "common_functions.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

std::string readKernel(const std::string& name)
{
    namespace fs = std::filesystem;
    fs::path kernel_path = fs::path("..") / "kernels" / name;
    std::ifstream file(kernel_path.string());
    if (!file.is_open()) {
        std::cerr << "Error! Cannot read kernel: " << name << std::endl;
        return "";
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

int clBuildProgramWrapper(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options)
{
    int err = clBuildProgram(program, num_devices, device_list, options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t len;
        std::string log;
        clGetProgramBuildInfo(program, *device_list, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        log.resize(len);
        clGetProgramBuildInfo(program, *device_list, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
        std::cerr << "clBuildProgramWrapper, error: " << log << std::endl;
    }
    return err;
}
