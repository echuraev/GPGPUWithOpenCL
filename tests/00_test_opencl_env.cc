#include <gtest/gtest.h>

#include <OpenCL/OpenCL.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <vector>

TEST(OpenCL_00, basic_test)
{
    cl_uint ret_num_platforms;
    int err = clGetPlatformIDs(0, nullptr, &ret_num_platforms);
    ASSERT_EQ(err, CL_SUCCESS);
    std::vector<cl_platform_id> platforms(ret_num_platforms);
    err = clGetPlatformIDs(ret_num_platforms, platforms.data(), &ret_num_platforms);
    ASSERT_EQ(err, CL_SUCCESS);
    std::cout << "Available platforms: " << std::endl;
    for (int i = 0; i < ret_num_platforms; ++i) {
        size_t ret_size;
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &ret_size);
        ASSERT_EQ(err, CL_SUCCESS);
        std::string ret;
        ret.resize(ret_size);
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, ret_size, &ret[0], nullptr);
        ASSERT_EQ(err, CL_SUCCESS);
        std::cout << "\t" << i + 1 << ". " << ret << std::endl;

        cl_uint ret_num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &ret_num_devices);
        ASSERT_EQ(err, CL_SUCCESS);

        std::vector<cl_device_id> devices(ret_num_devices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, ret_num_devices, devices.data(), &ret_num_devices);
        ASSERT_EQ(err, CL_SUCCESS);
        std::cout << "\t   Devices:" << std::endl;
        for (int j = 0; j < ret_num_devices; ++j) {
            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, nullptr, &ret_size);
            ASSERT_EQ(err, CL_SUCCESS);
            ret.resize(ret_size);
            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, ret_size, &ret[0], nullptr);
            ASSERT_EQ(err, CL_SUCCESS);
            std::cout << "\t\t" << j + 1 << ". " << ret << std::endl;
        }
        for (size_t j = 0; j < devices.size(); ++j) {
            clReleaseDevice(devices[j]);
        }
    }
}
