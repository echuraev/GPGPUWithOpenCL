#include "common_test_class.h"

#include <iostream>

void CommonTestClass::SetUp()
{
    initialized = false;
    cl_uint ret_num_platforms;
    int err = clGetPlatformIDs(0, nullptr, &ret_num_platforms);
    ASSERT_EQ(err, CL_SUCCESS);
    platforms.resize(ret_num_platforms);
    err = clGetPlatformIDs(ret_num_platforms, platforms.data(), &ret_num_platforms);
    ASSERT_EQ(err, CL_SUCCESS);

    for (auto& plat : platforms) {
        cl_uint ret_num_devices;
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &ret_num_devices);
        ASSERT_EQ(err, CL_SUCCESS);

        platform2devices[plat] = {};
        platform2devices[plat].resize(ret_num_devices);
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, ret_num_devices, platform2devices[plat].data(),
                             &ret_num_devices);
        ASSERT_EQ(err, CL_SUCCESS);
    }
}

void CommonTestClass::TearDown()
{
    if (initialized) {
        clReleaseCommandQueue(cq);
        clReleaseContext(ctx);
    }
    for (auto& plat : platforms) {
        for (auto& dev : platform2devices[plat]) {
            clReleaseDevice(dev);
        }
    }
    platform2devices.clear();
    platforms.clear();
}

void CommonTestClass::InitOpenCL(size_t pid, size_t did, cl_command_queue_properties prop)
{
    if (pid >= platforms.size()) {
        std::cerr << "Warning! Cannot find platform " << pid << ". The first platform will be used." << std::endl;
        pid = 0;
    }
    platform_id = platforms[pid];
    if (did >= platform2devices[platform_id].size()) {
        std::cerr << "Warning! Cannot find device " << did << ". The first device will be used." << std::endl;
        did = 0;
    }
    device_id = platform2devices[platform_id][did];

    // Create context
    int err;
    ctx = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    cq = clCreateCommandQueue(ctx, device_id, prop, &err);
    ASSERT_EQ(err, CL_SUCCESS);
}
