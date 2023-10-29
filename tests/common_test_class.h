#ifndef COMMON_TEST_CLASS_H
#define COMMON_TEST_CLASS_H

#include <gtest/gtest.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <unordered_map>
#include <vector>

class CommonTestClass : public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;
    void InitOpenCL(size_t pid, size_t did, cl_command_queue_properties prop = 0);

protected:
    bool initialized {false};
    using PlatformDevMap = std::unordered_map<cl_platform_id, std::vector<cl_device_id>>;
    std::vector<cl_platform_id> platforms;
    PlatformDevMap platform2devices;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context ctx;
    cl_command_queue cq;
};

template<typename T>
void compareVecs(const std::vector<T>& a, const std::vector<T>& b)
{
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a[i], b[i]) << i;
    }
}

#endif  // COMMON_TEST_CLASS_H
