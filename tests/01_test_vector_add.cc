#include "common_functions.h"
#include "common_test_class.h"

namespace
{
template<typename T>
std::vector<T> refImpl(const std::vector<T>& a, const std::vector<T>& b)
{
    std::vector<T> res(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        res[i] = a[i] + b[i];
    }
    return res;
}
}  // namespace

using TestParams = std::tuple<std::string,          // kernel_name
                              size_t,               // platform_id
                              size_t,               // device_id
                              std::vector<size_t>,  // gws
                              std::vector<size_t>,  // lws
                              size_t>;              // vector_size

class VectorAddTest_01
    : public testing::WithParamInterface<TestParams>
    , public CommonTestClass
{
protected:
    void SetUp() final
    {
        CommonTestClass::SetUp();

        kernel_src = readKernel("01_vector_add.cl");

        size_t pid, did;
        std::tie(kernel_name, pid, did, gws, lws, vec_size) = GetParam();

        InitOpenCL(pid, did, CL_QUEUE_PROFILING_ENABLE);
    }

protected:
    std::string kernel_src;
    const size_t repeats {1000};
    std::string kernel_name;
    std::vector<size_t> gws;
    std::vector<size_t> lws;
    size_t vec_size;
    int err;
};

TEST_P(VectorAddTest_01, basic_impl)
{
    // Create vectors
    auto a_vec = generateRandomVec<int>(-100, 100, vec_size);
    auto b_vec = generateRandomVec<int>(-100, 100, vec_size);
    auto ref_out = refImpl(a_vec, b_vec);
    std::vector<int> c_vec(vec_size);

    // Create buffers
    cl_mem a_mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a_vec.size() * sizeof(int),
                                  a_vec.data(), &err);
    ASSERT_EQ(err, CL_SUCCESS);
    cl_mem b_mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b_vec.size() * sizeof(int),
                                  b_vec.data(), &err);
    ASSERT_EQ(err, CL_SUCCESS);
    cl_mem c_mem = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, c_vec.size() * sizeof(int), nullptr, &err);
    ASSERT_EQ(err, CL_SUCCESS);

    const char* str = kernel_src.c_str();
    // Create program
    cl_program program = clCreateProgramWithSource(ctx, 1, &str, nullptr, &err);
    ASSERT_EQ(err, CL_SUCCESS);

    // Build program
    err = clBuildProgramWrapper(program, 1, &device_id);
    ASSERT_EQ(err, CL_SUCCESS);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    ASSERT_EQ(err, CL_SUCCESS);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem);
    ASSERT_EQ(err, CL_SUCCESS);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem);
    ASSERT_EQ(err, CL_SUCCESS);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem);
    ASSERT_EQ(err, CL_SUCCESS);
    if (kernel_name == "vector_add_return") {
        cl_uint size = vec_size;
        err = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void*)&size);
        ASSERT_EQ(err, CL_SUCCESS);
    }

    double cpu_time = 0;
    double gpu_time = 0;
    for (size_t i = 0; i < repeats; ++i) {
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cl_event event;
        err = clEnqueueNDRangeKernel(cq, kernel, 3, nullptr, gws.data(), lws.data(), 0, nullptr, &event);
        ASSERT_EQ(err, CL_SUCCESS);
        err = clWaitForEvents(1, &event);
        ASSERT_EQ(err, CL_SUCCESS);
        err = clFinish(cq);
        ASSERT_EQ(err, CL_SUCCESS);
        auto cpuEnd = std::chrono::high_resolution_clock::now();

        // Measure execution time
        cl_ulong time_start;
        cl_ulong time_end;

        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);

        gpu_time += (time_end - time_start) * 1e-6;  // from ns to ms
        cpu_time += std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart).count() * 1e-6;
        err = clReleaseEvent(event);
        ASSERT_EQ(err, CL_SUCCESS);
    }
    cpu_time /= repeats;
    gpu_time /= repeats;
    std::cout << "Execution time, cpu: " << cpu_time << " ms., gpu: " << gpu_time << "ms." << std::endl;

    // Read buffer with result of calculation
    err = clEnqueueReadBuffer(cq, c_mem, CL_TRUE, 0, vec_size * sizeof(cl_int), c_vec.data(), 0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS);
    compareVecs(c_vec, ref_out);

    err = clReleaseMemObject(a_mem);
    ASSERT_EQ(err, CL_SUCCESS);
    err = clReleaseMemObject(b_mem);
    ASSERT_EQ(err, CL_SUCCESS);
    err = clReleaseMemObject(c_mem);
    ASSERT_EQ(err, CL_SUCCESS);
    err = clReleaseKernel(kernel);
    ASSERT_EQ(err, CL_SUCCESS);
    err = clReleaseProgram(program);
    ASSERT_EQ(err, CL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
        OpenCL_01,
        VectorAddTest_01,
        // TODO: Probably more and auto parameterized tests
        ::testing::Values(TestParams("vector_add", 0, 2, {1000000, 1, 1}, {1, 1, 1}, 1000000),
                          TestParams("vector_add_return", 0, 2, {1000000, 1, 1}, {1, 1, 1}, 1000000),
                          TestParams("vector_add", 0, 2, {1000064, 1, 1}, {128, 1, 1}, 1000000),
                          TestParams("vector_add_return", 0, 2, {1000064, 1, 1}, {128, 1, 1}, 1000000)));
