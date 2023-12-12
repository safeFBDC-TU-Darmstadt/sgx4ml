//
// Created by adrian on 10.06.22.
//

#include <numeric>
#include <cstring>
#include "dnnl_utils.hpp"

void initMemoryWithRandomValues(dnnl::memory & m, std::function<float()> generator) {
    auto md = m.get_desc();
    auto dims = md.data.dims;
    auto num_dims = md.data.ndims;
    std::vector<float> data( std::accumulate(dims, dims + num_dims, 1, std::multiplies<>()));
    generate(begin(data), end(data), generator);
    write_to_dnnl_memory(data.data(), m);
}

void write_to_dnnl_memory(const void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        std::memcpy(mem.get_data_handle(), handle, bytes);
        //uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        //for (size_t i = 0; i < bytes; ++i)
        //    dst[i] = ((uint8_t *)handle)[i];
    }
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    else if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();
        size_t bytes = mem.get_desc().get_size();

        cl_int ret = clEnqueueWriteBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error(
                    "clEnqueueWriteBuffer failed. Status Code: "
                    + std::to_string(ret) + "\n");
    }
#endif
}

void reset_dnnl_memory(void *handle, dnnl::memory &mem) {
    mem.set_data_handle(handle);
}

// Read from memory, write to handle
void read_from_dnnl_memory(void *handle, const dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        std::memcpy(handle, mem.get_data_handle(), bytes);
        //uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        //for (size_t i = 0; i < bytes; ++i)
        //    ((uint8_t *)handle)[i] = src[i];
    }
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    else if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();

        cl_int ret = clEnqueueReadBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error("clEnqueueReadBuffer failed. Status Code: "
                                     + std::to_string(ret) + "\n");
    }
#endif
}


