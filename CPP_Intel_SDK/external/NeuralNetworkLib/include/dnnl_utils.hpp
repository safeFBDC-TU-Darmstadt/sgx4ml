//
// Created by adrian on 08.06.22.
//

#ifndef DNNLFASHIONMLP_DNNL_UTILS_HPP
#define DNNLFASHIONMLP_DNNL_UTILS_HPP

#include <dnnl.hpp>
#include <functional>

void initMemoryWithRandomValues(dnnl::memory &, std::function<float()>);

// Read from handle, write to memory
void write_to_dnnl_memory(const void *, dnnl::memory &);

void reset_dnnl_memory(void *handle, dnnl::memory &mem);

// Read from memory, write to handle
void read_from_dnnl_memory(void *, const dnnl::memory &);

#endif //DNNLFASHIONMLP_DNNL_UTILS_HPP
