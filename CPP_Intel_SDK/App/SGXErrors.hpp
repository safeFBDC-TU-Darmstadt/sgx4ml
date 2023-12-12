//
// Created by adrian on 19.08.22.
//

#ifndef SGX_DNNL_SGXERRORS_HPP
#define SGX_DNNL_SGXERRORS_HPP

#include "sgx_error.h"

struct sgx_errlist_t {
    const sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
};

/* Error code returned by sgx_create_enclave */
constexpr std::array<sgx_errlist_t, 16> sgx_errlist = {
        {
                {
                        SGX_ERROR_UNEXPECTED,
                        "Unexpected error occurred.",
                        nullptr
                },
                {
                        SGX_ERROR_INVALID_PARAMETER,
                        "Invalid parameter.",
                        nullptr
                },
                {
                        SGX_ERROR_OUT_OF_MEMORY,
                        "Out of memory.",
                        nullptr
                },
                {
                        SGX_ERROR_ENCLAVE_LOST,
                        "Power transition occurred.",
                        "Please refer to the sample \"PowerTransition\" for details."
                },
                {
                        SGX_ERROR_INVALID_ENCLAVE,
                        "Invalid enclave image.",
                        nullptr
                },
                {
                        SGX_ERROR_INVALID_ENCLAVE_ID,
                        "Invalid enclave identification.",
                        nullptr
                },
                {
                        SGX_ERROR_INVALID_SIGNATURE,
                        "Invalid enclave signature.",
                        nullptr
                },
                {
                        SGX_ERROR_OUT_OF_EPC,
                        "Out of EPC memory.",
                        nullptr
                },
                {
                        SGX_ERROR_NO_DEVICE,
                        "Invalid SGX device.",
                        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
                },
                {
                        SGX_ERROR_MEMORY_MAP_CONFLICT,
                        "Memory map conflicted.",
                        nullptr
                },
                {
                        SGX_ERROR_INVALID_METADATA,
                        "Invalid enclave metadata.",
                        nullptr
                },
                {
                        SGX_ERROR_DEVICE_BUSY,
                        "SGX device was busy.",
                        nullptr
                },
                {
                        SGX_ERROR_INVALID_VERSION,
                        "Enclave version was invalid.",
                        nullptr
                },
                {
                        SGX_ERROR_INVALID_ATTRIBUTE,
                        "Enclave was not authorized.",
                        nullptr
                },
                {
                        SGX_ERROR_ENCLAVE_FILE_ACCESS,
                        "Can't open enclave file.",
                        nullptr
                },
                {
                        SGX_ERROR_MEMORY_MAP_FAILURE,
                        "Failed to reserve memory for the enclave.",
                        nullptr
                },
        }
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret) {
    for (const auto & idx : sgx_errlist) {
        if (ret == idx.err) {
            if (nullptr != idx.sug)
                printf("Info: %s\n", idx.sug);
            printf("Error: %s\n", idx.msg);
            return;
        }
    }
    printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", ret);
}

#endif //SGX_DNNL_SGXERRORS_HPP
