#define MIGRATE_TO_DEVICE_1D(devicePtr, hostPtr, size, type) \
    cudaMalloc(&(devicePtr), size * sizeof(type)); \
    cudaMemcpy((devicePtr), (hostPtr), size * sizeof(type), cudaMemcpyHostToDevice);

#define MIGRATE_TO_DEVICE_2D(devicePtr, hostPtr, dim1, dim0, type) \
    cudaMalloc(&(devicePtr), dim1 * dim0 * sizeof(type)); \
    for (unsigned i_dim1 = 0; i_dim1 < dim1; ++i_dim1) { \
        cudaMemcpy((devicePtr) + i_dim1 * dim0, \
            (hostPtr)[i_dim1], dim0 * sizeof(type), cudaMemcpyHostToDevice); \
    }

#define MIGRATE_TO_DEVICE_3D(devicePtr, hostPtr, dim2, dim1, dim0, type) \
    cudaMalloc(&(devicePtr), dim2 * dim1 * dim0 * sizeof(type)); \
    for (unsigned i_dim2 = 0; i_dim2 < dim2; ++i_dim2) { \
        for (unsigned i_dim1 = 0; i_dim1 < dim1; ++i_dim1) { \
            cudaMemcpy((devicePtr) + i_dim2 * dim1 * dim0 + i_dim1 * dim0, \
                (hostPtr)[i_dim2][i_dim1], dim0 * sizeof(type), cudaMemcpyHostToDevice); \
        } \
    }

#define MIGRATE_TO_HOST_1D(hostPtr, devicePtr, size, type) \
    cudaMemcpy((hostPtr), (devicePtr), size * sizeof(type), cudaMemcpyDeviceToHost); \
    cudaFree(devicePtr);               \
    devicePtr = nullptr;

#define MIGRATE_TO_HOST_2D(hostPtr, devicePtr, dim1, dim0, type) \
    for (unsigned i_dim1 = 0; i_dim1 < dim1; ++i_dim1) { \
        cudaMemcpy((hostPtr)[i_dim1], (devicePtr) + i_dim1 * dim0, \
            dim0 * sizeof(type), cudaMemcpyDeviceToHost); \
    } \
    cudaFree(devicePtr); \
    devicePtr = nullptr;

#define MIGRATE_TO_HOST_3D(hostPtr, devicePtr, dim2, dim1, dim0, type) \
    for (unsigned i_dim2 = 0; i_dim2 < dim2; ++i_dim2) { \
        for (unsigned i_dim1 = 0; i_dim1 < dim1; ++i_dim1) { \
            cudaMemcpy((hostPtr) + i_dim2 * dim1 * dim0 + i_dim1 * dim0, \
                (devicePtr)[i_dim2][i_dim1], dim0 * sizeof(type), cudaMemcpyHostToDevice); \
        } \
    } \
    cudaFree(devicePtr);                           \
    devicePtr = nullptr;

inline void cudaErrorCheck(cudaError code, const char *file = nullptr, const int line = 0, bool abort = true) {
    if (code != cudaSuccess) {
        if (file != nullptr) {
            std::cerr << "CUDA error: " << file << ' ' << line << ' '
                      << cudaGetErrorString(code) << '\n';
        } else {
            std::cerr << "CUDA error: " << cudaGetErrorString(code) << '\n';
        }
        if (abort) exit(code);
    }
}

// CUDA Error Check
#define CUDAECHK(ans) { cudaErrorCheck((ans), __FILE__, __LINE__); }
