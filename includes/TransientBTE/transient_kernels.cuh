#ifdef USE_GPU
#ifndef TRANSIENT_KERNELS_CUH
#define TRANSIENT_KERNELS_CUH

#include "TransientBTE/transient.h"

#ifdef USE_GPU
#define __cuda_device__ __device__
#define __cuda_shared__ __shared__
#else
#define __cuda_device__
#define __cuda_shared__
#endif

void copy_to_device(Transient& solver);
void copy_from_device(Transient& solver);

__global__ void ttdr_iterate(const Transient& solver, int nt);
__device__ void _gpu_get_gradient_larger(const Transient& solver, int Use_limiter, int iband_local, int inf_local);
__device__ void _gpu_get_explicit_Re(const Transient& solver, int itime, int spatial_order, int Use_limiter,int iband_local, int inf_local,double deltaTime);
__device__ void _gpu_get_bound_ee(const Transient& solver, int iband_local, int inf_local);
__device__ void _gpu_recover_temperature(const Transient& solver, int iband, int inf_local);
__device__ void _gpu_get_total_energy(const Transient& solver, int iband, int inf_local);
__device__ void _gpu_get_heat_flux(const Transient& solver, int iband, int inf_local);
#endif // TRANSIENT_KERNELS_CUH
#endif //USE_GPU
