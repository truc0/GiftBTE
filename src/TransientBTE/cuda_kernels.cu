#include "TransientBTE/cuda_kernels.cuh"
#include "utility/cuda_utility.cuh"

__global__ void
calcEnergyDensity(double deltaT, double *d_energyDensity, const double *d_Re, const double *d_relaxationTime) {
    const auto iCell = threadIdx.x;
    d_energyDensity[iCell] = d_energyDensity[iCell] * (1 - deltaT / d_relaxationTime[iCell]) - deltaT * d_Re[iCell];
}