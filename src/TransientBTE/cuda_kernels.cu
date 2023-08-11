#include "TransientBTE/cuda_kernels.cuh"
#include "utility/cuda_utility.cuh"

__global__ void
calcEnergyDensity(double deltaT, double *d_energyDensity, const double *d_Re, const double *d_relaxationTime) {
    const auto iCell = threadIdx.x;
    d_energyDensity[iCell] = d_energyDensity[iCell] * (1 - deltaT / d_relaxationTime[iCell]) - deltaT * d_Re[iCell];
}


__global__ void
calcRecoverTemperature(double *d_temperature, const double *d_latticeRatio, const double *d_energyDensity,
                       const double *d_modeWeight, const double *d_heatCapacity) {
    const auto ie = threadIdx.x;
    d_temperature[ie] += d_latticeRatio[ie] * d_energyDensity[ie] * d_modeWeight[ie] / d_heatCapacity[ie];
}


__global__ void
calcGetTotalEnergy(double *d_totalEnergy, const double *d_energyDensity, const double *d_modeWeight,
                   const double *d_capacityBulk) {
    const auto ie = threadIdx.x;
    d_totalEnergy[ie] += d_energyDensity[ie] * d_modeWeight[ie] / d_capacityBulk[ie];
}


__global__ void
calcGetHeatFlux(double *d_heatFlux, const double *d_groupVelocity, const double *d_modeWeight,
                const double *d_energyDensity) {
    const auto ie = threadIdx.x;
    d_heatFlux[ie] += d_groupVelocity[ie] * d_modeWeight[ie] * d_energyDensity[ie];
}
