#ifndef TRANSIENT_BTE_CUDA_KERNELS_H
#define TRANSIENT_BTE_CUDA_KERNELS_H

#include "TransientBTE/transient.h"

/* energyDensity calculation kernel */
void copyEnergyDensityToGPU(Transient &solver);

void copyEnergyDensityToCPU(Transient &solver);

__global__ void
calcEnergyDensity(double deltaT, double *d_energyDensity, const double *d_Re, const double *d_relaxationTime);

__global__ void
calcRecoverTemperature(double *d_temperature, const double *d_latticeRatio, const double *d_energyDensity,
                       const double *d_modeWeight, const double *d_heatCapacity);

__global__ void
calcGetTotalEnergy(double *d_totalEnergy, const double *d_energyDensity, const double *d_modeWeight,
                   const double *d_capacityBulk);

__global__ void
calcGetHeatFlux(double *d_heatFlux, const double *d_groupVelocity, const double *d_modeWeight,
                const double *d_energyDensity);
#endif