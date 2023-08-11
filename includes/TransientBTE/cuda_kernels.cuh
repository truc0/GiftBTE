#ifndef TRANSIENT_BTE_CUDA_KERNELS_H
#define TRANSIENT_BTE_CUDA_KERNELS_H

#include "TransientBTE/transient.h"

/* energyDensity calculation kernel */
void copyEnergyDensityToGPU(Transient& solver);
void copyEnergyDensityToCPU(Transient& solver);

__global__ void
calcEnergyDensity(double deltaT, double *d_energyDensity, const double *d_Re, const double *d_relaxationTime);

#endif