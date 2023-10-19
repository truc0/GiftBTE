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

__global__ void
calcGetGradientLargerDimension1(const int *d_elementFaceBound, const double *d_energyDensity,
                                const double *d_elementVolume, double *d_gradientX);

__global__ void
calcGetGradientLargerDimension2(double L_x, int numCell, double *d_gradientX, double *d_gradientY, double *d_gradientZ,
                                const int *d_elementNeighborList, const int *d_elementNeighborListSize,
                                const double *d_energyDensity, const double *d_cellMatrix);

__global__ void
calcGetGradientLargerDimension3(double L_x, int numCell, double *d_gradientX, double *d_gradientY, double *d_gradientZ,
                                const int *d_elementNeighborList, const int *d_elementNeighborListSize,
                                const double *d_energyDensity, const double *d_cellMatrix);

__global__ void
calcGetGradientLargerUseLimit(int magic, double *d_limit, const double *d_ebound,
                              const int *d_boundaryType, const double *d_energyDensity, const int *d_elementFaceSize,
                              const int *d_elementFaceBound, const int *d_elementFaceNeighbor,
                              const double *d_elementFaceCenterX, const double *d_elementFaceCenterY,
                              const double *d_elementFaceCenterZ, const double *d_elementCenterX,
                              const double *d_elementCenterY, const double *d_elementCenterZ, const double *d_gradientX,
                              const double *d_gradientY, const double *d_gradientZ, const double *d_groupVelocityX,
                              const double *d_groupVelocityY, const double *d_groupVelocityZ,
                              const double *d_elementFaceNormX, const double *d_elementFaceNormY,
                              const double *d_elementFaceNormZ);

__global__ void
calcGetExplicitRe(int use_TDTR, double deltaTime, const int *d_elementFaceSize, double repetition_frequency,
                  double modulation_frequency, double pulse_time, double itime, double *d_Re,
                  const double *d_groupVelocityX, const double *d_groupVelocityY, const double *d_groupVelocityZ,
                  const double *d_elementFaceNormX, const double *d_elementFaceNormY, const double *d_elementFaceNormZ,
                  const double *d_elementFaceArea, const double *d_elementVolume, const double *d_elementFaceCenterX,
                  const double *d_elementFaceCenterY, const double *d_elementFaceCenterZ,
                  const double *d_elementCenterX, const double *d_elementCenterY, const double *d_elementCenterZ,
                  const double *d_energyDensity, const double *d_gradientX, const double *d_gradientY,
                  const double *d_gradientZ, const double *d_limit, const int *d_elementFaceBound,
                  const int *d_elementFaceNeighbor, const double *d_elementHeatSource, const double *d_heatRatio,
                  const double *d_heatCapacity, const double *d_relaxationTime, const double *d_temperatureOld);
#endif