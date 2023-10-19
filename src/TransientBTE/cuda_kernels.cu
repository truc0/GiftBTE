#include "TransientBTE/cuda_kernels.cuh"
#include "utility/cuda_utility.cuh"

#define SGN(x) ((x) > 0 ? 1.0 : ((x) < 0 ? -1.0 : 0.0))

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


__global__ void
calcGetGradientLargerDimension1(const int *d_elementFaceBound, const double *d_energyDensity,
                                const double *d_elementVolume, double *d_gradientX) {
    const auto ie = threadIdx.x;
    if (d_elementFaceBound[ie * 6] == -1 &&
        d_elementFaceBound[ie * 6 + 1] == -1) {
        double s1 = (d_energyDensity[ie] - d_energyDensity[ie - 1]) /
                    (d_elementVolume[ie] / 2 + d_elementVolume[ie - 1] / 2);
        double s2 = (d_energyDensity[ie + 1] - d_energyDensity[ie]) /
                    (d_elementVolume[ie] / 2 + d_elementVolume[ie + 1] / 2);
        if ((abs(s1) + abs(s2)) != 0)
            d_gradientX[ie] =
                    (SGN(s1) + SGN(s2)) * abs(s1) * abs(s2) / (abs(s1) + abs(s2));
        else
            d_gradientX[ie] = 0;
    } else if (d_elementFaceBound[ie * 6] != -1) {
        double s1 = (d_energyDensity[ie + 1] -
                     d_energyDensity[ie]) /
                    (d_elementVolume[ie] / 2 + d_elementVolume[ie + 1] / 2);
        double s2 = s1;
        if ((abs(s1) + abs(s2)) != 0)
            d_gradientX[ie] =
                    (SGN(s1) + SGN(s2)) * abs(s1) * abs(s2) / (abs(s1) + abs(s2));
        else
            d_gradientX[ie] = 0;
    } else if (d_elementFaceBound[ie * 6 + 1] != -1) {
        double s1 = (d_energyDensity[ie] -
                     d_energyDensity[ie - 1]) /
                    (d_elementVolume[ie] / 2 + d_elementVolume[ie - 1] / 2);
        double s2 = s1;
        if ((abs(s1) + abs(s2)) != 0)
            d_gradientX[ie] =
                    (SGN(s1) + SGN(s2)) * abs(s1) * abs(s2) / (abs(s1) + abs(s2));
        else
            d_gradientX[ie] = 0;
    }
}

__global__ void
calcGetGradientLargerDimension2(double L_x, int numCell, double *d_gradientX, double *d_gradientY, double *d_gradientZ,
                                const int *d_elementNeighborList, const int *d_elementNeighborListSize,
                                const double *d_energyDensity, const double *d_cellMatrix) {
    const auto i = threadIdx.x;
    if (d_elementNeighborListSize[i] < 3) {
        d_gradientX[i] = 0;
        d_gradientY[i] = 0;
        d_gradientZ[i] = 0;
    } else {
        double *d1; // numCell
        cudaMalloc(&d1, d_elementNeighborListSize[i] * sizeof(double));
        for (int j = 0; j < d_elementNeighborListSize[i]; ++j) {
            d1[j] = 1.0 / L_x * (d_energyDensity[d_elementNeighborList[i * numCell + j]] - d_energyDensity[i]);
        }
        double gradientX = 0;
        double gradientY = 0;
        double gradientZ = 0;
        for (int m = 0; m < d_elementNeighborListSize[i]; ++m) {
            // CellMatrix: numCell * 3 * numCell
            gradientX += d_cellMatrix[i * numCell * 3 + 0 * numCell + m] * d1[m];
            gradientY += d_cellMatrix[i * numCell * 3 + 1 * numCell + m] * d1[m];
        }
        if (isnan(gradientX) || isnan(gradientY) || isnan(gradientZ)) {
            d_gradientX[i] = 0;
            d_gradientY[i] = 0;
            d_gradientZ[i] = 0;
        } else {
            d_gradientX[i] = gradientX;
            d_gradientY[i] = gradientY;
            d_gradientZ[i] = gradientZ;
        }
        cudaFree(d1);
    }
}

__global__ void
calcGetGradientLargerDimension3(double L_x, int numCell, double *d_gradientX, double *d_gradientY, double *d_gradientZ,
                                const int *d_elementNeighborList, const int *d_elementNeighborListSize,
                                const double *d_energyDensity, const double *d_cellMatrix) {
    const auto i = threadIdx.x;

    if (d_elementNeighborListSize[i] < 3) {
        d_gradientX[i] = 0;
        d_gradientY[i] = 0;
        d_gradientZ[i] = 0;
    } else {
//        vector<double> d1(d_elementNeighborListSize[i], 0);
        double *d1;
        cudaMalloc(&d1, d_elementNeighborListSize[i] * sizeof(double));
        for (int j = 0; j < d_elementNeighborList[i]; ++j) {
            d1[j] = 1.0 / L_x * (d_energyDensity[d_elementNeighborList[i * numCell + j]] - d_energyDensity[i]);
        }
        double gradientX = 0;
        double gradientY = 0;
        double gradientZ = 0;
        for (int m = 0; m < d_elementNeighborListSize[i]; ++m) {
            gradientX += d_cellMatrix[i * 3 * numCell + 0 * numCell + m] * d1[m];
            gradientY += d_cellMatrix[i * 3 * numCell + 1 * numCell + m] * d1[m];
            gradientZ += d_cellMatrix[i * 3 * numCell + 2 * numCell + m] * d1[m];
        }
        if (isnan(gradientX) || isnan(gradientY) || isnan(gradientZ)) {
            d_gradientX[i] = 0;
            d_gradientY[i] = 0;
            d_gradientZ[i] = 0;
        } else {
            d_gradientX[i] = gradientX;
            d_gradientY[i] = gradientY;
            d_gradientZ[i] = gradientZ;
        }
        cudaFree(d1);
    }
}


// magic = iband * numDirection * numBound * 2 + inf + numBound * 2
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
                              const double *d_elementFaceNormZ) {
    const auto i = threadIdx.x;

    double max = d_energyDensity[i];
    double min = d_energyDensity[i];

    for (int j = 0; j < d_elementFaceSize[i]; ++j) {
        if (d_elementFaceBound[i * 6 + j] == -1) {
            if (d_energyDensity[d_elementFaceNeighbor[j + i * 6]] > max) {
                max = d_energyDensity[d_elementFaceNeighbor[j + i * 6]];
            }
            if (d_energyDensity[d_elementFaceNeighbor[j + i * 6]] < min) {
                min = d_energyDensity[d_elementFaceNeighbor[j + i * 6]];
            }
        } else if (d_boundaryType[d_elementFaceBound[i * 6 + j]] != 3 &&
                   d_groupVelocityX[i] * d_elementFaceNormX[j + i * 6] +
                   d_groupVelocityY[i] * d_elementFaceNormY[j + i * 6] +
                   d_groupVelocityZ[i] * d_elementFaceNormZ[j + i * 6] < 0) {
            if (d_ebound[magic + d_elementFaceBound[i * 6 + j] * 2 + 0] > max) {
                max = d_ebound[magic + d_elementFaceBound[i * 6 + j] * 2 + 0];
            }
            if (d_ebound[magic + d_elementFaceBound[i * 6 + j] * 2 + 0] < min) {
                min = d_ebound[magic + d_elementFaceBound[i * 6 + j] * 2 + 0];
            }
        }
    }
    for (int j = 0; j < d_elementFaceSize[i]; ++j) {
        double ax = d_elementFaceCenterX[i * 6 + j] - d_elementCenterX[i];
        double ay = d_elementFaceCenterY[i * 6 + j] - d_elementCenterY[i];
        double az = d_elementFaceCenterZ[i * 6 + j] - d_elementCenterZ[i];
        if ((d_gradientX[i] * ax + d_gradientY[i] * ay + d_gradientZ[i] * az) * d_limit[i] + d_energyDensity[i] < min) {
            double y = (min - d_energyDensity[i]) / (d_gradientX[i] * ax + d_gradientY[i] * ay + d_gradientZ[i] * az);
            d_limit[i] = (pow(y, 2) + 2 * y) / (pow(y, 2) + y + 2);
        }
        if ((d_gradientX[i] * ax + d_gradientY[i] * ay + d_gradientZ[i] * az) * d_limit[i] + d_energyDensity[i] > max) {
            double y = (max - d_energyDensity[i]) / (d_gradientX[i] * ax + d_gradientY[i] * ay + d_gradientZ[i] * az);
            d_limit[i] = (pow(y, 2) + 2 * y) / (pow(y, 2) + y + 2);
        }
    }
}


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
                  const double *d_heatCapacity, const double *d_relaxationTime, const double *d_temperatureOld) {
    const auto ie = threadIdx.x;
    for (int jface = 0; jface < d_elementFaceSize[ie]; ++jface) {
        double dotproduct = (d_groupVelocityX[ie] *
                             d_elementFaceNormX[jface + ie * 6] +
                             d_groupVelocityY[ie] *
                             d_elementFaceNormY[jface + ie * 6] +
                             d_groupVelocityZ[ie] *
                             d_elementFaceNormZ[jface + ie * 6]);
        double temp = d_elementFaceArea[jface + ie * 6] / d_elementVolume[ie] * dotproduct; //
        if (dotproduct >= 0) {
            double ax = d_elementFaceCenterX[jface + ie * 6] - d_elementCenterX[ie];
            double ay = d_elementFaceCenterY[jface + ie * 6] - d_elementCenterY[ie];
            double az = d_elementFaceCenterZ[jface + ie * 6] - d_elementCenterZ[ie];
            double e =
                    (d_energyDensity[ie] +
                     (ax * d_gradientX[ie] + ay * d_gradientY[ie] + az * d_gradientZ[ie]) *
                     d_limit[ie]);
            d_Re[ie] += temp * e;
        } else if (d_elementFaceBound[jface + ie * 6] == -1) {
            int neiindex = d_elementFaceNeighbor[jface + ie * 6];
            double ax = d_elementFaceCenterX[jface + ie * 6] - d_elementCenterX[neiindex];
            double ay = d_elementFaceCenterY[jface + ie * 6] - d_elementCenterY[neiindex];
            double az = d_elementFaceCenterZ[jface + ie * 6] - d_elementCenterZ[neiindex];
            double e = (d_energyDensity[neiindex] +
                        (ax * d_gradientX[neiindex] + ay * d_gradientY[neiindex] + az * d_gradientZ[neiindex]) *
                        d_limit[neiindex]);
            d_Re[ie] += temp * e;
        }
    }
    // equlibrium
    d_Re[ie] -= d_temperatureOld[ie] * d_heatCapacity[ie] / d_relaxationTime[ie];
    // TDTR_heatsource
    if (use_TDTR == 1) {
        double heatindex = 0;
        int times = pulse_time / deltaTime;
        int new_itime = itime / times;
        int tt = 1.0 / repetition_frequency * 1e-6 / deltaTime;
        int numheat = (new_itime) / tt;
        int checkheat = (new_itime) - numheat * tt;
        double interg = 1.0;
        double TT = 1.0 / (tt * (repetition_frequency / modulation_frequency) *
                           deltaTime);
        if (checkheat > 1) {
            heatindex = 0;
        } else {
            interg = interg + (double) (new_itime);
            heatindex = sin(interg * deltaTime * 2 * PI * TT) + 1;
        }
        double h = heatindex * d_heatRatio[ie] * d_elementHeatSource[ie];
        d_Re[ie] -= h;
    }
}