#include "TransientBTE/transient_kernels.cuh"

/**
 * Copy the required data from the host to the device
 *
 * The origin data is store with prefix `host_` after processing, and the pointer to
 * data copy on device will be saved using the original name.
 * Don't forget to call `copy_from_device` after the calculation.
 *
 * The replaced data are listed as following:
 * - gradient{X,Y,Z}
 * - limit
 * - elementFaceBound
 * - elementFaceSize
 * - energyDensity
 * - elementVolume
 * - elementNeighborList
 * - CellMatrix
 * - groupVelocity{X,Y,Z}
 * - elementFaceNorm{X,Y,Z}
 * - ebound
 * - elementCenter{X,Y,Z}
 *
 * - elementFaceArea
 * - elementFaceCenter{X,Y,Z}
 * - elementFaceNeighbor
 * - Re
 * - temperatureOld
 * - heatCapacity
 * - matter
 * - relaxationTime
 * - heatRatio
 * - elementHeatSource
 * - boundaryCell
 * - boundaryFace
 *
 * - eboundLocal
 * - latticeRatio
 * - temperatureLocal
 * - modeWeight
 * - totalEnergyLocal
 * - capacityBulk
 * - heatFlux{X,Y,Z}Local
 *
 * @param solver A transient solver instance
 */
void copy_to_device(Transient& solver)
{
    // TODO: change to cudaMemcpyAsync
#define MIGRATE_TO_DEVICE_1D(variable, size, type) \
    cudaMalloc(&(solver.d_##variable), size * sizeof(type)); \
    cudaMemcpy(solver.d_##variable, solver.variable, size * sizeof(type), cudaMemcpyHostToDevice);

#define MIGRATE_TO_DEVICE_2D(variable, dim1, dim0, type) \
    cudaMalloc(&(solver.d_##variable), dim1 * dim0 * sizeof(type)); \
    for (unsigned i_dim1 = 0; i_dim1 < dim1; ++i_dim1) { \
        cudaMemcpy(solver.d_##variable + i_dim1 * dim0, \
            solver.variable[i_dim1], dim0 * sizeof(type), cudaMemcpyHostToDevice); \
    }

#define MIGRATE_TO_DEVICE_3D(variable, dim2, dim1, dim0, type) \
    cudaMalloc(&(solver.d_##variable), dim2 * dim1 * dim0 * sizeof(type)); \
    for (unsigned i_dim2 = 0; i_dim2 < dim2; ++i_dim2) { \
        for (unsigned i_dim1 = 0; i_dim1 < dim1; ++i_dim1) { \
            cudaMemcpy(solver.d_##variable + i_dim2 * dim1 * dim0 + i_dim1 * dim0, \
                solver.variable[i_dim2][i_dim1], dim0 * sizeof(type), cudaMemcpyHostToDevice); \
        } \
    }


    // helper variables
    const auto numCell = solver.numCell;
    const auto numBand = solver.numBand;
    const auto numBound = solver.numBound;
    const auto numDirection = solver.numDirection;
    const auto numofMatter = solver.numofMatter;

    MIGRATE_TO_DEVICE_1D(gradientX, numCell, double);
    MIGRATE_TO_DEVICE_1D(gradientY, numCell, double);
    MIGRATE_TO_DEVICE_1D(gradientZ, numCell, double);
    MIGRATE_TO_DEVICE_1D(limit, numCell, double);
    MIGRATE_TO_DEVICE_1D(elementFaceSize, numCell, int);
    MIGRATE_TO_DEVICE_1D(elementFaceBound, numCell * 6, int);
    MIGRATE_TO_DEVICE_3D(energyDensity, numBand, numDirection, numCell, double);
    MIGRATE_TO_DEVICE_1D(elementVolume, numCell, double);
    // TODO: elementNeighborList
    // TODO: CellMatrix
    MIGRATE_TO_DEVICE_3D(groupVelocityX, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_DEVICE_3D(groupVelocityY, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_DEVICE_3D(groupVelocityZ, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_DEVICE_1D(elementFaceNormX, numCell * 6, double);
    MIGRATE_TO_DEVICE_1D(elementFaceNormY, numCell * 6, double);
    MIGRATE_TO_DEVICE_1D(elementFaceNormZ, numCell * 6, double);
    MIGRATE_TO_DEVICE_1D(ebound, numBand * numDirection * numBand * 2, double);
    MIGRATE_TO_DEVICE_1D(elementCenterX, numCell, double);
    MIGRATE_TO_DEVICE_1D(elementCenterY, numCell, double);
    MIGRATE_TO_DEVICE_1D(elementCenterZ, numCell, double);

    MIGRATE_TO_DEVICE_1D(elementFaceArea, numCell * 6, int);
    MIGRATE_TO_DEVICE_1D(elementFaceCenterX, numCell * 6, double);
    MIGRATE_TO_DEVICE_1D(elementFaceCenterY, numCell * 6, double);
    MIGRATE_TO_DEVICE_1D(elementFaceCenterZ, numCell * 6, double);
    MIGRATE_TO_DEVICE_1D(elementFaceNeighobr, numCell * 6, int);
    MIGRATE_TO_DEVICE_1D(Re, numCell, double);
    MIGRATE_TO_DEVICE_1D(temperatureOld, numCell, double);
    MIGRATE_TO_DEVICE_3D(heatCapacity, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_DEVICE_1D(matter, numCell, int);
    MIGRATE_TO_DEVICE_3D(relaxationTime, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_DEVICE_1D(elementHeatSource, numCell, double);
    MIGRATE_TO_DEVICE_2D(boundaryCell, numBound, 2, int);
    MIGRATE_TO_DEVICE_2D(boundaryFace, numBound, 2, int);

    MIGRATE_TO_DEVICE_1D(eboundLocal, sizeof(solver.eboundLocal) / sizeof(int), int);
    MIGRATE_TO_DEVICE_3D(latticeRatio, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_DEVICE_1D(temperatureLocal, numCell, double);
    MIGRATE_TO_DEVICE_3D(modeWeight, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_DEVICE_1D(totalEnergyLocal, numCell, double);
    MIGRATE_TO_DEVICE_1D(capacityBulk, numofMatter, double);
    MIGRATE_TO_DEVICE_1D(heatFluxXLocal, numCell, double);
    MIGRATE_TO_DEVICE_1D(heatFluxYLocal, numCell, double);
    MIGRATE_TO_DEVICE_1D(heatFluxZLocal, numCell, double);

    cudaDeviceSynchronize();

#undef MIGRATE_TO_DEVICE_3D
#undef MIGRATE_TO_DEVICE_2D
#undef MIGRATE_TO_DEVICE_1D
}

void copy_from_device(Transient& solver)
{
    // TODO: change to cudaMemcpyAsync
#define MIGRATE_TO_HOST_1D(variable, size, type) \
    cudaMemcpy(solver.variable, solver.d_##variable, size * sizeof(type), cudaMemcpyDeviceToHost); \
    cudaFree(solver.d_##variable);               \
    solver.d_##variable = nullptr;

#define MIGRATE_TO_HOST_2D(variable, dim1, dim0, type) \
    for (unsigned i_dim1 = 0; i_dim1 < dim1; ++i_dim1) { \
        cudaMemcpy(solver.variable[i_dim1], solver.d_##variable + i_dim1 * dim0, \
            dim0 * sizeof(type), cudaMemcpyDeviceToHost); \
    } \
    cudaFree(solver.d_##variable); \
    solver.d_##variable = nullptr;

#define MIGRATE_TO_HOST_3D(variable, dim2, dim1, dim0, type) \
    for (unsigned i_dim2 = 0; i_dim2 < dim2; ++i_dim2) { \
        for (unsigned i_dim1 = 0; i_dim1 < dim1; ++i_dim1) { \
            cudaMemcpy(solver.d_##variable + i_dim2 * dim1 * dim0 + i_dim1 * dim0, \
                solver.variable[i_dim2][i_dim1], dim0 * sizeof(type), cudaMemcpyHostToDevice); \
        } \
    } \
    cudaFree(solver.d_##variable);                           \
    solver.d_##variable = nullptr;                           \

    // helper variables
    const auto numCell = solver.numCell;
    const auto numBand = solver.numBand;
    const auto numBound = solver.numBound;
    const auto numDirection = solver.numDirection;
    const auto numofMatter = solver.numofMatter;

    MIGRATE_TO_HOST_1D(gradientX, numCell, double);
    MIGRATE_TO_HOST_1D(gradientY, numCell, double);
    MIGRATE_TO_HOST_1D(gradientZ, numCell, double);
    MIGRATE_TO_HOST_1D(limit, numCell, double);
    MIGRATE_TO_HOST_1D(elementFaceSize, numCell, double);
    MIGRATE_TO_HOST_1D(elementFaceBound, numCell, double);
    MIGRATE_TO_HOST_3D(energyDensity, numBand, numDirection, numCell, double);
    MIGRATE_TO_HOST_1D(elementVolume, numCell, double);
    // TODO: elementNeighborList
    // TODO: CellMatrix
    MIGRATE_TO_HOST_3D(groupVelocityX, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_HOST_3D(groupVelocityY, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_HOST_3D(groupVelocityZ, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_HOST_1D(elementFaceNormX, numCell * 6, double);
    MIGRATE_TO_HOST_1D(elementFaceNormY, numCell * 6, double);
    MIGRATE_TO_HOST_1D(elementFaceNormZ, numCell * 6, double);
    MIGRATE_TO_HOST_1D(ebound, numBand * numDirection * numBand * 2, double);
    MIGRATE_TO_HOST_1D(elementCenterX, numCell, double);
    MIGRATE_TO_HOST_1D(elementCenterY, numCell, double);
    MIGRATE_TO_HOST_1D(elementCenterZ, numCell, double);

    MIGRATE_TO_HOST_1D(elementFaceArea, numCell * 6, int);
    MIGRATE_TO_HOST_1D(elementFaceCenterX, numCell * 6, double);
    MIGRATE_TO_HOST_1D(elementFaceCenterY, numCell * 6, double);
    MIGRATE_TO_HOST_1D(elementFaceCenterZ, numCell * 6, double);
    MIGRATE_TO_HOST_1D(elementFaceNeighobr, numCell * 6, int);
    MIGRATE_TO_HOST_1D(Re, numCell, double);
    MIGRATE_TO_HOST_1D(temperatureOld, numCell, double);
    MIGRATE_TO_HOST_3D(heatCapacity, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_HOST_1D(matter, numCell, int);
    MIGRATE_TO_HOST_3D(relaxationTime, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_HOST_1D(elementHeatSource, numCell, double);
    MIGRATE_TO_HOST_2D(boundaryCell, numBound, 2, int);
    MIGRATE_TO_HOST_2D(boundaryFace, numBound, 2, int);

    MIGRATE_TO_HOST_1D(eboundLocal, sizeof(solver.eboundLocal) / sizeof(int), int);
    MIGRATE_TO_HOST_3D(latticeRatio, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_HOST_1D(temperatureLocal, numCell, double);
    MIGRATE_TO_HOST_3D(modeWeight, numofMatter, numBand, numDirection, double);
    MIGRATE_TO_HOST_1D(totalEnergyLocal, numCell, double);
    MIGRATE_TO_HOST_1D(capacityBulk, numofMatter, double);
    MIGRATE_TO_HOST_1D(heatFluxXLocal, numCell, double);
    MIGRATE_TO_HOST_1D(heatFluxYLocal, numCell, double);
    MIGRATE_TO_HOST_1D(heatFluxZLocal, numCell, double);

    cudaDeviceSynchronize();

#undef MIGRATE_TO_HOST_3D
#undef MIGRATE_TO_HOST_2D
#undef MIGRATE_TO_HOST_1D
}

__global__ void ttdr_iterate(const Transient& solver, int nt)
{
    const int inf_local = blockIdx.x;
    const int iband_local = threadIdx.x;
    const int inf = ((inf_local) * solver.numProc + solver.worldRank) % solver.numDirection;
    const int iband = iband_local * (ceil(double(solver.numProc) / double(solver.numDirection))) +
                      solver.worldRank / solver.numDirection;

    _gpu_get_gradient_larger(solver, 0, iband_local, inf_local);
    _gpu_get_explicit_Re(solver, nt, 2, 0, iband_local, inf_local, solver.deltaT);

    for (int icell = 0; icell < solver.numCell; ++icell) {
        solver.energyDensity[iband_local][inf_local][icell] *=
                (1 - solver.deltaT / solver.relaxationTime[solver.matter[icell]][iband][inf]) -
                solver.deltaT * solver.Re[icell];
    }

    _gpu_get_bound_ee(solver, iband_local, inf_local);
    _gpu_recover_temperature(solver, iband_local, inf_local);
    _gpu_get_total_energy(solver, iband_local, inf_local);
    _gpu_get_heat_flux(solver, iband_local, inf_local);

//    MPI_Allreduce(solver.totalEnergyLocal, solver.totalEnergy, solver.numCell, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//    MPI_Allreduce(solver.temperatureLocal, solver.temperature, solver.numCell, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//    MPI_Allreduce(solver.heatFluxXLocal, solver.heatFluxXGlobal, solver.numCell, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//    MPI_Allreduce(solver.heatFluxYLocal, solver.heatFluxYGlobal, solver.numCell, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//    MPI_Allreduce(solver.heatFluxZLocal, solver.heatFluxZGlobal, solver.numCell, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

__device__ void _gpu_get_gradient_larger(const Transient& solver, int Use_limiter, int iband_local, int inf_local)
{

}

__device__ void _gpu_get_explicit_Re(const Transient& solver, int itime, int spatial_order, int Use_limiter,int iband_local, int inf_local,double deltaTime)
{
    const auto numCell = solver.numCell;
    const auto numBand = solver.numBand;
    const auto numBound = solver.numBound;
    const auto numDirection = solver.numDirection;
    const auto numofMatter = solver.numofMatter;
    const auto numProc = solver.numProc;
    const auto worldRank = solver.worldRank;

    /* solver variables on device */
    const auto Re = solver.d_Re;
    const auto limit = solver.d_limit;
    const auto matter = solver.d_matter;

    const auto elementCenterX = solver.d_elementCenterX;
    const auto elementCenterY = solver.d_elementCenterY;
    const auto elementCenterZ = solver.d_elementCenterZ;

    const auto elementFaceSize = solver.d_elementFaceSize;
    const auto elementFaceArea = solver.d_elementFaceArea;
    const auto elementFaceBound = solver.d_elementFaceBound;

    const auto groupVelocityX = solver.d_groupVelocityX;
    const auto groupVelocityY = solver.d_groupVelocityY;
    const auto groupVelocityZ = solver.d_groupVelocityZ;

    const auto elementFaceNormX = solver.d_elementFaceNormX;
    const auto elementFaceNormY = solver.d_elementFaceNormY;
    const auto elementFaceNormZ = solver.d_elementFaceNormZ;

    const auto elementFaceCenterX = solver.d_elementFaceCenterX;
    const auto elementFaceCenterY = solver.d_elementFaceCenterY;
    const auto elementFaceCenterZ = solver.d_elementFaceCenterZ;

    const auto elementVolume = solver.d_elementVolume;
    const auto energyDensity = solver.d_energyDensity;
    const auto elementFaceNeighobr = solver.d_elementFaceNeighobr;
    const auto elementHeatSource = solver.d_elementHeatSource;

    const auto gradientX = solver.d_gradientX;
    const auto gradientY = solver.d_gradientY;
    const auto gradientZ = solver.d_gradientZ;

    const auto heatRatio = solver.d_heatRatio;
    const auto heatCapacity = solver.d_heatCapacity;
    const auto relaxationTime = solver.d_relaxationTime;
    const auto temperatureOld = solver.d_temperatureOld;

    const auto boundaryCell = solver.d_boundaryCell;
    const auto boundaryFace = solver.d_boundaryFace;
    const auto ebound = solver.d_ebound;

    const int inf = ((inf_local) * numProc + worldRank) % numDirection;
    const int iband = iband_local * (ceil(double(numProc) / double(numDirection))) + worldRank / numDirection;

    for (int i = 0; i < numCell; ++i) {
        Re[i] = 0;
    }

    // max_y is the maximum value of elementCenterY
    double max_y = 0;
    for (int ie = 0; ie < numCell; ie++) {
        if (elementCenterY[ie] > max_y) {
            max_y = elementCenterY[ie];
        }
    }

    // internal
    for (int ie = 0; ie < numCell; ++ie) {
        for (int jface = 0; jface < elementFaceSize[ie]; ++jface) {
            const auto groupVelocityIdx = matter[ie] * numBand * numDirection + iband * numDirection + inf;
            const auto elementFaceIdx = jface + ie * 6;
            double dotProduct = (groupVelocityX[groupVelocityIdx] * elementFaceNormX[elementFaceIdx]
                                 + groupVelocityY[groupVelocityIdx] * elementFaceNormY[elementFaceIdx]
                                 + groupVelocityZ[groupVelocityIdx] * elementFaceNormZ[elementFaceIdx]);
            double temp = elementFaceArea[elementFaceIdx] / elementVolume[ie] * dotProduct;           //

            if (dotProduct >= 0 || elementFaceBound[elementFaceIdx] == -1) {
                const auto elementCenterIdx = dotProduct >= 0 ? ie : elementFaceNeighobr[elementFaceIdx];
                const double ax = elementFaceCenterX[elementFaceIdx] - elementCenterX[elementCenterIdx];
                const double ay = elementFaceCenterY[elementFaceIdx] - elementCenterY[elementCenterIdx];
                const double az = elementFaceCenterZ[elementFaceIdx] - elementCenterZ[elementCenterIdx];
                const double e = (energyDensity[iband_local * solver.numDirectionLocal * numCell + inf_local * numCell + elementCenterIdx] +
                                  (ax * gradientX[elementCenterIdx] + ay * gradientY[elementCenterIdx] + az * gradientZ[elementCenterIdx]) * limit[elementCenterIdx]);
                Re[ie] += temp * e;
            }
        }

        // equlibrium
        const auto idx = matter[ie] * numBand * numDirection + iband * numDirection + inf;
        Re[ie] -= temperatureOld[ie] * heatCapacity[idx] / relaxationTime[idx];

        // TDTR_heatSource
        if (solver.use_TDTR == 1) {
            const auto pulse_time = solver.pulse_time;
            const auto repetition_frequency = solver.repetition_frequency;
            const auto modulation_frequency = solver.modulation_frequency;

            double heatIndex = 0;
            const int times = pulse_time / deltaTime;
            const int new_itime = itime / times;
            const int tt = 1.0 / repetition_frequency * 1e-6 / deltaTime;
            const int numheat = (new_itime) / tt;
            const int checkheat = (new_itime) - numheat * tt;
            double interg = 1.0;
            const double TT = 1.0 / (tt * (repetition_frequency / modulation_frequency) * deltaTime);

            if (checkheat > 1) {
                heatIndex = 0;
            } else {
                interg = interg + (double) (new_itime);
                heatIndex = sin(interg * deltaTime * 2 * PI * TT) + 1;
            }
            const double h = heatIndex * heatRatio[matter[ie] * numBand * numDirection + iband * numDirection + inf] *
                       elementHeatSource[ie];
            Re[ie] -= h;
        }
    }

    for (int ib = 0; ib < numBound; ++ib) {
        for (int icell = 0; icell < 2; ++icell) {
            int ie = boundaryCell[ib * 2 + icell];
            int jface = boundaryFace[ib * 2 + icell];

            if (ie >= 0) {
                const auto elementFaceIdx = jface + ie * 6;
                const auto groupVelocityIdx = matter[ie] * numBand * numDirection + iband * numDirection + inf;

                const double dotProduct = (groupVelocityX[groupVelocityIdx] * elementFaceNormX[elementFaceIdx]
                                     + groupVelocityY[groupVelocityIdx] * elementFaceNormY[elementFaceIdx]
                                     + groupVelocityZ[groupVelocityIdx] * elementFaceNormZ[elementFaceIdx]);
                if (dotProduct < 0) {
                    double temp = elementFaceArea[elementFaceIdx] / elementVolume[ie] * dotProduct;
                    Re[ie] += temp * ebound[iband * numDirection * numBound * 2 + inf * numBound * 2 + ib * 2 + icell];
                }
            }
        }
    }
}

__device__ void _gpu_get_bound_ee(const Transient& solver, int iband_local, int inf_local) {}

__device__ void _gpu_recover_temperature(const Transient& solver, int iband_local, int inf_local)
{
    const auto numCell = solver.numCell;
    const auto numProc = solver.numProc;
    const auto worldRank = solver.worldRank;
    const auto numDirection = solver.numDirection;
    const auto numDirectionLocal = solver.numDirectionLocal;
    const auto numBand = solver.numBand;

    const auto matter = solver.d_matter;
    const auto temperatureLocal = solver.d_temperatureLocal;
    const auto latticeRatio = solver.d_latticeRatio;
    const auto energyDensity = solver.d_energyDensity;
    const auto modeWeight = solver.d_modeWeight;
    const auto heatCapacity = solver.d_heatCapacity;

    const int inf = ((inf_local) * numProc + worldRank) % numDirection;
    const int iband = iband_local * (ceil(double(numProc) / double(numDirection))) + worldRank / numDirection;

    for (int ie = 0; ie < numCell; ++ie) {
        const auto mie = matter[ie];
        const auto idx = mie * numBand * numDirection + iband * numDirection + inf;
        temperatureLocal[ie] += latticeRatio[idx] * modeWeight[idx] / heatCapacity[idx]
                * energyDensity[iband_local * numDirectionLocal * numCell + inf_local * numCell + ie];
    }
}

__device__ void _gpu_get_total_energy(const Transient& solver, int iband_local, int inf_local)
{
    const auto numCell = solver.numCell;
    const auto numProc = solver.numProc;
    const auto worldRank = solver.worldRank;
    const auto numDirection = solver.numDirection;
    const auto numDirectionLocal = solver.numDirectionLocal;
    const auto numBand = solver.numBand;

    const auto matter = solver.d_matter;
    const auto energyDensity = solver.d_energyDensity;
    const auto modeWeight = solver.d_modeWeight;
    const auto capacityBulk = solver.d_capacityBulk;
    const auto totalEnergyLocal = solver.totalEnergyLocal;

    int inf = ((inf_local) * numProc + worldRank) % numDirection;
    int iband = iband_local * (ceil(double(numProc) / double(numDirection))) + worldRank / numDirection;

    for (int ie = 0; ie < numCell; ++ie) {
        const auto mie = matter[ie];
        totalEnergyLocal[ie] += energyDensity[iband_local * numDirectionLocal * numCell + inf_local * numCell + ie]
                * modeWeight[mie * numBand * numDirection + iband * numDirection + inf]
                / capacityBulk[mie];
    }
}

__device__ void _gpu_get_heat_flux(const Transient& solver, int iband_local, int inf_local)
{
    const auto numBand = solver.numBand;
    const auto numCell = solver.numCell;
    const auto numProc = solver.numProc;
    const auto worldRank = solver.worldRank;
    const auto numDirection = solver.numDirection;
    const auto numDirectionLocal = solver.numDirectionLocal;

    const auto matter = solver.d_matter;
    const auto heatFluxXLocal = solver.d_heatFluxXLocal;
    const auto heatFluxYLocal = solver.d_heatFluxYLocal;
    const auto heatFluxZLocal = solver.d_heatFluxZLocal;
    const auto groupVelocityX = solver.d_groupVelocityX;
    const auto groupVelocityY = solver.d_groupVelocityY;
    const auto groupVelocityZ = solver.d_groupVelocityZ;
    const auto modeWeight = solver.d_modeWeight;
    const auto energyDensity = solver.d_energyDensity;

    int inf = ((inf_local) * numProc + worldRank) % numDirection;
    int iband = iband_local * (ceil(double(numProc) / double(numDirection))) + worldRank / numDirection;

    for (int ie = 0; ie < numCell; ++ie) {
        const auto idx = matter[ie] * numBand * numDirection + iband * numDirection + inf;
        const auto energyDensityIdx = iband_local * numDirectionLocal * numCell + inf_local * numCell + ie;
        heatFluxXLocal[ie] += groupVelocityX[idx] * modeWeight[idx] * energyDensity[energyDensityIdx];
        heatFluxYLocal[ie] += groupVelocityY[idx] * modeWeight[idx] * energyDensity[energyDensityIdx];
        heatFluxZLocal[ie] += groupVelocityZ[idx] * modeWeight[idx] * energyDensity[energyDensityIdx];
    }
}

