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
    MIGRATE_TO_DEVICE_1D(elementFaceBound, numCell, double);
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

__device__ void _gpu_get_gradient_larger(const Transient& solver, int Use_limiter, int iband_local, int inf_local) {}
__device__ void _gpu_get_explicit_Re(const Transient& solver, int itime, int spatial_order, int Use_limiter,int iband_local, int inf_local,double deltaTime) {}
__device__ void _gpu_get_bound_ee(const Transient& solver, int iband_local, int inf_local) {}
__device__ void _gpu_recover_temperature(const Transient& solver, int iband, int inf_local) {}
__device__ void _gpu_get_total_energy(const Transient& solver, int iband, int inf_local) {}
__device__ void _gpu_get_heat_flux(const Transient& solver, int iband, int inf_local) {}
