//
// Created by yuehu on 2023/5/21.
//

#ifdef USE_GPU

#include "TransientBTE/cuda_kernels.cuh"
#include "utility/cuda_utility.cuh"

#endif

#include "TransientBTE/transient.h"
#include <algorithm>
#include <chrono>
#include <iomanip>

using namespace std;

void Transient::solve(int Use_Backup, double error_temp_limit,
                      double error_flux_limit, double deltaT, double totalT,
                      int use_TDTR, double pulse_time,
                      double repetition_frequency, double modulation_frequency,
                      double xy_r) {
    double Num_Max_Iter = totalT / deltaT;
    _set_cell_matrix_larger();

    errorIncreaseTime = 0;
    _set_initial(Use_Backup);

    for (int inf_local = 0; inf_local < numDirectionLocal; inf_local++) {
        for (int iband_local = 0; iband_local < numBandLocal; ++iband_local) {
            _get_bound_ee(iband_local, inf_local);
        }
    }
    _set_bound_ee_1();

    for (int ib = 0; ib < numBound; ++ib) {
        for (int inf_local = 0; inf_local < numDirectionLocal; ++inf_local) {
            int inf = ((inf_local) * numProc + worldRank) % numDirection;
            for (int iband_local = 0; iband_local < numBandLocal; ++iband_local) {
                int iband =
                        iband_local * (ceil(double(numProc) / double(numDirection))) +
                        worldRank / numDirection;
                for (int icell = 0; icell < 2; ++icell) {
                    int ie = boundaryCell[ib][icell];
                    int jface = boundaryFace[ib][icell];
                    if (ie >= 0) {
                        if (heatRatio[matter[ie]][iband][inf] != 0) {
                            double dotproduct = (groupVelocityX[matter[ie]][iband][inf] *
                                                 elementFaceNormX[jface + ie * 6] +
                                                 groupVelocityY[matter[ie]][iband][inf] *
                                                 elementFaceNormY[jface + ie * 6] +
                                                 groupVelocityZ[matter[ie]][iband][inf] *
                                                 elementFaceNormZ[jface + ie * 6]);
                            if (dotproduct < 0) {
                                if (boundaryType[ib] == 1) {
                                    double e = heatCapacity[matter[ie]][iband][inf] *
                                               boundaryThermal[ib];
                                    ebound[iband * numDirection * numBound * 2 +
                                           inf * numBound * 2 + ib * 2] = e;
                                } else if (boundaryType[ib] == 2) {
                                    double einsum1 = 0;
                                    double temp1 = 0;
                                    for (int nft = 0; nft < numDirectionLocal; ++nft) {
                                        double dotproduct1 =
                                                (groupVelocityX[matter[ie]][iband][nft] *
                                                 elementFaceNormX[jface + ie * 6] +
                                                 groupVelocityY[matter[ie]][iband][nft] *
                                                 elementFaceNormY[jface + ie * 6] +
                                                 groupVelocityZ[matter[ie]][iband][nft] *
                                                 elementFaceNormZ[jface + ie * 6]);
                                        if (dotproduct1 >= 0) {
                                            einsum1 +=
                                                    energyDensity[iband_local][nft][ib] *
                                                    (dotproduct1 * modeWeight[matter[ie]][iband][nft]);
                                            temp1 +=
                                                    (dotproduct1 * modeWeight[matter[ie]][iband][nft]);
                                        }
                                    }
                                    double e = einsum1 / temp1;
                                    ebound[iband * numDirection * numBound * 2 +
                                           inf * numBound * 2 + ib * 2] = 0; // e;
                                } else if (boundaryType[ib] == 3) {
                                    vec Reflectr;
                                    double dotproduct1 =
                                            (directionX[inf] * elementFaceNormX[jface + ie * 6] +
                                             directionY[inf] * elementFaceNormY[jface + ie * 6] +
                                             directionZ[inf] * elementFaceNormZ[jface + ie * 6]);
                                    double ReflectrX =
                                            directionX[inf] -
                                            elementFaceNormX[jface + ie * 6] * dotproduct1 * 2;
                                    double ReflectrY =
                                            directionY[inf] -
                                            elementFaceNormY[jface + ie * 6] * dotproduct1 * 2;
                                    double ReflectrZ =
                                            directionZ[inf] -
                                            elementFaceNormZ[jface + ie * 6] * dotproduct1 * 2;
                                    // Reflectr = angles->direction[inf] -
                                    // mesh->Elements[ie].faces[jface].norm *
                                    // (angles->direction[inf] *
                                    // mesh->Elements[ie].faces[jface].norm) * 2;
                                    double close = 1;
                                    int nf = -1;
                                    nf = 0; // add ru
                                    for (int k = 0; k < numDirectionLocal; ++k) {
                                        double length = sqrt(pow(ReflectrX - directionX[k], 2) +
                                                             pow(ReflectrY - directionY[k], 2) +
                                                             pow(ReflectrZ - directionZ[k], 2));
                                        if (length < close) {
                                            nf = k;
                                            close = length;
                                        }
                                    }
                                    ebound[iband * numDirection * numBound * 2 +
                                           inf * numBound * 2 + ib * 2] =
                                            energyDensity[iband_local][nf][ib];

                                } else if (boundaryType[ib] < 0) {
                                    double e =
                                            energyDensity[iband_local][inf_local]
                                            [boundaryCell[boundaryConnect[ib]][icell]];
                                    e = e + heatCapacity[matter[ie]][iband][inf] *
                                            (boundaryThermal[ib] -
                                             boundaryThermal[boundaryConnect[ib]]);

                                    ebound[iband * numDirection * numBound * 2 +
                                           inf * numBound * 2 + ib * 2] = e;
                                } else {
                                }
                            } else {
                            }
                        } else {
                        }
                    }
                }
            }
        }
    }
    for (int iband = 0; iband < numBand; ++iband) {
        for (int inf = 0; inf < numDirection; ++inf) {
            for (int ib = 0; ib < numBound * 2; ++ib) {
                // cout<< ebound[iband * numDirection * numBound * 2 + inf * numBound *
                // 2 + ib]<<endl;
            }
        }
    }

    auto total_iter_time = chrono::microseconds(0);
    auto get_gradient_time = chrono::microseconds(0);
    auto get_Re_time = chrono::microseconds(0);
    auto solver1_time = chrono::microseconds(0);
    auto Boundary_time = chrono::microseconds(0);
    auto set_vertex_time = chrono::microseconds(0);
    auto face_time = chrono::microseconds(0);
    auto non_frourier_time = chrono::microseconds(0);
    auto set_bound_time = chrono::microseconds(0);
    auto macro_bound_time = chrono::microseconds(0);
    auto macro_iter_time = chrono::microseconds(0);
    auto trasfer1_time = chrono::microseconds(0);

    _get_CellMatrix_larger();
    if (use_TDTR == 1) {
        ofstream outputT("TTG.dat");
        ofstream outputTemptopave("TDTR.dat");
        for (int nt = 0; nt < Num_Max_Iter; ++nt) {
            total_iter_time = chrono::microseconds(0);
            get_gradient_time = chrono::microseconds(0);
            get_Re_time = chrono::microseconds(0);
            solver1_time = chrono::microseconds(0);
            Boundary_time = chrono::microseconds(0);
            set_vertex_time = chrono::microseconds(0);
            face_time = chrono::microseconds(0);
            non_frourier_time = chrono::microseconds(0);
            set_bound_time = chrono::microseconds(0);
            macro_iter_time = chrono::microseconds(0);
            trasfer1_time = chrono::microseconds(0);

            auto total_iter_start = chrono::high_resolution_clock::now();
            copy();
            for (int inf_local = 0; inf_local < numDirectionLocal; inf_local++) {
                for (int iband_local = 0; iband_local < numBandLocal; ++iband_local) {
                    auto get_gradient_start = chrono::high_resolution_clock::now();
                    _get_gradient_larger(0, iband_local, inf_local);
                    auto get_gradient_end = chrono::high_resolution_clock::now();
                    get_gradient_time += chrono::duration_cast<chrono::microseconds>(
                            get_gradient_end - get_gradient_start);

                    auto get_Re_start = chrono::high_resolution_clock::now();
                    _get_explicit_Re(nt, 2, 0, iband_local, inf_local, deltaT);
                    auto get_Re_end = chrono::high_resolution_clock::now();
                    get_Re_time += chrono::duration_cast<chrono::microseconds>(
                            get_Re_end - get_Re_start);

                    auto solve_start = chrono::high_resolution_clock::now();

                    for (int icell = 0; icell < numCell; ++icell) {
                        // if(Re[icell]!=0)
                        // cout<<Re[icell]<<endl;
                        int inf = ((inf_local) * numProc + worldRank) % numDirection;
                        int iband =
                                iband_local * (ceil(double(numProc) / double(numDirection))) +
                                worldRank / numDirection;
                        energyDensity[iband_local][inf_local][icell] =
                                energyDensity[iband_local][inf_local][icell] *
                                (1 - deltaT / relaxationTime[matter[icell]][iband][inf]) -
                                deltaT * Re[icell];
                    }
                    auto solve_end = chrono::high_resolution_clock::now();
                    solver1_time += chrono::duration_cast<chrono::microseconds>(
                            solve_end - solve_start);

                    auto boundary_start = chrono::high_resolution_clock::now();
                    _get_bound_ee(iband_local, inf_local);
                    auto boundary_end = chrono::high_resolution_clock::now();
                    Boundary_time += chrono::duration_cast<chrono::microseconds>(
                            boundary_end - boundary_start);

                    _recover_temperature(iband_local, inf_local);
                    _get_total_energy(iband_local, inf_local);
                    _get_heat_flux(iband_local, inf_local);

                    MPI_Allreduce(totalEnergyLocal, totalEnergy, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(temperatureLocal, temperature, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(heatFluxXLocal, heatFluxXGlobal, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(heatFluxYLocal, heatFluxYGlobal, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(heatFluxZLocal, heatFluxZGlobal, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                }
            }

            auto set_bound_start = chrono::high_resolution_clock::now();
            _set_bound_ee_1();
            auto set_bound_end = chrono::high_resolution_clock::now();
            set_bound_time += chrono::duration_cast<chrono::microseconds>(
                    set_bound_end - set_bound_start);

            outputT << (nt + 1) * deltaT << " " << temperature[0] << endl;
            // memorize TDTR temp
            double Temptop_ave = 0;
            double rr = 0;
            double RR = xy_r * 2;
            double savve = 0;
            double heatratio = 0;
            for (int icell = 0; icell < numCell; icell++) {
                rr = pow(elementCenterX[icell] - 0, 2) +
                     pow(elementCenterY[icell] - 0, 2);
                if (rr < pow(RR, 2)) {
                    for (int jface = 0; jface < elementFaceSize[icell]; jface++) {
                        if (elementFaceCenterZ[jface + icell * 6] == 0) {
                            heatratio = exp(-2 * rr / pow(RR, 2));
                            savve = savve + heatratio;
                            Temptop_ave += temperature[icell] * heatratio;
                            break;
                        }
                    }
                }
            }
            Temptop_ave = Temptop_ave / savve;
            outputTemptopave << nt + 1 << "   " << Temptop_ave << endl;

            auto total_iter_end = chrono::high_resolution_clock::now();
            total_iter_time += chrono::duration_cast<chrono::microseconds>(
                    total_iter_end - total_iter_start);
            MPI_Barrier(MPI_COMM_WORLD);

            if (_get_magin_check_error(nt, error_temp_limit, error_flux_limit)) {
                nt = Num_Max_Iter;
                if (worldRank == 0)
                    _print_out();
            }
            if (errorIncreaseTime >= 10000) {
                nt = Num_Max_Iter;
                if (worldRank == 0)
                    _print_out();
                if (worldRank == 0)
                    cout << "error increases for 10 times, maybe the solution can not "
                            "converge. Try \"-1\" in limiter"
                         << endl;
                MPI_Barrier(MPI_COMM_WORLD);
            }
            if (nt % 1 == 0) {
                if (worldRank == 0)
                    _print_out();
            }

            /*if (worldRank == 0) {

                cout << "  Time taken by inner loop: " << 1.0 *
            total_iter_time.count() / 1000 << " milliseconds"
                     << endl;
                cout << "  Time taken by gradient 1: " << 1.0 *
            get_gradient_time.count() / 1000 << " milliseconds"
                     << endl;
                cout << "  Time taken by BTE Re: " << 1.0 * get_Re_time.count() / 1000
            << " milliseconds" << endl; cout << "  Time taken by BTE solver: " << 1.0
            * solver1_time.count() / 1000 << " milliseconds" << endl; cout << "  Time
            taken by Boundary: " << 1.0 * Boundary_time.count() / 1000 << "
            milliseconds"
                     << " " << 1.0 * set_bound_time.count() / 1000 << " milliseconds"
            << endl; cout << "  Time taken by set_vertex: " << 1.0 *
            set_vertex_time.count() / 1000 << " milliseconds"
                     << endl;
                cout << "  Time taken by face " << 1.0 * face_time.count() / 1000 << "
            milliseconds" << endl; cout << "  Time taken by non_fourier: " << 1.0 *
            non_frourier_time.count() / 1000 << " milliseconds"
                     << endl;
                cout << "  Time taken by macro: " << 1.0 * macro_iter_time.count() /
            1000 << " milliseconds" << endl; cout << "  Time taken by transfer: "
            << 1.0 * trasfer1_time.count() / 1000 << " milliseconds" << endl; cout <<
            "----------------------------------------------------------------------------------"
            << endl;
            }*/

            if (worldRank == 0) {
                for (int j = 1; j < numProc; ++j) {
                    MPI_Send(&nt, 1, MPI_INT, j, 10, MPI_COMM_WORLD);
                }

            } else {
                MPI_Status status;
                MPI_Recv(&nt, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
            }
            // cout<<nt<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
        }
        outputT.close();
        outputTemptopave.close();
    }
    if (use_TDTR == 0) {
        ofstream outputT("TTG.dat");
        std::cout << "IMPT::numCell " << numCell << std::endl;
        int blockCnt = 1;
        int threadCnt = numCell;
        while (threadCnt % 2 == 0 && blockCnt < threadCnt) {
            threadCnt /= 2;
            blockCnt *= 2;
        }
        for (int nt = 0; nt < Num_Max_Iter; ++nt) {
            total_iter_time = chrono::microseconds(0);
            get_gradient_time = chrono::microseconds(0);
            get_Re_time = chrono::microseconds(0);
            solver1_time = chrono::microseconds(0);
            Boundary_time = chrono::microseconds(0);
            set_vertex_time = chrono::microseconds(0);
            face_time = chrono::microseconds(0);
            non_frourier_time = chrono::microseconds(0);
            set_bound_time = chrono::microseconds(0);
            macro_iter_time = chrono::microseconds(0);
            trasfer1_time = chrono::microseconds(0);

            auto total_iter_start = chrono::high_resolution_clock::now();
            copy();

#ifdef USE_GPU
            // migrate elementFaceCenter{X,Y,Z} to GPU
            double *d_elementFaceCenterX, *d_elementFaceCenterY, *d_elementFaceCenterZ;
            MIGRATE_TO_DEVICE_1D(d_elementFaceCenterX, elementFaceCenterX, numCell * 6, double);
            MIGRATE_TO_DEVICE_1D(d_elementFaceCenterY, elementFaceCenterY, numCell * 6, double);
            MIGRATE_TO_DEVICE_1D(d_elementFaceCenterZ, elementFaceCenterZ, numCell * 6, double);

            // migrate elementCenter
            double *d_elementCenterX, *d_elementCenterY, *d_elementCenterZ;
            MIGRATE_TO_DEVICE_1D(d_elementCenterX, elementCenterX, numCell, double);
            MIGRATE_TO_DEVICE_1D(d_elementCenterY, elementCenterY, numCell, double);
            MIGRATE_TO_DEVICE_1D(d_elementCenterZ, elementCenterZ, numCell, double);

            // migrate elementFaceNorm
            double *d_elementFaceNormX, *d_elementFaceNormY, *d_elementFaceNormZ;
            MIGRATE_TO_DEVICE_1D(d_elementFaceNormX, elementFaceNormX, numCell * 6, double);
            MIGRATE_TO_DEVICE_1D(d_elementFaceNormY, elementFaceNormY, numCell * 6, double);
            MIGRATE_TO_DEVICE_1D(d_elementFaceNormZ, elementFaceNormZ, numCell * 6, double);

            // migrate boundaryCell and boundaryFace
            int *d_boundaryCell, *d_boundaryFace;
            auto *h_boundaryCell = new int[numBound * 2];
            auto *h_boundaryFace = new int[numBound * 2];

            for (int ib = 0; ib < numBound; ++ib) {
                for (int icell = 0; icell < 2; ++icell) {
                    h_boundaryCell[ib * 2 + icell] = boundaryCell[ib][icell];
                    h_boundaryFace[ib * 2 + icell] = boundaryFace[ib][icell];
                }
            }

            MIGRATE_TO_DEVICE_1D(d_boundaryCell, h_boundaryCell, numBound * 2, int);
            MIGRATE_TO_DEVICE_1D(d_boundaryFace, h_boundaryFace, numBound * 2, int);

            // migrate gradient{X,Y,Z} to GPU
            double *d_gradientX, *d_gradientY, *d_gradientZ;
            // d_gradient{X,Y,Z} will be set to 0 later, so no need to migrate
            cudaMalloc(&d_gradientX, numCell * sizeof(double));
            cudaMalloc(&d_gradientY, numCell * sizeof(double));
            cudaMalloc(&d_gradientZ, numCell * sizeof(double));

            // migrate elementNeighborList and size to GPU
            // vector<vector<int>> to double[numCell][numCell]
            auto *h_elementNeighborList = new int[numCell * numCell];
            auto *h_elementNeighborListSize = new int[numCell];
            for (int i = 0; i < numCell; ++i) {
                h_elementNeighborListSize[i] = elementNeighborList[i].size();
                for (int j = 0; j < elementNeighborList[i].size(); ++j) {
                    h_elementNeighborList[i * numCell + j] = elementNeighborList[i][j];
                }
            }
            int *d_elementNeighborList;
            int *d_elementNeighborListSize;
            MIGRATE_TO_DEVICE_1D(d_elementNeighborList, h_elementNeighborList, numCell * numCell, int);
            MIGRATE_TO_DEVICE_1D(d_elementNeighborListSize, h_elementNeighborListSize, numCell, int);

            // migrate CellMatrix to GPU
            auto h_CellMatrix = new double[numCell * numCell * 3];
            for (int i = 0; i < numCell; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int m = 0; m < numCell; ++m) {
                        h_CellMatrix[i * 3 * numCell + j * numCell + m] = CellMatrix[i][j][m];
                    }
                }
            }
            double *d_cellMatrix;
            MIGRATE_TO_DEVICE_1D(d_cellMatrix, h_CellMatrix, numCell * numCell * 3, double);

            // migrate Re
            double *d_Re;
            cudaMalloc(&d_Re, numCell * sizeof(double));

            // migrate elementFaceBound and elementVolume
            int *d_elementFaceBound;
            double *d_elementVolume;
            MIGRATE_TO_DEVICE_1D(d_elementFaceBound, elementFaceBound, numCell * 6, int);
            MIGRATE_TO_DEVICE_1D(d_elementVolume, elementVolume, numCell, double);

            // migrate elementFaceSize
            int *d_elementFaceSize;
            MIGRATE_TO_DEVICE_1D(d_elementFaceSize, elementFaceSize, numCell, int);

            // migrate capacityBulk
            double *d_capacityBulk;
            auto *h_capacityBulk = new double[numCell];
            for (int ie = 0; ie < numCell; ++ie) {
                h_capacityBulk[ie] = capacityBulk[matter[ie]];
            }
            MIGRATE_TO_DEVICE_1D(d_capacityBulk, h_capacityBulk, numCell, double);

            int *d_boundaryType, *d_elementFaceNeighbor;
            MIGRATE_TO_DEVICE_1D(d_boundaryType, boundaryType, numBound, int);
            MIGRATE_TO_DEVICE_1D(d_elementFaceNeighbor, elementFaceNeighobr, numCell * 6, int);

            double *d_ebound;
            MIGRATE_TO_DEVICE_1D(d_ebound, ebound, numBand * numDirection * numBound * 2, double);

            double *d_elementHeatSource;
            MIGRATE_TO_DEVICE_1D(d_elementHeatSource, elementHeatSource, numCell, double);

            double *d_totalEnergyLocal;
            MIGRATE_TO_DEVICE_1D(d_totalEnergyLocal, totalEnergyLocal, numCell, double);

            double *d_temperatureLocal;
            MIGRATE_TO_DEVICE_1D(d_temperatureLocal, temperatureLocal, numCell, double);

            // migrate limit
            auto initialValueForLimit = new double[numCell];
            for (int i = 0; i < numCell; ++i) {
                initialValueForLimit[i] = 1;
            }
            double *d_limit;
            cudaMalloc(&d_limit, numCell * sizeof(double));

            // migrate heatFluxLocal
            double *d_heatFluxXLocal, *d_heatFluxYLocal, *d_heatFluxZLocal;
            MIGRATE_TO_DEVICE_1D(d_heatFluxXLocal, heatFluxXLocal, numCell, double);
            MIGRATE_TO_DEVICE_1D(d_heatFluxYLocal, heatFluxYLocal, numCell, double);
            MIGRATE_TO_DEVICE_1D(d_heatFluxZLocal, heatFluxZLocal, numCell, double);

            double *d_temperatureOld;
            double *d_elementFaceArea;
            MIGRATE_TO_DEVICE_1D(d_temperatureOld, temperatureOld, numCell, double);
            MIGRATE_TO_DEVICE_1D(d_elementFaceArea, elementFaceArea, numCell * 6, double);

            double *d_eboundLocal;
            cudaMalloc(&d_eboundLocal, numBand * numDirection * numBound * 2 * sizeof(double));

            // migrate energyDensity, groupVelocity{X,Y,Z}
            auto d_energyDensityArray = new double *[numDirectionLocal * numBandLocal];
            auto d_groupVelocityXArray = new double *[numDirectionLocal * numBandLocal];
            auto d_groupVelocityYArray = new double *[numDirectionLocal * numBandLocal];
            auto d_groupVelocityZArray = new double *[numDirectionLocal * numBandLocal];

            auto h_groupVelocityX = new double[numCell];
            auto h_groupVelocityY = new double[numCell];
            auto h_groupVelocityZ = new double[numCell];

            auto *h_heatCapacity = new double[numCell];
            auto *h_heatRatio = new double[numCell];
            auto *h_relaxationTime = new double[numCell];
            auto d_heatCapacityArray = new double *[numDirectionLocal * numBandLocal];
            auto d_heatRatioArray = new double *[numDirectionLocal * numBandLocal];
            auto d_relaxationTimeArray = new double *[numDirectionLocal * numBandLocal];

            auto h_latticeRatio = new double[numCell];
            auto h_modeWeight = new double[numCell];
            auto d_latticeRatioArray = new double *[numDirectionLocal * numBandLocal];
            auto d_modeWeightArray = new double *[numDirectionLocal * numBandLocal];

            for (int inf_local = 0; inf_local < numDirectionLocal; inf_local++) {
                for (int iband_local = 0; iband_local < numBandLocal; ++iband_local) {
                    const int inf = ((inf_local) * numProc + worldRank) % numDirection;
                    const int iband = iband_local * (ceil(double(numProc) / double(numDirection))) +
                                      worldRank / numDirection;

                    // energyDensity
                    MIGRATE_TO_DEVICE_1D(d_energyDensityArray[inf_local * numBandLocal + iband_local],
                                         energyDensity[iband_local][inf_local], numCell, double);

                    // groupVelocity
                    for (int i = 0; i < numCell; ++i) {
                        h_groupVelocityX[i] = groupVelocityX[matter[i]][iband][inf];
                        h_groupVelocityY[i] = groupVelocityY[matter[i]][iband][inf];
                        h_groupVelocityZ[i] = groupVelocityZ[matter[i]][iband][inf];
                    }
                    MIGRATE_TO_DEVICE_1D(d_groupVelocityXArray[inf_local * numBandLocal + iband_local],
                                         h_groupVelocityX, numCell, double);
                    MIGRATE_TO_DEVICE_1D(d_groupVelocityYArray[inf_local * numBandLocal + iband_local],
                                         h_groupVelocityY, numCell, double);
                    MIGRATE_TO_DEVICE_1D(d_groupVelocityZArray[inf_local * numBandLocal + iband_local],
                                         h_groupVelocityZ, numCell, double);

                    // heatCapactiy, heatRatio, relaxationTime
                    for (int i = 0; i < numCell; ++i) {
                        h_heatCapacity[i] = heatCapacity[matter[i]][iband][inf];
                        h_heatRatio[i] = heatRatio[matter[i]][iband][inf];
                        h_relaxationTime[i] = relaxationTime[matter[i]][iband][inf];
                    }
                    MIGRATE_TO_DEVICE_1D(d_heatRatioArray[inf_local * numBandLocal + iband_local], h_heatRatio, numCell,
                                         double);
                    MIGRATE_TO_DEVICE_1D(d_heatCapacityArray[inf_local * numBandLocal + iband_local], h_heatCapacity,
                                         numCell, double);
                    MIGRATE_TO_DEVICE_1D(d_relaxationTimeArray[inf_local * numBandLocal + iband_local],
                                         h_relaxationTime, numCell, double);

                    // modeWeight, latticeRatio
                    for (int ie = 0; ie < numCell; ++ie) {
                        h_latticeRatio[ie] = latticeRatio[matter[ie]][iband][inf];
                        h_modeWeight[ie] = modeWeight[matter[ie]][iband][inf];
                    }
                    MIGRATE_TO_DEVICE_1D(d_latticeRatioArray[inf_local * numBandLocal + iband_local], h_latticeRatio,
                                         numCell, double);
                    MIGRATE_TO_DEVICE_1D(d_modeWeightArray[inf_local * numBandLocal + iband_local], h_modeWeight,
                                         numCell, double);
                }
            }
#endif

            for (int inf_local = 0; inf_local < numDirectionLocal; inf_local++) {
                for (int iband_local = 0; iband_local < numBandLocal; ++iband_local) {
#ifdef USE_GPU
                    const int inf = ((inf_local) * numProc + worldRank) % numDirection;
                    const int iband = iband_local * (ceil(double(numProc) / double(numDirection))) +
                                      worldRank / numDirection;

                    auto d_energyDensity = d_energyDensityArray[inf_local * numBandLocal + iband_local];

                    auto d_groupVelocityX = d_groupVelocityXArray[inf_local * numBandLocal + iband_local];
                    auto d_groupVelocityY = d_groupVelocityYArray[inf_local * numBandLocal + iband_local];
                    auto d_groupVelocityZ = d_groupVelocityZArray[inf_local * numBandLocal + iband_local];

                    cudaMemcpy(d_limit, initialValueForLimit, numCell * sizeof(double), cudaMemcpyHostToDevice);

                    /* ================================================================ */
                    // migration of _get_explicit_Re
                    cudaMemset(d_Re, 0, numCell * sizeof(double));

                    auto d_heatRatio = d_heatRatioArray[inf_local * numBandLocal + iband_local];
                    auto d_heatCapacity = d_heatCapacityArray[inf_local * numBandLocal + iband_local];
                    auto d_relaxationTime = d_relaxationTimeArray[inf_local * numBandLocal + iband_local];

                    auto d_latticeRatio = d_latticeRatioArray[inf_local * numBandLocal + iband_local];
                    auto d_modeWeight = d_modeWeightArray[inf_local * numBandLocal + iband_local];
#endif

#ifndef USE_GPU
                    auto get_gradient_start = chrono::high_resolution_clock::now();
                    _get_gradient_larger(0, iband_local, inf_local);
                    auto get_gradient_end = chrono::high_resolution_clock::now();
                    get_gradient_time += chrono::duration_cast<chrono::microseconds>(
                            get_gradient_end - get_gradient_start);
#else
                    // _get_gradient_larger(0, iband_local, inf_local);
                    // clear gradient{X,Y,Z}
                    cudaMemset(d_gradientX, 0, numCell * sizeof(double));
                    cudaMemset(d_gradientY, 0, numCell * sizeof(double));
                    cudaMemset(d_gradientZ, 0, numCell * sizeof(double));
                    if (dimension == 1) {
                        calcGetGradientLargerDimension1<<<blockCnt, threadCnt>>>(threadCnt, d_elementFaceBound, d_energyDensity,
                                                                        d_elementVolume,
                                                                        d_gradientX);
                    } else if (dimension == 2) {
                        calcGetGradientLargerDimension2<<<blockCnt, threadCnt>>>(threadCnt, L_x, numCell, d_gradientX, d_gradientY,
                                                                        d_gradientZ,
                                                                        d_elementNeighborList,
                                                                        d_elementNeighborListSize, d_energyDensity,
                                                                        d_cellMatrix);
                    } else if (dimension == 3) {
                        calcGetGradientLargerDimension3<<<blockCnt, threadCnt>>>(threadCnt, L_x, numCell, d_gradientX, d_gradientY,
                                                                        d_gradientZ,
                                                                        d_elementNeighborList,
                                                                        d_elementNeighborListSize,
                                                                        d_energyDensity, d_cellMatrix);
                    }
                    // here Use_limiter is 0

#endif

#ifndef USE_GPU
                    auto get_Re_start = chrono::high_resolution_clock::now();
                    _get_explicit_Re(nt, 2, 0, iband_local, inf_local, deltaT);
                    auto get_Re_end = chrono::high_resolution_clock::now();
                    get_Re_time += chrono::duration_cast<chrono::microseconds>(
                            get_Re_end - get_Re_start);
#else
                    const auto deltaTime = deltaT;
                    const auto itime = nt;
                    calcGetExplicitRe<<<blockCnt, threadCnt>>>(threadCnt, use_TDTR, deltaTime, d_elementFaceSize, repetition_frequency,
                                                      modulation_frequency, pulse_time, itime, d_Re,
                                                      d_groupVelocityX, d_groupVelocityY, d_groupVelocityZ,
                                                      d_elementFaceNormX, d_elementFaceNormY, d_elementFaceNormZ,
                                                      d_elementFaceArea, d_elementVolume,
                                                      d_elementFaceCenterX, d_elementFaceCenterY, d_elementFaceCenterZ,
                                                      d_elementCenterX, d_elementCenterY, d_elementCenterZ,
                                                      d_energyDensity, d_gradientX, d_gradientY, d_gradientZ,
                                                      d_limit, d_elementFaceBound, d_elementFaceNeighbor,
                                                      d_elementHeatSource, d_heatRatio, d_heatCapacity,
                                                      d_relaxationTime, d_temperatureOld);

                    int magic = iband * numDirection * numBound * 2 + inf * numBound * 2;
                    calcGetExplicitReRest<<<1, numBound * 2>>>(magic, d_Re, d_ebound, d_boundaryCell, d_boundaryFace,
                                                               d_groupVelocityX, d_groupVelocityY, d_groupVelocityZ,
                                                               d_elementFaceNormX, d_elementFaceNormY,
                                                               d_elementFaceNormZ, d_elementFaceArea, d_elementVolume);

#endif

                    auto solve_start = chrono::high_resolution_clock::now();

#ifndef USE_GPU
                    for (int icell = 0; icell < numCell; ++icell) {
                        // if(Re[icell]!=0)
                        // cout<<Re[icell]<<endl;
                        int inf = ((inf_local) * numProc + worldRank) % numDirection;
                        int iband =
                                iband_local * (ceil(double(numProc) / double(numDirection))) +
                                worldRank / numDirection;
                        energyDensity[iband_local][inf_local][icell] =
                                energyDensity[iband_local][inf_local][icell] *
                                (1 - deltaT / relaxationTime[matter[icell]][iband][inf]) -
                                deltaT * Re[icell];
                    }
#else
                    calcEnergyDensity<<<blockCnt, threadCnt>>>(threadCnt, deltaT, d_energyDensity, d_Re, d_relaxationTime);
#endif

                    auto solve_end = chrono::high_resolution_clock::now();
                    solver1_time += chrono::duration_cast<chrono::microseconds>(
                            solve_end - solve_start);

#ifndef USE_GPU
                    auto boundary_start = chrono::high_resolution_clock::now();
                    _get_bound_ee(iband_local, inf_local);
                    auto boundary_end = chrono::high_resolution_clock::now();
                    Boundary_time += chrono::duration_cast<chrono::microseconds>(
                            boundary_end - boundary_start);
#else
                    // TODO: change me, is eBoundLocal useful in single-process mode?
                    calcGetBoundEE<<<1, numBound * 2>>>(iband, iband_local, inf, numBound, numDirection,
                                                        d_boundaryCell, d_boundaryFace, d_groupVelocityX,
                                                        d_groupVelocityY, d_groupVelocityZ, d_elementFaceNormX,
                                                        d_elementFaceNormY, d_elementFaceNormZ, d_elementFaceCenterX,
                                                        d_elementFaceCenterY, d_elementFaceCenterZ, d_elementCenterX,
                                                        d_elementCenterY, d_elementCenterZ, d_energyDensity,
                                                        d_limit, d_gradientX, d_gradientY, d_gradientZ,
                                                        d_eboundLocal, d_ebound);
                    cudaMemcpy(
                            (d_ebound + numBound * 2 * (inf - worldRank % numDirection)) +
                            numDirection * numBound * 2 * (iband - worldRank / numDirection),
                            d_eboundLocal + iband_local * numBound * 2,
                            numBound * 2 * sizeof(double),
                            cudaMemcpyDeviceToDevice
                    );
#endif

#ifndef USE_GPU
                    _recover_temperature(iband_local, inf_local);
                    _get_total_energy(iband_local, inf_local);
                    _get_heat_flux(iband_local, inf_local);
#else
                    calcRecoverTemperature<<<blockCnt, threadCnt>>>(threadCnt, d_temperatureLocal, d_latticeRatio, d_energyDensity,
                                                           d_modeWeight,
                                                           d_heatCapacity);
                    calcGetTotalEnergy<<<blockCnt, threadCnt>>>(threadCnt, d_totalEnergyLocal, d_energyDensity, d_modeWeight,
                                                       d_capacityBulk);

                    calcGetHeatFlux<<<blockCnt, threadCnt>>>(threadCnt, d_heatFluxXLocal, d_groupVelocityX, d_modeWeight, d_energyDensity);
                    calcGetHeatFlux<<<blockCnt, threadCnt>>>(threadCnt, d_heatFluxYLocal, d_groupVelocityY, d_modeWeight, d_energyDensity);
                    calcGetHeatFlux<<<blockCnt, threadCnt>>>(threadCnt, d_heatFluxZLocal, d_groupVelocityZ, d_modeWeight, d_energyDensity);
#endif

#ifndef USE_GPU
                    MPI_Allreduce(totalEnergyLocal, totalEnergy, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(temperatureLocal, temperature, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(heatFluxXLocal, heatFluxXGlobal, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(heatFluxYLocal, heatFluxYGlobal, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(heatFluxZLocal, heatFluxZGlobal, numCell, MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
#else
                    // TODO: migrate to NCCL
//                    memcpy(totalEnergy, totalEnergyLocal, numCell * sizeof(double));
//                    memcpy(temperature, temperatureLocal, numCell * sizeof(double));
//                    memcpy(heatFluxXGlobal, heatFluxXLocal, numCell * sizeof(double));
//                    memcpy(heatFluxYGlobal, heatFluxYLocal, numCell * sizeof(double));
//                    memcpy(heatFluxZGlobal, heatFluxZLocal, numCell * sizeof(double));
#endif
                }
            }

#ifdef USE_GPU
            cudaFree(d_elementFaceCenterX);
            cudaFree(d_elementFaceCenterY);
            cudaFree(d_elementFaceCenterZ);

            cudaFree(d_elementCenterX);
            cudaFree(d_elementCenterY);
            cudaFree(d_elementCenterZ);

            cudaFree(d_elementFaceNormX);
            cudaFree(d_elementFaceNormY);
            cudaFree(d_elementFaceNormZ);

            delete[] h_boundaryCell;
            delete[] h_boundaryFace;
            cudaFree(d_boundaryCell);
            cudaFree(d_boundaryFace);

            MIGRATE_TO_HOST_1D(gradientX, d_gradientX, numCell, double);
            MIGRATE_TO_HOST_1D(gradientY, d_gradientY, numCell, double);
            MIGRATE_TO_HOST_1D(gradientZ, d_gradientZ, numCell, double);

            delete[] h_elementNeighborListSize;
            delete[] h_elementNeighborList;
            cudaFree(d_elementNeighborList);
            cudaFree(d_elementNeighborListSize);

            delete[] h_CellMatrix;
            cudaFree(d_cellMatrix);

            MIGRATE_TO_HOST_1D(Re, d_Re, numCell, double);

            cudaFree(d_elementFaceBound);
            cudaFree(d_elementVolume);

            cudaFree(d_elementFaceSize);

            delete[] h_capacityBulk;
            cudaFree(d_capacityBulk);

            cudaFree(d_boundaryType);
            cudaFree(d_elementFaceNeighbor);

            MIGRATE_TO_HOST_1D(ebound, d_ebound, numBand * numDirection * numBound * 2, double);

            cudaFree(d_elementHeatSource);

            MIGRATE_TO_HOST_1D(totalEnergyLocal, d_totalEnergyLocal, numCell, double);
            MIGRATE_TO_HOST_1D(temperatureLocal, d_temperatureLocal, numCell, double);
            memcpy(totalEnergy, totalEnergyLocal, numCell * sizeof(double));
            memcpy(temperature, temperatureLocal, numCell * sizeof(double));

            MIGRATE_TO_HOST_1D(limit, d_limit, numCell, double);

            MIGRATE_TO_HOST_1D(heatFluxXLocal, d_heatFluxXLocal, numCell, double);
            MIGRATE_TO_HOST_1D(heatFluxYLocal, d_heatFluxYLocal, numCell, double);
            MIGRATE_TO_HOST_1D(heatFluxZLocal, d_heatFluxZLocal, numCell, double);
            memcpy(heatFluxXGlobal, heatFluxXLocal, numCell * sizeof(double));
            memcpy(heatFluxYGlobal, heatFluxYLocal, numCell * sizeof(double));
            memcpy(heatFluxZGlobal, heatFluxZLocal, numCell * sizeof(double));

            cudaFree(d_temperatureOld);
            cudaFree(d_elementFaceArea);

            MIGRATE_TO_HOST_1D(eboundLocal, d_eboundLocal, numBand * numDirection * numBound * 2, double);

            for (int inf_local = 0; inf_local < numDirectionLocal; inf_local++) {
                for (int iband_local = 0; iband_local < numBandLocal; ++iband_local) {
                    // energyDensity
                    MIGRATE_TO_HOST_1D(energyDensity[iband_local][inf_local],
                                       d_energyDensityArray[inf_local * numBandLocal + iband_local], numCell, double);
                    // groupVelocity
                    cudaFree(d_groupVelocityXArray[inf_local * numBandLocal + iband_local]);
                    cudaFree(d_groupVelocityYArray[inf_local * numBandLocal + iband_local]);
                    cudaFree(d_groupVelocityZArray[inf_local * numBandLocal + iband_local]);
                    // heatRatio, heatCapacity and relaxationTime
                    cudaFree(d_heatRatioArray[inf_local * numBandLocal + iband_local]);
                    cudaFree(d_heatCapacityArray[inf_local * numBandLocal + iband_local]);
                    cudaFree(d_relaxationTimeArray[inf_local * numBandLocal + iband_local]);
                    // latticeRatio, modeWeight
                    cudaFree(d_latticeRatioArray);
                    cudaFree(d_modeWeightArray);
                }
            }
            delete[] d_energyDensityArray;

            delete[] d_groupVelocityXArray;
            delete[] d_groupVelocityYArray;
            delete[] d_groupVelocityZArray;

            delete[] h_groupVelocityX;
            delete[] h_groupVelocityY;
            delete[] h_groupVelocityZ;

            delete[] h_heatRatio;
            delete[] h_heatCapacity;
            delete[] h_relaxationTime;
            delete[] d_heatRatioArray;
            delete[] d_heatCapacityArray;
            delete[] d_relaxationTimeArray;

            delete[] h_latticeRatio;
            delete[] h_modeWeight;
            delete[] d_latticeRatioArray;
#endif

            auto set_bound_start = chrono::high_resolution_clock::now();
            _set_bound_ee_1();
            auto set_bound_end = chrono::high_resolution_clock::now();
            set_bound_time += chrono::duration_cast<chrono::microseconds>(
                    set_bound_end - set_bound_start);

            outputT << (nt + 1) * deltaT << " " << temperature[0] << endl;
            auto total_iter_end = chrono::high_resolution_clock::now();
            total_iter_time += chrono::duration_cast<chrono::microseconds>(
                    total_iter_end - total_iter_start);
            MPI_Barrier(MPI_COMM_WORLD);

            if (_get_magin_check_error(nt, error_temp_limit, error_flux_limit)) {
                nt = Num_Max_Iter;
                if (worldRank == 0)
                    _print_out();
            }
            if (errorIncreaseTime >= 10000) {
                nt = Num_Max_Iter;
                if (worldRank == 0)
                    _print_out();
                if (worldRank == 0)
                    cout << "error increases for 10 times, maybe the solution can not "
                            "converge. Try \"-1\" in limiter"
                         << endl;
                MPI_Barrier(MPI_COMM_WORLD);
            }
            if (nt % 1 == 0) {
                if (worldRank == 0)
                    _print_out();
            }

            /*if (worldRank == 0) {

                cout << "  Time taken by inner loop: " << 1.0 *
            total_iter_time.count() / 1000 << " milliseconds"
                     << endl;
                cout << "  Time taken by gradient 1: " << 1.0 *
            get_gradient_time.count() / 1000 << " milliseconds"
                     << endl;
                cout << "  Time taken by BTE Re: " << 1.0 * get_Re_time.count() / 1000
            << " milliseconds" << endl; cout << "  Time taken by BTE solver: " << 1.0
            * solver1_time.count() / 1000 << " milliseconds" << endl; cout << "  Time
            taken by Boundary: " << 1.0 * Boundary_time.count() / 1000 << "
            milliseconds"
                     << " " << 1.0 * set_bound_time.count() / 1000 << " milliseconds"
            << endl; cout << "  Time taken by set_vertex: " << 1.0 *
            set_vertex_time.count() / 1000 << " milliseconds"
                     << endl;
                cout << "  Time taken by face " << 1.0 * face_time.count() / 1000 << "
            milliseconds" << endl; cout << "  Time taken by non_fourier: " << 1.0 *
            non_frourier_time.count() / 1000 << " milliseconds"
                     << endl;
                cout << "  Time taken by macro: " << 1.0 * macro_iter_time.count() /
            1000 << " milliseconds" << endl; cout << "  Time taken by transfer: "
            << 1.0 * trasfer1_time.count() / 1000 << " milliseconds" << endl; cout <<
            "----------------------------------------------------------------------------------"
            << endl;
            }*/

            if (worldRank == 0) {
                for (int j = 1; j < numProc; ++j) {
                    MPI_Send(&nt, 1, MPI_INT, j, 10, MPI_COMM_WORLD);
                }

            } else {
                MPI_Status status;
                MPI_Recv(&nt, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
            }
            // cout<<nt<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
        }
        outputT.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //_get_bound_temp();
    //_get_bound_flux();

#ifdef USE_TIME
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // cout << "end_iter1" << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    if (worldRank == 0)
        _print_out();
    if (worldRank == 0)

#ifdef USE_TIME
        cout << "Time taken by iteration: " << duration.count() * 0.001
             << " milliseconds" << endl;
#endif

        _delete_cell_matrix();
}

void Transient::solve_first_order(int Use_Backup, int Use_Limiter,
                                  double error_temp_limit,
                                  double error_flux_limit, double deltaT,
                                  double totalT) {}
