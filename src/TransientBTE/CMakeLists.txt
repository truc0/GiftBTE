if (USE_GPU)
    set_source_files_properties(transient.h PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(transient.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(TransientSolver.cpp PROPERTIES LANGUAGE CUDA)
    add_library(TransientBTE transient.cpp TransientSolver.cpp cuda_kernels.cu)
    # set_property(TARGET TransientBTE PROPERTY CUDA_ARCHITECTURES "native")
    # set_property(TARGET TransientBTE PROPERTY CUDA_SELECT_NVCC_ARCH_FLAGS "AUTO")
else()
    add_library(TransientBTE   transient.cpp TransientSolver.cpp)
endif()

target_include_directories(TransientBTE PUBLIC "/home/yuehu/Downloads" "${StaticBTESolver_SOURCE_DIR}/includes/StaticBTESolver"
        PRIVATE "${PETSC_INCLUDES}" "${HYPRE_INC_PATH}")

target_link_libraries(TransientBTE PUBLIC BTEBand BTEBoundaryCondition BTEMesh BTEAngle BTEutility
        PRIVATE "${PETSC_LIBRARIES}" MPI::MPI_CXX "${HYPRE_LIB_PATH}")
