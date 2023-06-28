if (USE_GPU)
    add_library(TransientBTE transient_kernels.cu transient.cpp TransientSolver.cpp)
    set_source_files_properties(transient.cpp PROPERTIES LANGUAGE CUDA)
else()
    add_library(TransientBTE transient.cpp TransientSolver.cpp)
endif()
target_include_directories(TransientBTE PUBLIC "/home/yuehu/Downloads" "${StaticBTESolver_SOURCE_DIR}/includes/StaticBTESolver"
        PRIVATE "${PETSC_INCLUDES}" "${HYPRE_INC_PATH}")

target_link_libraries(TransientBTE PUBLIC BTEBand BTEBoundaryCondition BTEMesh BTEAngle BTEutility
        PRIVATE "${PETSC_LIBRARIES}" MPI::MPI_CXX "${HYPRE_LIB_PATH}")
