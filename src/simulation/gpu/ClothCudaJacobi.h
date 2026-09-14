#pragma once

#include "simulation/xpbd/ClothXpbd.h"

namespace poseweaver
{
    bool cuda_simulation_available(std::string* reason = nullptr);

    struct CudaSmplhView
    {
        int vertex_count = 0;
        int joint_count = 0;
        const double* v_shaped = nullptr;
        const double* joints = nullptr;
        const int* parents = nullptr;
        const double* posedirs = nullptr;
        const double* weights = nullptr;
    };

    bool build_smplh_sequence_cuda(
        const CudaSmplhView& model,
        const std::vector<Eigen::VectorXd>& poses,
        std::vector<Eigen::MatrixXd>& frames_out,
        std::string* error,
        std::atomic_bool* cancel = nullptr);

    bool run_cloth_cuda_jacobi(
        const Eigen::MatrixXd& cloth_V0,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const ClothXpbdParams& params,
        ClothXpbdResult* out,
        std::string* error,
        std::atomic_bool* cancel = nullptr,
        std::atomic_int* progress_steps_done = nullptr,
        ClothXpbdOnFrameFn on_frame = nullptr,
        void* on_frame_user = nullptr);
}
