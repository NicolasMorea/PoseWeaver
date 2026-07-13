#pragma once

#include "simulation/xpbd/ClothXpbd.h"

namespace poseweaver
{
    bool cuda_simulation_available(std::string* reason = nullptr);

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
