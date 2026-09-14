#include "simulation/gpu/ClothCudaJacobi.h"

namespace poseweaver
{
    bool cuda_simulation_available(std::string* reason)
    {
        if (reason)
            *reason = "PoseWeaver was built without a CUDA compiler";
        return false;
    }

    bool build_smplh_sequence_cuda(
        const CudaSmplhView&,
        const std::vector<Eigen::VectorXd>&,
        std::vector<Eigen::MatrixXd>&,
        std::string* error,
        std::atomic_bool*)
    {
        if (error)
            *error = "PoseWeaver was built without CUDA support";
        return false;
    }

    bool run_cloth_cuda_jacobi(
        const Eigen::MatrixXd&,
        const Eigen::MatrixXi&,
        const Eigen::MatrixXd&,
        const Eigen::MatrixXi&,
        const ClothXpbdParams&,
        ClothXpbdResult*,
        std::string* error,
        std::atomic_bool*,
        std::atomic_int*,
        ClothXpbdOnFrameFn,
        void*)
    {
        if (error)
            *error = "PoseWeaver was built without CUDA support";
        return false;
    }
}
