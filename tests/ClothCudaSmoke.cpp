#include "simulation/gpu/ClothCudaJacobi.h"
#include "smpl/Smplh.h"

#include <cmath>
#include <iostream>

int main()
{
    std::string status;
    if (!poseweaver::cuda_simulation_available(&status))
    {
        std::cerr << status << '\n';
        return 1;
    }

    poseweaver::SmplhModel smpl_model;
    smpl_model.v_template.resize(2, 3);
    smpl_model.v_template << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0;
    smpl_model.shapedirs.resize(6, 0);
    smpl_model.posedirs = poseweaver::MatrixXdrm::Zero(6, 9);
    smpl_model.posedirs(0, 0) = 0.25;
    smpl_model.J_regressor.resize(2, 2);
    smpl_model.J_regressor << 0.0, 0.0, 0.0, 0.5;
    smpl_model.weights.resize(2, 2);
    smpl_model.weights << 1.0, 0.0, 0.0, 1.0;
    smpl_model.kintree_table.resize(2, 2);
    smpl_model.kintree_table << 0, 1, 4294967295LL, 0;

    poseweaver::SmplhLbsCache smpl_cache;
    if (!poseweaver::smplh_prepare_lbs_cache(
            smpl_model, Eigen::VectorXd{}, smpl_cache, &status))
        return 2;
    const poseweaver::CudaSmplhView smpl{
        smpl_model.vertex_count(), smpl_model.joint_count(),
        smpl_cache.v_shaped_flat.data(), smpl_cache.joints.data(), smpl_cache.parents.data(),
        smpl_model.posedirs.data(), smpl_model.weights.data()
    };
    std::vector<Eigen::VectorXd> poses(3, Eigen::VectorXd::Zero(6));
    poses[1][2] = 1.5707963267948966;
    poses[2][5] = 1.5707963267948966;
    std::vector<Eigen::MatrixXd> smpl_frames;
    std::string error;
    if (!poseweaver::build_smplh_sequence_cuda(smpl, poses, smpl_frames, &error))
    {
        std::cerr << error << '\n';
        return 3;
    }
    if (smpl_frames.size() != 3
        || (smpl_frames[0].row(0) - Eigen::RowVector3d(1.0, 0.0, 0.0)).norm() > 1e-5
        || (smpl_frames[1].row(0) - Eigen::RowVector3d(0.0, 1.0, 0.0)).norm() > 1e-5
        || (smpl_frames[1].row(1) - Eigen::RowVector3d(-2.0, 0.0, 0.0)).norm() > 1e-5
        || (smpl_frames[2].row(0) - Eigen::RowVector3d(0.75, 0.0, 0.0)).norm() > 1e-5
        || (smpl_frames[2].row(1) - Eigen::RowVector3d(-1.0, 1.0, 0.0)).norm() > 1e-5)
        return 4;
    for (size_t frame = 0; frame < poses.size(); ++frame)
    {
        poseweaver::MatrixX3dRM cached_frame;
        poseweaver::MatrixX3dRM uncached_frame;
        poseweaver::SmplhParams params;
        params.pose_axis_angle = poses[frame];
        if (!poseweaver::smplh_evaluate_lbs_cached(
                smpl_model, smpl_cache, poses[frame], cached_frame, &error)
            || !poseweaver::smplh_evaluate_lbs(smpl_model, params, uncached_frame, &error)
            || (cached_frame - uncached_frame).cwiseAbs().maxCoeff() > 1e-12
            || (smpl_frames[frame] - cached_frame).cwiseAbs().maxCoeff() > 1e-5)
            return 5;
    }

    Eigen::MatrixXd cloth(4, 3);
    cloth <<
        -0.5, 0.005, -0.5,
         0.5, 0.005, -0.5,
         0.5, 0.005,  0.5,
        -0.5, 0.005,  0.5;
    Eigen::MatrixXi cloth_faces(2, 3);
    cloth_faces << 0, 1, 2, 0, 2, 3;

    Eigen::MatrixXd body(4, 3);
    body <<
        -2.0, 0.0, -2.0,
         2.0, 0.0, -2.0,
         2.0, 0.0,  2.0,
        -2.0, 0.0,  2.0;
    Eigen::MatrixXi body_faces(2, 3);
    body_faces << 0, 2, 1, 0, 3, 2;

    poseweaver::ClothXpbdParams params;
    params.dt = 1.0 / 60.0;
    params.duration_seconds = params.dt;
    params.solver_iterations = 8;
    params.substeps = 1;
    params.gravity_y = 0.0;
    params.bend_enabled = false;
    params.collision_enabled = true;
    params.collision_thickness = 0.01;
    params.max_frames = 2;

    poseweaver::ClothXpbdResult result;
    if (!poseweaver::run_cloth_cuda_jacobi(
            cloth, cloth_faces, body, body_faces, params, &result, &error))
    {
        std::cerr << error << '\n';
        return 4;
    }
    if (result.frames.size() != 2 || !result.frames.back().allFinite())
        return 5;
    if (result.frames.back().col(1).minCoeff() <= cloth.col(1).minCoeff())
        return 6;

    poseweaver::ClothXpbdParams stable_params = params;
    stable_params.duration_seconds = 10.0 * params.dt;
    stable_params.max_frames = 11;
    poseweaver::ClothXpbdResult stable_result;
    if (!poseweaver::run_cloth_cuda_jacobi(
            cloth, cloth_faces, body, body_faces, stable_params, &stable_result, &error))
    {
        std::cerr << error << '\n';
        return 5;
    }
    if (stable_result.frames.back().col(1).maxCoeff() > 0.02)
        return 6;

    Eigen::MatrixXi inverted_body_faces(2, 3);
    inverted_body_faces << 0, 1, 2, 0, 2, 3;
    poseweaver::ClothXpbdResult inverted_result;
    if (!poseweaver::run_cloth_cuda_jacobi(
            cloth, cloth_faces, body, inverted_body_faces, stable_params, &inverted_result, &error))
        return 7;
    if (inverted_result.frames.back().col(1).minCoeff() < 0.004
        || inverted_result.frames.back().col(1).maxCoeff() > 0.02)
        return 8;

    Eigen::MatrixXd distant_backface = cloth;
    distant_backface.col(1).setConstant(-1.0);
    poseweaver::ClothXpbdResult distant_result;
    if (!poseweaver::run_cloth_cuda_jacobi(
            distant_backface, cloth_faces, body, body_faces, params, &distant_result, &error))
    {
        std::cerr << error << '\n';
        return 9;
    }
    if ((distant_result.frames.back() - distant_backface).cwiseAbs().maxCoeff() > 1e-6)
        return 10;

    Eigen::MatrixXd enclosing_planes(8, 3);
    enclosing_planes <<
        -2.0, -2.0, -2.0,  2.0, -2.0, -2.0,  2.0, -2.0, 2.0, -2.0, -2.0, 2.0,
        -2.0,  2.0, -2.0,  2.0,  2.0, -2.0,  2.0,  2.0, 2.0, -2.0,  2.0, 2.0;
    Eigen::MatrixXi enclosing_faces(4, 3);
    enclosing_faces << 0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6;
    Eigen::MatrixXd centered_cloth = cloth;
    centered_cloth.col(1).setZero();
    poseweaver::ClothXpbdResult bounded_result;
    if (!poseweaver::run_cloth_cuda_jacobi(
            centered_cloth, cloth_faces, enclosing_planes, enclosing_faces, params, &bounded_result, &error))
        return 11;
    if ((bounded_result.frames.back() - centered_cloth).cwiseAbs().maxCoeff() > 1e-6)
        return 12;

    std::cout << status << ": CUDA SMPL skinning, Jacobi, and BVH collision tests passed\n";
    return 0;
}
