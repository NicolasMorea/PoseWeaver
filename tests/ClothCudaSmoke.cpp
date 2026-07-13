#include "simulation/gpu/ClothCudaJacobi.h"

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
    std::string error;
    if (!poseweaver::run_cloth_cuda_jacobi(
            cloth, cloth_faces, body, body_faces, params, &result, &error))
    {
        std::cerr << error << '\n';
        return 2;
    }
    if (result.frames.size() != 2 || !result.frames.back().allFinite())
        return 3;
    if (result.frames.back().col(1).minCoeff() <= cloth.col(1).minCoeff())
        return 4;

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

    std::cout << status << ": CUDA Jacobi, BVH collision, and distant-backface tests passed\n";
    return 0;
}
