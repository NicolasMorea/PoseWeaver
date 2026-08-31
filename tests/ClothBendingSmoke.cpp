#include "simulation/xpbd/ClothXpbd.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace
{
    double hinge_angle(const Eigen::MatrixXd& vertices)
    {
        const Eigen::Vector3d xi = vertices.row(1);
        const Eigen::Vector3d xj = vertices.row(2);
        const Eigen::Vector3d xk0 = vertices.row(0);
        const Eigen::Vector3d xk1 = vertices.row(3);
        const Eigen::Vector3d edge = xj - xi;
        const Eigen::Vector3d n0 = edge.cross(xk0 - xi).normalized();
        const Eigen::Vector3d n1 = (xk1 - xi).cross(edge).normalized();
        const Eigen::Vector3d direction = edge.normalized();
        return std::atan2(direction.dot(n0.cross(n1)),
            std::clamp(n0.dot(n1), -1.0, 1.0));
    }
}

int main()
{
    Eigen::MatrixXd rest(4, 3);
    rest <<
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0;

    Eigen::MatrixXd folded = rest;
    folded(3, 2) = 1.25;

    Eigen::MatrixXi faces(2, 3);
    faces << 0, 1, 2, 1, 3, 2;

    Eigen::MatrixXd distant_body(3, 3);
    distant_body <<
        -10.0, -10.0, -10.0,
        -9.0, -10.0, -10.0,
        -10.0, -9.0, -10.0;
    Eigen::MatrixXi body_faces(1, 3);
    body_faces << 0, 1, 2;

    poseweaver::ClothXpbdParams params;
    params.duration_seconds = params.dt;
    params.max_frames = 2;
    params.substeps = 1;
    params.solver_iterations = 12;
    params.gravity_y = 0.0;
    params.damping = 0.0;
    params.stretch_enabled = false;
    params.bend_enabled = true;
    params.bending_method = poseweaver::BendingMethod::DihedralAngle;
    params.dihedral_compliance = 0.0;
    params.collision_enabled = false;
    params.reference_positions = rest;

    poseweaver::ClothXpbdResult result;
    std::string error;
    if (!poseweaver::run_cloth_xpbd(
            folded, faces, distant_body, body_faces, params, &result, &error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    if (result.frames.size() != 2 || !result.frames.back().allFinite())
    {
        std::cerr << "Stable dihedral solve produced an invalid frame.\n";
        return 2;
    }

    const double initial_error = std::abs(hinge_angle(folded));
    const double final_error = std::abs(hinge_angle(result.frames.back()));
    if (!(final_error < initial_error * 0.25))
    {
        std::cerr << "Hinge did not converge: " << initial_error << " -> " << final_error << '\n';
        return 3;
    }

    const double largest_displacement =
        (result.frames.back() - folded).rowwise().norm().maxCoeff();
    if (!(largest_displacement < 1.5))
    {
        std::cerr << "Hinge correction exceeded its trust region.\n";
        return 4;
    }
    return 0;
}
