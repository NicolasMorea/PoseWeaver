#pragma once

#include <Eigen/Core>
#include <string>

namespace poseweaver
{
    struct ClothPushOutResult
    {
        int moved_vertices = 0;
    };

    // Pushes penetrating cloth vertices outside the body surface by projecting them to the closest point on the body mesh and adding a small outward offset.
    bool push_cloth_vertices_outside_body(
        Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        double surface_offset,
        ClothPushOutResult* result,
        std::string* error);
}
