#include "simulation/ClothMeshProcessor.h"

#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>

namespace poseweaver
{
    bool push_cloth_vertices_outside_body(
        Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const double surface_offset,
        ClothPushOutResult* result,
        std::string* error)
    {
        if (result)
            *result = ClothPushOutResult{};

        if (cloth_V.cols() != 3 || body_V.cols() != 3)
        {
            if (error)
                *error = "Expected 3D vertex positions (#V x 3).";
            return false;
        }
        if (cloth_F.size() > 0 && cloth_F.cols() != 3)
        {
            if (error)
                *error = "Expected cloth faces to be triangles (#F x 3).";
            return false;
        }
        if (body_F.cols() != 3)
        {
            if (error)
                *error = "Expected body faces to be triangles (#F x 3).";
            return false;
        }
        if (cloth_V.rows() == 0 || body_V.rows() == 0 || body_F.rows() == 0)
        {
            if (error)
                *error = "Missing cloth/body mesh data.";
            return false;
        }
        if (!(surface_offset >= 0.0) || !std::isfinite(surface_offset))
        {
            if (error)
                *error = "surface_offset must be a finite non-negative value.";
            return false;
        }

        const Eigen::MatrixXd P = cloth_V;

        Eigen::VectorXd S;
        Eigen::VectorXi I;
        Eigen::MatrixXd C;
        Eigen::MatrixXd N; // unused but required by the API

        igl::signed_distance(P, body_V, body_F, igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, S, I, C, N);

        const Eigen::RowVector3d body_centroid = body_V.colwise().mean();

        int moved = 0;
        for (int vi = 0; vi < P.rows(); ++vi)
        {
            const double sd = S[vi];
            if (!(sd < 0.0))
                continue;

            Eigen::RowVector3d dir = (C.row(vi) - P.row(vi));
            double n = dir.norm();
            if (!(n > 1e-12))
            {
                dir = (P.row(vi) - body_centroid);
                n = dir.norm();
                if (!(n > 1e-12))
                    continue;
            }
            dir /= n;

            const double delta = std::max(0.0, surface_offset - sd);
            cloth_V.row(vi) = P.row(vi) + delta * dir;
            moved++;
        }

        if (result)
            result->moved_vertices = moved;
        return true;
    }
}
