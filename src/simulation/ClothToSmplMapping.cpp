#include "simulation/ClothToSmplMapping.h"

#include "core/Geometry.h"

#include <igl/point_mesh_squared_distance.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>

namespace poseweaver
{
    bool compute_cloth_to_smpl_mapping(
        const Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& smpl_V,
        const Eigen::MatrixXi& smpl_F,
        ClothToSmplMapping* out,
        std::string* error)
    {
        if (out)
            out->clear();

        if (cloth_V.cols() != 3 || smpl_V.cols() != 3)
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
        if (smpl_F.cols() != 3)
        {
            if (error)
                *error = "Expected SMPL faces to be triangles (#F x 3).";
            return false;
        }
        if (cloth_V.rows() == 0 || smpl_V.rows() == 0 || smpl_F.rows() == 0)
        {
            if (error)
                *error = "Missing cloth/SMPL mesh data.";
            return false;
        }
        if (!out)
        {
            if (error)
                *error = "Output mapping pointer is null.";
            return false;
        }

        Eigen::VectorXd sqrD;
        Eigen::VectorXi I;
        Eigen::MatrixXd C;

        igl::point_mesh_squared_distance(cloth_V, smpl_V, smpl_F, sqrD, I, C);

        if (I.size() != cloth_V.rows() || C.rows() != cloth_V.rows() || C.cols() != 3)
        {
            if (error)
                *error = "Closest-point query returned unexpected sizes.";
            return false;
        }

        out->face_index = I;
        out->barycentric.resize(cloth_V.rows(), 3);
        out->offset.resize(cloth_V.rows(), 3);
        out->offset_local.resize(cloth_V.rows(), 3);

        for (int vi = 0; vi < cloth_V.rows(); ++vi)
        {
            const int fi = I[vi];
            if (fi < 0 || fi >= smpl_F.rows())
            {
                if (error)
                    *error = "Closest-point query returned an invalid face index.";
                out->clear();
                return false;
            }

            const int a_i = smpl_F(fi, 0);
            const int b_i = smpl_F(fi, 1);
            const int c_i = smpl_F(fi, 2);
            if (a_i < 0 || b_i < 0 || c_i < 0 ||
                a_i >= smpl_V.rows() || b_i >= smpl_V.rows() || c_i >= smpl_V.rows())
            {
                if (error)
                    *error = "SMPL face has out-of-range vertex index.";
                out->clear();
                return false;
            }

            const Eigen::RowVector3d p = C.row(vi);
            const Eigen::RowVector3d a = smpl_V.row(a_i);
            const Eigen::RowVector3d b = smpl_V.row(b_i);
            const Eigen::RowVector3d c = smpl_V.row(c_i);

            out->barycentric.row(vi) = core::barycentric_on_triangle(p, a, b, c);
            const Eigen::RowVector3d off = cloth_V.row(vi) - p;
            out->offset.row(vi) = off;

            Eigen::RowVector3d T, B, N;
            if (!core::triangle_frame(a, b, c, T, B, N))
            {
                // Degenerate face: store offset as-is in a fallback "basis"
                out->offset_local.row(vi) = off;
            }
            else
            {
                out->offset_local(vi, 0) = off.dot(T);
                out->offset_local(vi, 1) = off.dot(B);
                out->offset_local(vi, 2) = off.dot(N);
            }
        }

        return out->valid();
    }
}
