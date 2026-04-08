#include "simulation/ClothToSmplMapping.h"

#include <igl/point_mesh_squared_distance.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>

namespace poseweaver
{
    static bool triangle_frame(
        const Eigen::RowVector3d& a,
        const Eigen::RowVector3d& b,
        const Eigen::RowVector3d& c,
        Eigen::RowVector3d& T,
        Eigen::RowVector3d& B,
        Eigen::RowVector3d& N)
    {
        const Eigen::RowVector3d e0 = b - a;
        const Eigen::RowVector3d e1 = c - a;

        const double e0n = e0.norm();
        if (!(e0n > 1e-20) || !std::isfinite(e0n))
            return false;

        T = e0 / e0n;
        N = e0.cross(e1);
        const double nn = N.norm();
        if (!(nn > 1e-20) || !std::isfinite(nn))
            return false;
        N /= nn;

        B = N.cross(T);
        const double bn = B.norm();
        if (!(bn > 1e-20) || !std::isfinite(bn))
            return false;
        B /= bn;
        return true;
    }

    static Eigen::RowVector3d barycentric_on_triangle(
        const Eigen::RowVector3d& p,
        const Eigen::RowVector3d& a,
        const Eigen::RowVector3d& b,
        const Eigen::RowVector3d& c)
    {
        const Eigen::RowVector3d v0 = b - a;
        const Eigen::RowVector3d v1 = c - a;
        const Eigen::RowVector3d v2 = p - a;

        const double d00 = v0.dot(v0);
        const double d01 = v0.dot(v1);
        const double d11 = v1.dot(v1);
        const double d20 = v2.dot(v0);
        const double d21 = v2.dot(v1);

        const double denom = d00 * d11 - d01 * d01;
        if (!(std::abs(denom) > 1e-20) || !std::isfinite(denom))
        {
            // Degenerate triangle: fall back to nearest vertex
            const double da = (p - a).squaredNorm();
            const double db = (p - b).squaredNorm();
            const double dc = (p - c).squaredNorm();
            if (da <= db && da <= dc) return Eigen::RowVector3d(1.0, 0.0, 0.0);
            if (db <= da && db <= dc) return Eigen::RowVector3d(0.0, 1.0, 0.0);
            return Eigen::RowVector3d(0.0, 0.0, 1.0);
        }

        const double v = (d11 * d20 - d01 * d21) / denom;
        const double w = (d00 * d21 - d01 * d20) / denom;
        const double u = 1.0 - v - w;

        Eigen::RowVector3d bc(u, v, w);

        // Numerical noise can put us slightly outside; clamp + renormalize
        for (int k = 0; k < 3; ++k)
            bc[k] = std::clamp(bc[k], 0.0, 1.0);

        const double s = bc.sum();
        if (s > 1e-20 && std::isfinite(s))
            bc /= s;

        return bc;
    }

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

        // Closest point on triangle mesh for each cloth vertex
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

            out->barycentric.row(vi) = barycentric_on_triangle(p, a, b, c);
            const Eigen::RowVector3d off = cloth_V.row(vi) - p;
            out->offset.row(vi) = off;

            Eigen::RowVector3d T, B, N;
            if (!triangle_frame(a, b, c, T, B, N))
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
