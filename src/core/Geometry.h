#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace poseweaver::core
{
    struct Aabb
    {
        Eigen::Vector3d min = Eigen::Vector3d(
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity());
        Eigen::Vector3d max = Eigen::Vector3d(
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity());

        inline void expand(const Eigen::Vector3d& p)
        {
            min = min.cwiseMin(p);
            max = max.cwiseMax(p);
        }

        inline void expand(const Aabb& b)
        {
            min = min.cwiseMin(b.min);
            max = max.cwiseMax(b.max);
        }

        inline Eigen::Vector3d center() const { return 0.5 * (min + max); }
        inline Eigen::Vector3d extents() const { return max - min; }

        inline bool valid() const
        {
            return (min.array() <= max.array()).all() && min.allFinite() && max.allFinite();
        }
    };

    inline double sqr_distance_point_aabb(const Eigen::Vector3d& p, const Aabb& b)
    {
        // Squared distance from point to AABB (0 if inside)
        double d2 = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double v = p[axis];
            if (v < b.min[axis])
            {
                const double d = b.min[axis] - v;
                d2 += d * d;
            }
            else if (v > b.max[axis])
            {
                const double d = v - b.max[axis];
                d2 += d * d;
            }
        }
        return d2;
    }

    inline Eigen::Vector3d closest_point_on_triangle(
        const Eigen::Vector3d& p,
        const Eigen::Vector3d& a,
        const Eigen::Vector3d& b,
        const Eigen::Vector3d& c,
        Eigen::Vector3d* out_bary = nullptr)
    {
        const Eigen::Vector3d ab = b - a;
        const Eigen::Vector3d ac = c - a;
        const Eigen::Vector3d ap = p - a;

        const double d1 = ab.dot(ap);
        const double d2 = ac.dot(ap);
        if (d1 <= 0.0 && d2 <= 0.0)
        {
            if (out_bary)
                *out_bary = Eigen::Vector3d(1.0, 0.0, 0.0);
            return a; // barycentric (1,0,0)
        }

        const Eigen::Vector3d bp = p - b;
        const double d3 = ab.dot(bp);
        const double d4 = ac.dot(bp);
        if (d3 >= 0.0 && d4 <= d3)
        {
            if (out_bary)
                *out_bary = Eigen::Vector3d(0.0, 1.0, 0.0);
            return b; // (0,1,0)
        }

        const double vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
        {
            const double v = d1 / (d1 - d3);
            if (out_bary)
                *out_bary = Eigen::Vector3d(1.0 - v, v, 0.0);
            return a + v * ab; // on AB
        }

        const Eigen::Vector3d cp = p - c;
        const double d5 = ab.dot(cp);
        const double d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6)
        {
            if (out_bary)
                *out_bary = Eigen::Vector3d(0.0, 0.0, 1.0);
            return c; // (0,0,1)
        }

        const double vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
        {
            const double w = d2 / (d2 - d6);
            if (out_bary)
                *out_bary = Eigen::Vector3d(1.0 - w, 0.0, w);
            return a + w * ac; // on AC
        }

        const double va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
        {
            const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            if (out_bary)
                *out_bary = Eigen::Vector3d(0.0, 1.0 - w, w);
            return b + w * (c - b); // on BC
        }

        // Inside face region
        const double denom = 1.0 / (va + vb + vc);
        const double v = vb * denom;
        const double w = vc * denom;
        const double u = 1.0 - v - w;
        if (out_bary)
            *out_bary = Eigen::Vector3d(u, v, w);
        return a + ab * v + ac * w;
    }

    inline Eigen::Vector3d triangle_unit_normal(
        const Eigen::Vector3d& a,
        const Eigen::Vector3d& b,
        const Eigen::Vector3d& c)
    {
        Eigen::Vector3d n = (b - a).cross(c - a);
        const double nn = n.norm();
        if (!(nn > 1e-20) || !std::isfinite(nn))
            return Eigen::Vector3d(0.0, 0.0, 0.0);
        return n / nn;
    }

    inline bool triangle_unit_normal(
        const Eigen::RowVector3d& a,
        const Eigen::RowVector3d& b,
        const Eigen::RowVector3d& c,
        Eigen::RowVector3d& out_n)
    {
        out_n = (b - a).cross(c - a);
        const double nn = out_n.norm();
        if (!(nn > 1e-20) || !std::isfinite(nn))
            return false;
        out_n /= nn;
        return true;
    }

    // Computes an orthonormal tangent frame (T,B,N) for a triangle.
    inline bool triangle_frame(
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

    inline Eigen::RowVector3d barycentric_on_triangle(
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
}
