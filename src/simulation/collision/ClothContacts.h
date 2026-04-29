#pragma once

#include "core/Geometry.h"
#include "simulation/collision/TriangleBvh.h"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace poseweaver::collision
{
    struct VertexTriangleContact
    {
        int v = -1;
        int face = -1;
    };

    struct VertexTriangleContactQueryStats
    {
        int vertices_total = 0;
        int vertices_in_expanded = 0;
        int bvh_calls = 0;
        int bvh_full_calls = 0;
        int contacts_found = 0;
        double ms_total = 0.0;
        double ms_bvh = 0.0; // time spent inside TriangleBvh::closest_triangle()
    };

    inline bool find_vertex_triangle_contacts(
        const Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const TriangleBvh& bvh,
        double thickness,
        std::vector<VertexTriangleContact>& out,
        VertexTriangleContactQueryStats* stats = nullptr)
    {
        out.clear();
        if (stats)
            *stats = VertexTriangleContactQueryStats{};

        if (cloth_V.rows() == 0 || body_V.rows() == 0 || body_F.rows() == 0)
            return false;

        using Clock = std::chrono::steady_clock;
        const auto t0 = stats ? Clock::now() : Clock::time_point{};

        const core::Aabb bb = bvh.bounds();
        if (!bb.valid())
            return false;

        const double thick = std::max(0.0, thickness);
        const double thick2 = thick * thick;

        const double margin = std::max(0.02, thick * 4.0);
        const core::Aabb expanded{ bb.min - Eigen::Vector3d::Constant(margin), bb.max + Eigen::Vector3d::Constant(margin) };

        auto inside_expanded = [&](const Eigen::Vector3d& p)
        {
            return (p.array() >= expanded.min.array()).all() && (p.array() <= expanded.max.array()).all();
        };
        auto inside_bb = [&](const Eigen::Vector3d& p)
        {
            return (p.array() >= bb.min.array()).all() && (p.array() <= bb.max.array()).all();
        };

        if (stats)
            stats->vertices_total = cloth_V.rows();

        out.reserve((size_t)cloth_V.rows());
        for (int vi = 0; vi < cloth_V.rows(); ++vi)
        {
            const Eigen::Vector3d p = cloth_V.row(vi);
            if (!inside_expanded(p))
                continue;

            if (stats)
                stats->vertices_in_expanded++;

            int face = -1;
            Eigen::Vector3d closest;
            double dist2 = std::numeric_limits<double>::infinity();

            const double max_dist2 = margin * margin;

            bool ok = false;
            if (stats)
            {
                const auto tb = Clock::now();
                ok = bvh.closest_triangle(p, body_V, body_F, face, closest, dist2, max_dist2);
                stats->ms_bvh += std::chrono::duration<double, std::milli>(Clock::now() - tb).count();
                stats->bvh_calls++;
            }
            else
            {
                ok = bvh.closest_triangle(p, body_V, body_F, face, closest, dist2, max_dist2);
            }

            // Fallback to unbounded query for deep penetrations inside the body AABB.
            if (!ok)
            {
                if (!inside_bb(p))
                    continue;

                if (stats)
                {
                    const auto tb = Clock::now();
                    ok = bvh.closest_triangle(p, body_V, body_F, face, closest, dist2);
                    stats->ms_bvh += std::chrono::duration<double, std::milli>(Clock::now() - tb).count();
                    stats->bvh_calls++;
                    stats->bvh_full_calls++;
                }
                else
                {
                    ok = bvh.closest_triangle(p, body_V, body_F, face, closest, dist2);
                }

                if (!ok)
                    continue;
            }

            const int i0 = body_F(face, 0);
            const int i1 = body_F(face, 1);
            const int i2 = body_F(face, 2);
            if (i0 < 0 || i1 < 0 || i2 < 0)
                continue;
            if (i0 >= body_V.rows() || i1 >= body_V.rows() || i2 >= body_V.rows())
                continue;

            const Eigen::Vector3d n = core::triangle_unit_normal(
                body_V.row(i0).transpose(),
                body_V.row(i1).transpose(),
                body_V.row(i2).transpose());
            if (!(n.squaredNorm() > 0.0))
                continue;

            const double sd_plane = n.dot(p - closest);

            if (sd_plane >= 0.0)
            {
                if (!(dist2 < thick2) || !std::isfinite(dist2))
                    continue;
                const double dist = std::sqrt(std::max(0.0, dist2));
                const double depth = thick - dist;
                if (depth > 0.0 && std::isfinite(depth))
                {
                    out.push_back(VertexTriangleContact{ vi, face });
                    if (stats)
                        stats->contacts_found++;
                }
            }
            else
            {
                const double dist = std::sqrt(std::max(0.0, dist2));
                const double sd = (dist > 1e-12) ? -dist : sd_plane;
                const double depth = thick - sd;
                if (depth > 0.0 && std::isfinite(depth))
                {
                    out.push_back(VertexTriangleContact{ vi, face });
                    if (stats)
                        stats->contacts_found++;
                }
            }
        }

        if (stats)
            stats->ms_total = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        return true;
    }

    inline bool find_vertex_triangle_contacts(
        const Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const TriangleBvh& bvh,
        double thickness,
        const std::vector<int>& vertex_indices,
        std::vector<VertexTriangleContact>& out,
        VertexTriangleContactQueryStats* stats = nullptr)
    {
        out.clear();
        if (stats)
            *stats = VertexTriangleContactQueryStats{};

        if (vertex_indices.empty())
            return true;
        if (cloth_V.rows() == 0 || body_V.rows() == 0 || body_F.rows() == 0)
            return false;

        using Clock = std::chrono::steady_clock;
        const auto t0 = stats ? Clock::now() : Clock::time_point{};

        const core::Aabb bb = bvh.bounds();
        if (!bb.valid())
            return false;

        const double thick = std::max(0.0, thickness);
        const double thick2 = thick * thick;

        const double margin = std::max(0.02, thick * 4.0);
        const core::Aabb expanded{ bb.min - Eigen::Vector3d::Constant(margin), bb.max + Eigen::Vector3d::Constant(margin) };

        auto inside_expanded = [&](const Eigen::Vector3d& p)
        {
            return (p.array() >= expanded.min.array()).all() && (p.array() <= expanded.max.array()).all();
        };
        auto inside_bb = [&](const Eigen::Vector3d& p)
        {
            return (p.array() >= bb.min.array()).all() && (p.array() <= bb.max.array()).all();
        };

        if (stats)
            stats->vertices_total = (int)vertex_indices.size();

        out.reserve(vertex_indices.size());
        for (int vi : vertex_indices)
        {
            if (vi < 0 || vi >= cloth_V.rows())
                continue;

            const Eigen::Vector3d p = cloth_V.row(vi);
            if (!inside_expanded(p))
                continue;

            if (stats)
                stats->vertices_in_expanded++;

            int face = -1;
            Eigen::Vector3d closest;
            double dist2 = std::numeric_limits<double>::infinity();

            const double max_dist2 = margin * margin;

            bool ok = false;
            if (stats)
            {
                const auto tb = Clock::now();
                ok = bvh.closest_triangle(p, body_V, body_F, face, closest, dist2, max_dist2);
                stats->ms_bvh += std::chrono::duration<double, std::milli>(Clock::now() - tb).count();
                stats->bvh_calls++;
            }
            else
            {
                ok = bvh.closest_triangle(p, body_V, body_F, face, closest, dist2, max_dist2);
            }

            if (!ok)
            {
                if (!inside_bb(p))
                    continue;

                if (stats)
                {
                    const auto tb = Clock::now();
                    ok = bvh.closest_triangle(p, body_V, body_F, face, closest, dist2);
                    stats->ms_bvh += std::chrono::duration<double, std::milli>(Clock::now() - tb).count();
                    stats->bvh_calls++;
                    stats->bvh_full_calls++;
                }
                else
                {
                    ok = bvh.closest_triangle(p, body_V, body_F, face, closest, dist2);
                }

                if (!ok)
                    continue;
            }

            const int i0 = body_F(face, 0);
            const int i1 = body_F(face, 1);
            const int i2 = body_F(face, 2);
            if (i0 < 0 || i1 < 0 || i2 < 0)
                continue;
            if (i0 >= body_V.rows() || i1 >= body_V.rows() || i2 >= body_V.rows())
                continue;

            const Eigen::Vector3d n = core::triangle_unit_normal(
                body_V.row(i0).transpose(),
                body_V.row(i1).transpose(),
                body_V.row(i2).transpose());
            if (!(n.squaredNorm() > 0.0))
                continue;

            const double sd_plane = n.dot(p - closest);

            if (sd_plane >= 0.0)
            {
                if (!(dist2 < thick2) || !std::isfinite(dist2))
                    continue;
                const double dist = std::sqrt(std::max(0.0, dist2));
                const double depth = thick - dist;
                if (depth > 0.0 && std::isfinite(depth))
                {
                    out.push_back(VertexTriangleContact{ vi, face });
                    if (stats)
                        stats->contacts_found++;
                }
            }
            else
            {
                const double dist = std::sqrt(std::max(0.0, dist2));
                const double sd = (dist > 1e-12) ? -dist : sd_plane;
                const double depth = thick - sd;
                if (depth > 0.0 && std::isfinite(depth))
                {
                    out.push_back(VertexTriangleContact{ vi, face });
                    if (stats)
                        stats->contacts_found++;
                }
            }
        }

        if (stats)
            stats->ms_total = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        return true;
    }

    inline int apply_vertex_triangle_pushout(
        Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const std::vector<VertexTriangleContact>& contacts,
        double thickness,
        int passes = 1)
    {
        if (cloth_V.rows() == 0 || contacts.empty())
            return 0;

        const double thick = std::max(0.0, thickness);
        if (passes < 1)
            passes = 1;

        int moved = 0;
        for (int pass = 0; pass < passes; ++pass)
        {
            for (const VertexTriangleContact& c : contacts)
            {
                const int vi = c.v;
                const int face = c.face;
                if (vi < 0 || vi >= cloth_V.rows())
                    continue;
                if (face < 0 || face >= body_F.rows())
                    continue;

                const int i0 = body_F(face, 0);
                const int i1 = body_F(face, 1);
                const int i2 = body_F(face, 2);
                if (i0 < 0 || i1 < 0 || i2 < 0)
                    continue;
                if (i0 >= body_V.rows() || i1 >= body_V.rows() || i2 >= body_V.rows())
                    continue;

                const Eigen::Vector3d a = body_V.row(i0);
                const Eigen::Vector3d b = body_V.row(i1);
                const Eigen::Vector3d ctri = body_V.row(i2);

                const Eigen::Vector3d p = cloth_V.row(vi);
                const Eigen::Vector3d q = core::closest_point_on_triangle(p, a, b, ctri, nullptr);

                const Eigen::Vector3d n = core::triangle_unit_normal(a, b, ctri);
                if (!(n.squaredNorm() > 0.0))
                    continue;

                const Eigen::Vector3d pq = p - q;
                const double dist = pq.norm();
                const double sd_plane = n.dot(pq);
                const double sgn = (sd_plane >= 0.0) ? 1.0 : -1.0;

                double sd = sd_plane;
                Eigen::Vector3d grad = n;
                if (dist > 1e-12 && std::isfinite(dist))
                {
                    sd = sgn * dist;
                    grad = (sgn / dist) * pq;
                }

                const double depth = thick - sd;
                if (depth > 0.0 && std::isfinite(depth) && grad.allFinite())
                {
                    cloth_V.row(vi) += (depth * grad).transpose();
                    moved++;
                }
            }
        }

        return moved;
    }
}
