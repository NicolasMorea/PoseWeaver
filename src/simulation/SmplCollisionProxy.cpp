#include "simulation/SmplCollisionProxy.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>

namespace poseweaver
{
    namespace
    {
        struct CapsuleSpec
        {
            int a = -1;
            int b = -1;
            double t0 = 0.0;
            double t1 = 1.0;
            double scale_u = 1.0;
            double scale_v = 1.0;
            double min_u = 0.006;
            double max_u = 0.14;
            double min_v = 0.006;
            double max_v = 0.14;
            int frame_a = -1;
            int frame_b = -1;
            bool circular = true;
            bool allow_non_parent = false;
            int fit_group = 0; // 0 skinning pair, 1 torso, 2 neck, 3 head
            double axial_scale = 1.0;
            double fit_t0 = 0.0;
            double fit_t1 = 1.0;
            bool constrain_fit = false;
        };

        static const std::vector<CapsuleSpec>& proxy_capsule_specs()
        {
            static const std::vector<CapsuleSpec> specs = {
                { 1, 2, 0.0, 1.0, 1.00, 1.00, 0.035, 0.085, 0.035, 0.100, 0, 3, false, true, 0, 0.40, 0.0, 1.0, true }, // hips

                { 0, 3, 0.0, 1.0, 1.12, 0.96, 0.070, 0.155, 0.050, 0.125, 1, 2, false, false, 1, 0.22 },
                { 3, 6, 0.0, 1.0, 1.18, 1.00, 0.075, 0.170, 0.050, 0.130, 1, 2, false, false, 1, 0.22 },
                { 6, 9, 0.0, 1.0, 1.25, 1.02, 0.085, 0.190, 0.055, 0.140, 13, 14, false, false, 1, 0.22 },
                { 9, 12, 0.0, 0.78, 1.18, 0.98, 0.075, 0.175, 0.050, 0.130, 13, 14, false, false, 1, 0.18, 0.0, 0.55, true },
                { 12, 15, 0.02, 0.78, 1.00, 1.00, 0.022, 0.060, 0.020, 0.055, 13, 14, false, false, 2, 0.45, 0.12, 0.82 }, // neck
                { 12, 15, 0.82, 2.55, 1.10, 1.02, 0.060, 0.120, 0.052, 0.112, 13, 14, false, false, 3, 0.48, 0.08, 0.92 }, // head

                { 13, 16, 0.0, 1.0, 1.00, 1.00, 0.018, 0.055, 0.018, 0.055, -1, -1, true, false, 0, 0.45, 0.80, 1.0, true },
                { 14, 17, 0.0, 1.0, 1.00, 1.00, 0.018, 0.055, 0.018, 0.055, -1, -1, true, false, 0, 0.45, 0.80, 1.0, true },

                { 16, 18, 0.0, 0.5, 1.00, 1.00, 0.018, 0.075, 0.018, 0.075, -1, -1, true, false, 0 },
                { 16, 18, 0.5, 1.0, 1.00, 1.00, 0.016, 0.070, 0.016, 0.070, -1, -1, true, false, 0 },
                { 18, 20, 0.0, 0.5, 1.00, 1.00, 0.014, 0.060, 0.014, 0.060, -1, -1, true, false, 0 },
                { 18, 20, 0.5, 1.0, 1.00, 1.00, 0.012, 0.052, 0.012, 0.052, -1, -1, true, false, 0 },
                { 18, 20, 1.0, 1.36, 1.15, 0.70, 0.022, 0.070, 0.012, 0.045, -1, -1, false, false, 0 }, // left hand

                { 17, 19, 0.0, 0.5, 1.00, 1.00, 0.018, 0.075, 0.018, 0.075, -1, -1, true, false, 0 },
                { 17, 19, 0.5, 1.0, 1.00, 1.00, 0.016, 0.070, 0.016, 0.070, -1, -1, true, false, 0 },
                { 19, 21, 0.0, 0.5, 1.00, 1.00, 0.014, 0.060, 0.014, 0.060, -1, -1, true, false, 0 },
                { 19, 21, 0.5, 1.0, 1.00, 1.00, 0.012, 0.052, 0.012, 0.052, -1, -1, true, false, 0 },
                { 19, 21, 1.0, 1.36, 1.15, 0.70, 0.022, 0.070, 0.012, 0.045, -1, -1, false, false, 0 }, // right hand

                { 1, 4, 0.0, 0.5, 1.00, 1.00, 0.024, 0.095, 0.024, 0.095, -1, -1, true, false, 0 },
                { 1, 4, 0.5, 1.0, 1.00, 1.00, 0.022, 0.085, 0.022, 0.085, -1, -1, true, false, 0 },
                { 4, 7, 0.0, 0.5, 1.00, 1.00, 0.018, 0.070, 0.018, 0.070, -1, -1, true, false, 0 },
                { 4, 7, 0.5, 1.0, 1.00, 1.00, 0.014, 0.060, 0.014, 0.060, -1, -1, true, false, 0 },
                { 7, 10, 0.0, 1.0, 1.00, 0.82, 0.018, 0.080, 0.012, 0.055, -1, -1, false, false, 0 },

                { 2, 5, 0.0, 0.5, 1.00, 1.00, 0.024, 0.095, 0.024, 0.095, -1, -1, true, false, 0 },
                { 2, 5, 0.5, 1.0, 1.00, 1.00, 0.022, 0.085, 0.022, 0.085, -1, -1, true, false, 0 },
                { 5, 8, 0.0, 0.5, 1.00, 1.00, 0.018, 0.070, 0.018, 0.070, -1, -1, true, false, 0 },
                { 5, 8, 0.5, 1.0, 1.00, 1.00, 0.014, 0.060, 0.014, 0.060, -1, -1, true, false, 0 },
                { 8, 11, 0.0, 1.0, 1.00, 0.82, 0.018, 0.080, 0.012, 0.055, -1, -1, false, false, 0 },
            };
            return specs;
        }

        static bool joint_positions(
            const Eigen::MatrixXd& body_V,
            const SmplhModel& model,
            Eigen::MatrixXd& out_joints,
            std::string* error)
        {
            if (body_V.cols() != 3)
            {
                if (error)
                    *error = "Expected body vertices as #V x 3.";
                return false;
            }
            if (model.J_regressor.cols() != body_V.rows())
            {
                if (error)
                    *error = "SMPL joint regressor does not match body vertex count.";
                return false;
            }

            out_joints = model.J_regressor * body_V;
            if (out_joints.cols() != 3 || out_joints.rows() == 0)
            {
                if (error)
                    *error = "Failed to compute SMPL joint positions.";
                return false;
            }
            return true;
        }

        static bool smpl_parent_indices(const SmplhModel& model, std::vector<int>& out_parents)
        {
            const int J = model.joint_count();
            out_parents.assign((size_t)std::max(0, J), -1);
            if (J <= 0 || model.kintree_table.rows() != 2 || model.kintree_table.cols() != J)
                return false;

            auto is_identity_ids_row = [&](int row) -> bool
            {
                for (int col = 0; col < J; ++col)
                    if (model.kintree_table(row, col) != col)
                        return false;
                return true;
            };

            const bool row0_is_ids = is_identity_ids_row(0);
            const bool row1_is_ids = is_identity_ids_row(1);

            int ids_row = -1;
            int parents_row = -1;
            if (row0_is_ids && !row1_is_ids)
            {
                ids_row = 0;
                parents_row = 1;
            }
            else if (row1_is_ids && !row0_is_ids)
            {
                ids_row = 1;
                parents_row = 0;
            }
            else
            {
                const bool row0_has_root = (model.kintree_table(0, 0) == -1 || model.kintree_table(0, 0) == 4294967295LL);
                const bool row1_has_root = (model.kintree_table(1, 0) == -1 || model.kintree_table(1, 0) == 4294967295LL);
                if (row0_has_root == row1_has_root)
                    return false;
                parents_row = row0_has_root ? 0 : 1;
                ids_row = row0_has_root ? 1 : 0;
            }

            std::vector<int> id_to_col((size_t)J, -1);
            for (int col = 0; col < J; ++col)
            {
                const long long id = model.kintree_table(ids_row, col);
                if (id < 0 || id >= J)
                    return false;
                id_to_col[(size_t)id] = col;
            }

            for (int col = 0; col < J; ++col)
            {
                const long long pid = model.kintree_table(parents_row, col);
                if (pid == -1 || pid == 4294967295LL)
                    continue;
                if (pid < 0 || pid >= J)
                    return false;
                const int parent = id_to_col[(size_t)pid];
                if (parent < 0)
                    return false;
                out_parents[(size_t)col] = parent;
            }
            return true;
        }

        static bool allowed_capsule_spec(const std::vector<int>& parents, const CapsuleSpec& spec)
        {
            if (spec.allow_non_parent)
                return true;
            const int a = spec.a;
            const int b = spec.b;
            if (a < 0 || b < 0 || a >= (int)parents.size() || b >= (int)parents.size())
                return false;
            return parents[(size_t)a] == b || parents[(size_t)b] == a;
        }

        static Eigen::Vector3d capsule_endpoint(
            const Eigen::MatrixXd& joints,
            const CapsuleSpec& spec,
            double t)
        {
            const Eigen::Vector3d a = joints.row(spec.a).transpose();
            const Eigen::Vector3d b = joints.row(spec.b).transpose();
            return a + t * (b - a);
        }

        static void frame_from_axis(
            const Eigen::Vector3d& axis,
            Eigen::Vector3d& u,
            Eigen::Vector3d& v,
            Eigen::Vector3d& w);

        static void frame_for_spec(
            const Eigen::MatrixXd& joints,
            const CapsuleSpec& spec,
            const Eigen::Vector3d& a,
            const Eigen::Vector3d& b,
            Eigen::Vector3d& u,
            Eigen::Vector3d& v,
            Eigen::Vector3d& w)
        {
            frame_from_axis(b - a, u, v, w);
            if (spec.frame_a < 0 || spec.frame_b < 0
                || spec.frame_a >= joints.rows() || spec.frame_b >= joints.rows())
            {
                return;
            }

            Eigen::Vector3d hint = joints.row(spec.frame_b).transpose() - joints.row(spec.frame_a).transpose();
            hint -= hint.dot(w) * w;
            const double hn = hint.norm();
            if (!(hn > 1e-12) || !std::isfinite(hn))
                return;

            u = hint / hn;
            v = w.cross(u);
            const double vn = v.norm();
            if (vn > 1e-12 && std::isfinite(vn))
                v /= vn;
            else
                frame_from_axis(b - a, u, v, w);
        }

        static std::vector<Eigen::Vector2d> convex_hull_2d(std::vector<Eigen::Vector2d> points)
        {
            std::sort(points.begin(), points.end(), [](const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs)
            {
                return lhs.x() < rhs.x() || (lhs.x() == rhs.x() && lhs.y() < rhs.y());
            });
            points.erase(std::unique(points.begin(), points.end(), [](const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs)
            {
                return (lhs - rhs).squaredNorm() < 1e-16;
            }), points.end());
            if (points.size() < 3)
                return {};

            const auto cross = [](const Eigen::Vector2d& o, const Eigen::Vector2d& a, const Eigen::Vector2d& b)
            {
                return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
            };
            std::vector<Eigen::Vector2d> hull(2 * points.size());
            size_t k = 0;
            for (const Eigen::Vector2d& p : points)
            {
                while (k >= 2 && cross(hull[k - 2], hull[k - 1], p) <= 0.0)
                    --k;
                hull[k++] = p;
            }
            const size_t lower_size = k;
            for (size_t i = points.size() - 1; i > 0; --i)
            {
                const Eigen::Vector2d& p = points[i - 1];
                while (k > lower_size && cross(hull[k - 2], hull[k - 1], p) <= 0.0)
                    --k;
                hull[k++] = p;
            }
            if (k > 1)
                --k;
            hull.resize(k);
            return hull;
        }

        static bool polygon_area_center_aspect(
            const std::vector<Eigen::Vector2d>& polygon,
            double& out_area,
            Eigen::Vector2d& out_center,
            double& out_aspect)
        {
            if (polygon.size() < 3)
                return false;
            double twice_area = 0.0;
            Eigen::Vector2d center_sum = Eigen::Vector2d::Zero();
            for (size_t i = 0; i < polygon.size(); ++i)
            {
                const Eigen::Vector2d& p = polygon[i];
                const Eigen::Vector2d& q = polygon[(i + 1) % polygon.size()];
                const double cross = p.x() * q.y() - q.x() * p.y();
                twice_area += cross;
                center_sum += (p + q) * cross;
            }
            if (!(std::abs(twice_area) > 1e-12) || !std::isfinite(twice_area))
                return false;
            out_area = 0.5 * std::abs(twice_area);
            out_center = center_sum / (3.0 * twice_area);

            double extent_u = 0.0;
            double extent_v = 0.0;
            for (const Eigen::Vector2d& p : polygon)
            {
                extent_u = std::max(extent_u, std::abs(p.x() - out_center.x()));
                extent_v = std::max(extent_v, std::abs(p.y() - out_center.y()));
            }
            out_aspect = extent_v > 1e-9 ? extent_u / extent_v : 1.0;
            return out_area > 0.0 && out_center.allFinite() && std::isfinite(out_aspect);
        }

        struct SectionSegment
        {
            Eigen::Vector2d a;
            Eigen::Vector2d b;
        };

        static bool nearest_section_contour(
            const std::vector<SectionSegment>& segments,
            double& out_area,
            Eigen::Vector2d& out_center,
            double& out_aspect)
        {
            if (segments.size() < 3)
                return false;

            std::vector<int> parent(segments.size());
            for (int i = 0; i < (int)parent.size(); ++i)
                parent[(size_t)i] = i;
            const auto find_root = [&](int index)
            {
                int root = index;
                while (parent[(size_t)root] != root)
                    root = parent[(size_t)root];
                while (parent[(size_t)index] != index)
                {
                    const int next = parent[(size_t)index];
                    parent[(size_t)index] = root;
                    index = next;
                }
                return root;
            };
            const auto endpoints_touch = [](const SectionSegment& lhs, const SectionSegment& rhs)
            {
                constexpr double kTolerance2 = 1e-14;
                return (lhs.a - rhs.a).squaredNorm() <= kTolerance2
                    || (lhs.a - rhs.b).squaredNorm() <= kTolerance2
                    || (lhs.b - rhs.a).squaredNorm() <= kTolerance2
                    || (lhs.b - rhs.b).squaredNorm() <= kTolerance2;
            };
            for (int i = 0; i < (int)segments.size(); ++i)
            {
                for (int j = i + 1; j < (int)segments.size(); ++j)
                {
                    if (!endpoints_touch(segments[(size_t)i], segments[(size_t)j]))
                        continue;
                    const int ri = find_root(i);
                    const int rj = find_root(j);
                    if (ri != rj)
                        parent[(size_t)rj] = ri;
                }
            }

            bool found = false;
            double best_distance2 = std::numeric_limits<double>::infinity();
            for (int root_candidate = 0; root_candidate < (int)segments.size(); ++root_candidate)
            {
                if (find_root(root_candidate) != root_candidate)
                    continue;
                std::vector<Eigen::Vector2d> points;
                for (int i = 0; i < (int)segments.size(); ++i)
                {
                    if (find_root(i) == root_candidate)
                    {
                        points.push_back(segments[(size_t)i].a);
                        points.push_back(segments[(size_t)i].b);
                    }
                }

                const std::vector<Eigen::Vector2d> hull = convex_hull_2d(std::move(points));
                double area = 0.0;
                double aspect = 1.0;
                Eigen::Vector2d center = Eigen::Vector2d::Zero();
                if (!polygon_area_center_aspect(hull, area, center, aspect))
                    continue;
                const double distance2 = center.squaredNorm();
                if (!found || distance2 < best_distance2)
                {
                    found = true;
                    best_distance2 = distance2;
                    out_area = area;
                    out_center = center;
                    out_aspect = aspect;
                }
            }
            return found;
        }

        static void fitted_capsule_for_spec(
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            const Eigen::MatrixXd& joints,
            const CapsuleSpec& spec,
            double& out_u,
            double& out_v,
            Eigen::Vector2d& out_offset_a,
            Eigen::Vector2d& out_offset_b)
        {
            const Eigen::Vector3d a = capsule_endpoint(joints, spec, spec.t0);
            const Eigen::Vector3d b = capsule_endpoint(joints, spec, spec.t1);
            const Eigen::Vector3d ab = b - a;
            const double len2 = ab.squaredNorm();
            if (!(len2 > 1e-12) || !std::isfinite(len2))
            {
                out_u = spec.min_u;
                out_v = spec.min_v;
                out_offset_a.setZero();
                out_offset_b.setZero();
                return;
            }

            Eigen::Vector3d u, v, w;
            frame_for_spec(joints, spec, a, b, u, v, w);
            constexpr int kSliceCount = 7;
            double area_sum = 0.0;
            double log_aspect_sum = 0.0;
            Eigen::Vector2d center_sum = Eigen::Vector2d::Zero();
            Eigen::Vector2d center_t_sum = Eigen::Vector2d::Zero();
            double t_sum = 0.0;
            double t2_sum = 0.0;
            int valid_slices = 0;

            for (int slice = 0; slice < kSliceCount; ++slice)
            {
                const double fit_alpha = ((double)slice + 0.5) / (double)kSliceCount;
                const double t = spec.fit_t0 + fit_alpha * (spec.fit_t1 - spec.fit_t0);
                const Eigen::Vector3d center = a + t * ab;
                std::vector<SectionSegment> section_segments;
                section_segments.reserve(128);

                for (int fi = 0; fi < F.rows(); ++fi)
                {
                    const int ids[3] = { F(fi, 0), F(fi, 1), F(fi, 2) };
                    if (ids[0] < 0 || ids[1] < 0 || ids[2] < 0
                        || ids[0] >= V.rows() || ids[1] >= V.rows() || ids[2] >= V.rows())
                    {
                        continue;
                    }
                    Eigen::Vector3d p[3] = {
                        V.row(ids[0]).transpose(),
                        V.row(ids[1]).transpose(),
                        V.row(ids[2]).transpose()
                    };
                    double d[3] = {
                        (p[0] - center).dot(w),
                        (p[1] - center).dot(w),
                        (p[2] - center).dot(w)
                    };
                    std::vector<Eigen::Vector3d> intersections;
                    intersections.reserve(3);
                    for (int edge = 0; edge < 3; ++edge)
                    {
                        const int i0 = edge;
                        const int i1 = (edge + 1) % 3;
                        if ((d[i0] < 0.0 && d[i1] < 0.0) || (d[i0] > 0.0 && d[i1] > 0.0))
                            continue;
                        const double denom = d[i0] - d[i1];
                        if (std::abs(denom) < 1e-12)
                            continue;
                        const double alpha = std::clamp(d[i0] / denom, 0.0, 1.0);
                        const Eigen::Vector3d point = p[i0] + alpha * (p[i1] - p[i0]);
                        bool duplicate = false;
                        for (const Eigen::Vector3d& existing : intersections)
                            duplicate = duplicate || (point - existing).squaredNorm() < 1e-16;
                        if (!duplicate)
                            intersections.push_back(point);
                    }
                    if (intersections.size() == 2)
                    {
                        const Eigen::Vector3d da = intersections[0] - center;
                        const Eigen::Vector3d db = intersections[1] - center;
                        SectionSegment segment;
                        segment.a = Eigen::Vector2d(da.dot(u), da.dot(v));
                        segment.b = Eigen::Vector2d(db.dot(u), db.dot(v));
                        section_segments.push_back(segment);
                    }
                }

                double area = 0.0;
                double aspect = 1.0;
                Eigen::Vector2d section_center = Eigen::Vector2d::Zero();
                if (!nearest_section_contour(section_segments, area, section_center, aspect))
                    continue;

                area_sum += area;
                center_sum += section_center;
                center_t_sum += t * section_center;
                t_sum += t;
                t2_sum += t * t;
                log_aspect_sum += std::log(std::clamp(aspect, 0.25, 4.0));
                valid_slices++;
            }

            double target_product = valid_slices > 0
                ? (area_sum / (double)valid_slices) / 3.14159265358979323846
                : spec.min_u * spec.min_v;
            if (!(target_product > 0.0) || !std::isfinite(target_product))
                target_product = spec.min_u * spec.min_v;

            // Match the complete capped-capsule volume to the integrated section
            // volume. radius_w depends on the fitted section size, so solve the
            // small scalar relation by fixed-point iteration.
            const double segment_length = std::sqrt(len2);
            const double section_product = target_product;
            for (int iteration = 0; iteration < 4; ++iteration)
            {
                const double radius_w = spec.axial_scale * std::sqrt(target_product);
                target_product = section_product * segment_length
                    / std::max(1e-9, segment_length + (4.0 / 3.0) * radius_w);
            }

            const double base_aspect = valid_slices > 0
                ? std::exp(log_aspect_sum / (double)valid_slices)
                : 1.0;
            const double desired_aspect = std::clamp(
                base_aspect * spec.scale_u / std::max(1e-9, spec.scale_v),
                0.25,
                4.0);
            double ru = std::sqrt(target_product * desired_aspect);
            double rv = std::sqrt(target_product / desired_aspect);
            if (spec.circular)
            {
                double r = std::sqrt(target_product);
                if (valid_slices == 0 || spec.constrain_fit)
                {
                    const double min_r = std::max(spec.min_u, spec.min_v);
                    const double max_r = std::min(spec.max_u, spec.max_v);
                    r = std::clamp(r, min_r, max_r);
                }
                ru = r;
                rv = r;
            }
            else if (valid_slices == 0 || spec.constrain_fit)
            {
                ru = std::clamp(ru, spec.min_u, spec.max_u);
                rv = std::clamp(target_product / ru, spec.min_v, spec.max_v);
                ru = std::clamp(target_product / rv, spec.min_u, spec.max_u);
            }

            out_u = ru;
            out_v = rv;
            out_offset_a.setZero();
            out_offset_b.setZero();
            const double regression_denom =
                (double)valid_slices * t2_sum - t_sum * t_sum;
            if (valid_slices > 1 && std::abs(regression_denom) > 1e-12)
            {
                const Eigen::Vector2d slope =
                    ((double)valid_slices * center_t_sum - t_sum * center_sum)
                    / regression_denom;
                const Eigen::Vector2d intercept =
                    (center_sum - slope * t_sum) / (double)valid_slices;
                out_offset_a = intercept;
                out_offset_b = intercept + slope;
            }
            else if (valid_slices == 1)
            {
                out_offset_a = center_sum;
                out_offset_b = center_sum;
            }
            const double offset_limit = spec.constrain_fit ? 0.25 : 0.75;
            out_offset_a.x() = std::clamp(out_offset_a.x(), -offset_limit * ru, offset_limit * ru);
            out_offset_a.y() = std::clamp(out_offset_a.y(), -offset_limit * rv, offset_limit * rv);
            out_offset_b.x() = std::clamp(out_offset_b.x(), -offset_limit * ru, offset_limit * ru);
            out_offset_b.y() = std::clamp(out_offset_b.y(), -offset_limit * rv, offset_limit * rv);
        }

        static void frame_from_axis(
            const Eigen::Vector3d& axis,
            Eigen::Vector3d& u,
            Eigen::Vector3d& v,
            Eigen::Vector3d& w)
        {
            w = axis;
            const double wn = w.norm();
            if (wn > 1e-12 && std::isfinite(wn))
                w /= wn;
            else
                w = Eigen::Vector3d(0.0, 1.0, 0.0);

            Eigen::Vector3d seed = (std::abs(w.y()) < 0.9)
                ? Eigen::Vector3d(0.0, 1.0, 0.0)
                : Eigen::Vector3d(1.0, 0.0, 0.0);
            u = seed.cross(w);
            if (u.squaredNorm() < 1e-12)
                u = Eigen::Vector3d(0.0, 0.0, 1.0).cross(w);
            u.normalize();
            v = w.cross(u);
            v.normalize();
        }

        static void append_capsule_mesh(
            std::vector<Eigen::Vector3d>& verts,
            std::vector<Eigen::Vector3i>& faces,
            const Eigen::Vector3d& a,
            const Eigen::Vector3d& b,
            double radius_u,
            double radius_v,
            double radius_w,
            const Eigen::Vector3d& frame_u,
            const Eigen::Vector3d& frame_v)
        {
            radius_u = std::max(0.001, radius_u);
            radius_v = std::max(0.001, radius_v);
            radius_w = std::max(0.001, radius_w);
            const Eigen::Vector3d axis = b - a;
            double len = axis.norm();
            Eigen::Vector3d bb = b;
            if (!(len > 1e-9) || !std::isfinite(len))
            {
                bb = a + Eigen::Vector3d(0.0, 1.0, 0.0) * (2.0 * radius_w);
                len = (bb - a).norm();
            }

            Eigen::Vector3d u, v, w;
            frame_from_axis(bb - a, u, v, w);
            if (frame_u.squaredNorm() > 1e-12 && frame_v.squaredNorm() > 1e-12)
            {
                Eigen::Vector3d uu = frame_u - frame_u.dot(w) * w;
                const double un = uu.norm();
                if (un > 1e-12 && std::isfinite(un))
                {
                    u = uu / un;
                    v = w.cross(u);
                    const double vn = v.norm();
                    if (vn > 1e-12 && std::isfinite(vn))
                        v /= vn;
                }
            }

            constexpr int kSides = 10;
            constexpr int kHemisphere = 4;
            const int base = (int)verts.size();
            std::vector<std::pair<double, double>> rings;
            rings.reserve((size_t)(2 * kHemisphere + 2));

            constexpr double kPi = 3.14159265358979323846264338327950288;
            for (int i = 0; i <= kHemisphere; ++i)
            {
                const double theta = -0.5 * kPi + (0.5 * kPi) * (double)i / (double)kHemisphere;
                rings.push_back({ radius_w * std::sin(theta), std::cos(theta) });
            }
            rings.push_back({ len, 1.0 });
            for (int i = 1; i <= kHemisphere; ++i)
            {
                const double theta = (0.5 * kPi) * (double)i / (double)kHemisphere;
                rings.push_back({ len + radius_w * std::sin(theta), std::cos(theta) });
            }

            for (const auto& ring : rings)
            {
                const double z = ring.first;
                const double rr = std::max(0.0, ring.second);
                const Eigen::Vector3d center = a + z * w;
                for (int s = 0; s < kSides; ++s)
                {
                    const double ang = 2.0 * kPi * (double)s / (double)kSides;
                    verts.push_back(center + rr * (radius_u * std::cos(ang) * u + radius_v * std::sin(ang) * v));
                }
            }

            const int ring_count = (int)rings.size();
            for (int r = 0; r + 1 < ring_count; ++r)
            {
                for (int s = 0; s < kSides; ++s)
                {
                    const int sn = (s + 1) % kSides;
                    const int i0 = base + r * kSides + s;
                    const int i1 = base + r * kSides + sn;
                    const int j0 = base + (r + 1) * kSides + s;
                    const int j1 = base + (r + 1) * kSides + sn;
                    faces.push_back(Eigen::Vector3i(i0, i1, j1));
                    faces.push_back(Eigen::Vector3i(i0, j1, j0));
                }
            }
        }
    }

    bool build_smpl_collision_proxy(
        const Eigen::MatrixXd& reference_body_V,
        const SmplhModel& model,
        SmplCollisionProxy& out,
        std::string* error)
    {
        out = SmplCollisionProxy{};

        Eigen::MatrixXd joints;
        if (!joint_positions(reference_body_V, model, joints, error))
            return false;

        std::vector<int> parents;
        const bool have_parents = smpl_parent_indices(model, parents);
        for (const CapsuleSpec& spec : proxy_capsule_specs())
        {
            if (spec.a < 0 || spec.b < 0 || spec.a >= joints.rows() || spec.b >= joints.rows())
                continue;
            if (have_parents && !allowed_capsule_spec(parents, spec))
                continue;
            CapsuleSpec fitted_spec = spec;
            double radius_u = 0.03;
            double radius_v = 0.03;
            double radius_w = 0.03;
            Eigen::Vector2d offset_a = Eigen::Vector2d::Zero();
            Eigen::Vector2d offset_b = Eigen::Vector2d::Zero();
            for (int fit_iteration = 0; fit_iteration < 3; ++fit_iteration)
            {
                fitted_capsule_for_spec(
                    reference_body_V,
                    model.F,
                    joints,
                    fitted_spec,
                    radius_u,
                    radius_v,
                    offset_a,
                    offset_b);
                radius_w = std::max(
                    0.001,
                    fitted_spec.axial_scale * std::sqrt(radius_u * radius_v));

                if (fit_iteration < 2
                    && (fitted_spec.fit_group == 2 || fitted_spec.fit_group == 3))
                {
                    const double bone_length =
                        (joints.row(fitted_spec.b) - joints.row(fitted_spec.a)).norm();
                    if (bone_length > 1e-9 && std::isfinite(bone_length))
                    {
                        constexpr double kNeckHeadBoundary = 0.90;
                        if (fitted_spec.fit_group == 2)
                        {
                            fitted_spec.t1 = std::max(
                                fitted_spec.t0 + 0.08,
                                kNeckHeadBoundary - radius_w / bone_length);
                        }
                        else
                        {
                            fitted_spec.t0 = std::min(
                                fitted_spec.t1 - 0.12,
                                kNeckHeadBoundary + radius_w / bone_length);
                        }
                    }
                }
            }
            SmplCollisionCapsule capsule;
            capsule.joint_a = fitted_spec.a;
            capsule.joint_b = fitted_spec.b;
            capsule.t0 = fitted_spec.t0;
            capsule.t1 = fitted_spec.t1;
            capsule.radius_u = radius_u;
            capsule.radius_v = radius_v;
            capsule.radius_w = radius_w;
            capsule.offset_a_u = offset_a.x();
            capsule.offset_a_v = offset_a.y();
            capsule.offset_b_u = offset_b.x();
            capsule.offset_b_v = offset_b.y();
            capsule.radius = std::max(radius_u, radius_v);
            capsule.frame_joint_a = fitted_spec.frame_a;
            capsule.frame_joint_b = fitted_spec.frame_b;
            out.capsules.push_back(capsule);
        }

        if (out.capsules.empty())
        {
            if (error)
                *error = "No valid SMPL joints were available for the simple collision proxy.";
            return false;
        }
        return true;
    }

    bool apply_smpl_collision_proxy(
        const SmplCollisionProxy& proxy,
        const Eigen::MatrixXd& body_V,
        const SmplhModel& model,
        Eigen::MatrixXd& out_V,
        Eigen::MatrixXi& out_F,
        std::string* error)
    {
        out_V.resize(0, 3);
        out_F.resize(0, 3);
        if (!proxy.valid())
        {
            if (error)
                *error = "Invalid SMPL collision proxy.";
            return false;
        }

        Eigen::MatrixXd joints;
        if (!joint_positions(body_V, model, joints, error))
            return false;

        std::vector<Eigen::Vector3d> verts;
        std::vector<Eigen::Vector3i> faces;
        verts.reserve(proxy.capsules.size() * 100u);
        faces.reserve(proxy.capsules.size() * 180u);

        for (const SmplCollisionCapsule& capsule : proxy.capsules)
        {
            if (capsule.joint_a < 0 || capsule.joint_b < 0
                || capsule.joint_a >= joints.rows() || capsule.joint_b >= joints.rows())
            {
                continue;
            }
            CapsuleSpec spec;
            spec.a = capsule.joint_a;
            spec.b = capsule.joint_b;
            spec.t0 = capsule.t0;
            spec.t1 = capsule.t1;
            spec.frame_a = capsule.frame_joint_a;
            spec.frame_b = capsule.frame_joint_b;
            Eigen::Vector3d a = capsule_endpoint(joints, spec, capsule.t0);
            Eigen::Vector3d b = capsule_endpoint(joints, spec, capsule.t1);
            Eigen::Vector3d u, v, w;
            frame_for_spec(joints, spec, a, b, u, v, w);
            a += capsule.offset_a_u * u + capsule.offset_a_v * v;
            b += capsule.offset_b_u * u + capsule.offset_b_v * v;
            append_capsule_mesh(
                verts,
                faces,
                a,
                b,
                capsule.radius_u,
                capsule.radius_v,
                capsule.radius_w,
                u,
                v);
        }

        if (verts.empty() || faces.empty())
        {
            if (error)
                *error = "SMPL collision proxy generated an empty mesh.";
            return false;
        }

        out_V.resize((int)verts.size(), 3);
        for (int i = 0; i < (int)verts.size(); ++i)
            out_V.row(i) = verts[(size_t)i].transpose();

        out_F.resize((int)faces.size(), 3);
        for (int i = 0; i < (int)faces.size(); ++i)
            out_F.row(i) = faces[(size_t)i].transpose();

        return true;
    }

    bool apply_smpl_collision_proxy_capsules(
        const SmplCollisionProxy& proxy,
        const Eigen::MatrixXd& body_V,
        const SmplhModel& model,
        Eigen::MatrixXd& out_capsules,
        std::string* error)
    {
        out_capsules.resize(0, 15);
        if (!proxy.valid())
        {
            if (error)
                *error = "Invalid SMPL collision proxy.";
            return false;
        }

        Eigen::MatrixXd joints;
        if (!joint_positions(body_V, model, joints, error))
            return false;

        std::vector<SmplCollisionCapsule> valid;
        valid.reserve(proxy.capsules.size());
        for (const SmplCollisionCapsule& capsule : proxy.capsules)
        {
            if (capsule.joint_a < 0 || capsule.joint_b < 0
                || capsule.joint_a >= joints.rows() || capsule.joint_b >= joints.rows())
            {
                continue;
            }
            if (!(capsule.radius_u > 0.0) || !(capsule.radius_v > 0.0) || !(capsule.radius_w > 0.0)
                || !std::isfinite(capsule.radius_u) || !std::isfinite(capsule.radius_v)
                || !std::isfinite(capsule.radius_w))
                continue;
            valid.push_back(capsule);
        }

        if (valid.empty())
        {
            if (error)
                *error = "SMPL collision proxy generated no valid capsules.";
            return false;
        }

        out_capsules.resize((int)valid.size(), 15);
        for (int i = 0; i < (int)valid.size(); ++i)
        {
            const SmplCollisionCapsule& c = valid[(size_t)i];
            CapsuleSpec spec;
            spec.a = c.joint_a;
            spec.b = c.joint_b;
            spec.t0 = c.t0;
            spec.t1 = c.t1;
            spec.frame_a = c.frame_joint_a;
            spec.frame_b = c.frame_joint_b;
            Eigen::Vector3d a = capsule_endpoint(joints, spec, c.t0);
            Eigen::Vector3d b = capsule_endpoint(joints, spec, c.t1);
            Eigen::Vector3d u, v, w;
            frame_for_spec(joints, spec, a, b, u, v, w);
            a += c.offset_a_u * u + c.offset_a_v * v;
            b += c.offset_b_u * u + c.offset_b_v * v;
            out_capsules.block<1, 3>(i, 0) = a.transpose();
            out_capsules.block<1, 3>(i, 3) = b.transpose();
            out_capsules(i, 6) = c.radius_u;
            out_capsules(i, 7) = c.radius_v;
            out_capsules(i, 8) = c.radius_w;
            out_capsules.block<1, 3>(i, 9) = u.transpose();
            out_capsules.block<1, 3>(i, 12) = v.transpose();
        }
        return true;
    }
}
