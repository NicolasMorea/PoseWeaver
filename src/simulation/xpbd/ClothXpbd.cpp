#include "simulation/xpbd/ClothXpbd.h"

#include "core/Geometry.h"
#include "simulation/collision/ClothContacts.h"
#include "simulation/collision/TriangleBvh.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace poseweaver
{
    namespace
    {
        constexpr double kEps = 1e-12;

        static inline long long edge_key(int a, int b)
        {
            const int lo = std::min(a, b);
            const int hi = std::max(a, b);
            return (static_cast<long long>(lo) << 32) | static_cast<unsigned int>(hi);
        }

        struct StretchEdge
        {
            int i = -1;
            int j = -1;
            double rest_len = 0.0;
        };

        struct OrientedEdgeFace
        {
            int v0 = -1;
            int v1 = -1;
            int opp = -1;
        };

        struct BendEdge
        {
            int k0 = -1;
            int k1 = -1;
            double rest_len = 0.0;
        };

        static void build_stretch_edges(
            const Eigen::MatrixXd& V_rest,
            const Eigen::MatrixXi& F,
            std::vector<StretchEdge>& out_edges)
        {
            out_edges.clear();
            if (V_rest.rows() == 0 || F.rows() == 0)
                return;

            std::unordered_map<long long, StretchEdge> edges;
            edges.reserve((size_t)F.rows() * 2u);

            auto add_edge = [&](int a, int b)
            {
                if (a < 0 || b < 0)
                    return;
                if (a >= V_rest.rows() || b >= V_rest.rows())
                    return;

                const long long key = edge_key(a, b);
                if (edges.find(key) != edges.end())
                    return;

                StretchEdge e;
                e.i = std::min(a, b);
                e.j = std::max(a, b);
                e.rest_len = (V_rest.row(e.j) - V_rest.row(e.i)).norm();
                edges.emplace(key, e);
            };

            for (int fi = 0; fi < F.rows(); ++fi)
            {
                const int a = F(fi, 0);
                const int b = F(fi, 1);
                const int c = F(fi, 2);
                add_edge(a, b);
                add_edge(b, c);
                add_edge(c, a);
            }

            out_edges.reserve(edges.size());
            for (const auto& kv : edges)
                out_edges.push_back(kv.second);
        }

        static void build_bend_edges(
            const Eigen::MatrixXd& V_rest,
            const Eigen::MatrixXi& F,
            std::vector<BendEdge>& out_edges)
        {
            out_edges.clear();
            if (V_rest.rows() == 0 || F.rows() == 0)
                return;

            struct EdgeInfo
            {
                OrientedEdgeFace f[2];
                int count = 0;
            };

            std::unordered_map<long long, EdgeInfo> edge_faces;
            edge_faces.reserve((size_t)F.rows() * 2u);

            for (int fi = 0; fi < F.rows(); ++fi)
            {
                const int a = F(fi, 0);
                const int b = F(fi, 1);
                const int c = F(fi, 2);
                if (a < 0 || b < 0 || c < 0)
                    continue;
                if (a >= V_rest.rows() || b >= V_rest.rows() || c >= V_rest.rows())
                    continue;

                const int v[3] = { a, b, c };
                const int opp[3] = { c, a, b };

                for (int ei = 0; ei < 3; ++ei)
                {
                    const int v0 = v[ei];
                    const int v1 = v[(ei + 1) % 3];
                    const int vo = opp[ei];

                    const long long key = edge_key(v0, v1);
                    auto& info = edge_faces[key];
                    if (info.count >= 2)
                        continue;

                    info.f[info.count++] = OrientedEdgeFace{ v0, v1, vo };
                }
            }

            out_edges.reserve(edge_faces.size());
            for (const auto& kv : edge_faces)
            {
                const EdgeInfo& e = kv.second;
                if (e.count != 2)
                    continue;

                const int k0 = e.f[0].opp;
                const int k1 = e.f[1].opp;
                if (k0 < 0 || k1 < 0)
                    continue;
                if (k0 >= V_rest.rows() || k1 >= V_rest.rows())
                    continue;

                BendEdge be;
                be.k0 = k0;
                be.k1 = k1;
                be.rest_len = (V_rest.row(k0) - V_rest.row(k1)).norm();
                out_edges.push_back(be);
            }
        }

        static bool compute_lumped_masses(
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            Eigen::VectorXd& out_mass)
        {
            out_mass = Eigen::VectorXd::Zero(V.rows());
            if (V.rows() == 0)
                return false;
            if (F.rows() == 0)
            {
                out_mass.setOnes();
                return true;
            }

            for (int fi = 0; fi < F.rows(); ++fi)
            {
                const int a = F(fi, 0);
                const int b = F(fi, 1);
                const int c = F(fi, 2);
                if (a < 0 || b < 0 || c < 0)
                    continue;
                if (a >= V.rows() || b >= V.rows() || c >= V.rows())
                    continue;

                const Eigen::Vector3d va = V.row(a);
                const Eigen::Vector3d vb = V.row(b);
                const Eigen::Vector3d vc = V.row(c);
                const double area = 0.5 * (vb - va).cross(vc - va).norm();
                if (!(area > 0.0) || !std::isfinite(area))
                    continue;

                const double m = area / 3.0;
                out_mass[a] += m;
                out_mass[b] += m;
                out_mass[c] += m;
            }

            const double mean = (out_mass.size() > 0) ? out_mass.mean() : 1.0;
            const double fallback = (mean > 0.0 && std::isfinite(mean)) ? mean : 1.0;
            for (int i = 0; i < out_mass.size(); ++i)
            {
                if (!(out_mass[i] > 0.0) || !std::isfinite(out_mass[i]))
                    out_mass[i] = fallback;
            }
            return true;
        }

        static void apply_wind_prediction(
            Eigen::MatrixXd& X_pred,
            const Eigen::MatrixXd& X,
            int step,
            double dt,
            const ClothXpbdParams& params)
        {
            if (!(params.wind_force > 0.0))
                return;

            Eigen::Vector3d dir = params.wind_dir;
            const double dn = dir.norm();
            if (!(dn > 1e-12) || !std::isfinite(dn))
                return;
            dir /= dn;

            const double t = (double)step * dt;
            const double w_time = 1.7;
            const Eigen::Vector3d k_space(12.9898, 78.233, 37.719);

            const double time_term = (params.wind_random_time > 0.0)
                ? std::sin(t * w_time)
                : 0.0;

            for (int i = 0; i < X.rows(); ++i)
            {
                const Eigen::Vector3d pos = X.row(i);
                const double space_term = (params.wind_random_space > 0.0)
                    ? std::sin(pos.dot(k_space))
                    : 0.0;

                double scale = 1.0;
                scale += 0.25 * params.wind_random_space * space_term;
                scale += 0.25 * params.wind_random_time * time_term;
                if (!std::isfinite(scale))
                    scale = 1.0;
                scale = std::max(0.0, scale);

                const Eigen::Vector3d a = (params.wind_force * scale) * dir;
                X_pred.row(i) += ((dt * dt) * a).transpose();
            }
        }

        static inline double compliance_alpha(double compliance, double dt)
        {
            compliance = std::max(0.0, compliance);
            if (!(dt > 0.0))
                return compliance;
            return compliance / (dt * dt);
        }

        static inline void apply_collision_constraint(
            Eigen::MatrixXd& Xs,
            const Eigen::MatrixXd& body_V,
            const Eigen::MatrixXi& body_F,
            const collision::VertexTriangleContact& c,
            const Eigen::VectorXd& inv_mass,
            int N,
            double thickness,
            double col_alpha,
            std::vector<double>& lambda_col)
        {
            const int vi = c.v;
            const int face = c.face;
            if (vi < 0 || vi >= N || face < 0 || face >= body_F.rows())
                return;

            const double w = inv_mass[vi];
            if (!(w > 0.0))
                return;

            const int i0 = body_F(face, 0);
            const int i1 = body_F(face, 1);
            const int i2 = body_F(face, 2);
            if (i0 < 0 || i1 < 0 || i2 < 0)
                return;
            if (i0 >= body_V.rows() || i1 >= body_V.rows() || i2 >= body_V.rows())
                return;

            const Eigen::Vector3d a = body_V.row(i0);
            const Eigen::Vector3d b = body_V.row(i1);
            const Eigen::Vector3d ctri = body_V.row(i2);

            const Eigen::Vector3d p = Xs.row(vi);
            const Eigen::Vector3d q = core::closest_point_on_triangle(p, a, b, ctri, nullptr);
            const Eigen::Vector3d n = core::triangle_unit_normal(a, b, ctri);
            if (!(n.squaredNorm() > 0.0))
                return;

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

            const double C = sd - thickness;
            if (!(C < 0.0) || !std::isfinite(C) || !grad.allFinite())
                return;

            const double denom = w + col_alpha;
            if (!(denom > kEps))
                return;

            double& lam = lambda_col[(size_t)vi];
            const double dlambda = (-C - col_alpha * lam) / denom;
            lam += dlambda;

            Xs.row(vi) += (w * dlambda * grad).transpose();
        }

        static bool find_self_collision_contacts(
            const Eigen::MatrixXd& X,
            const Eigen::MatrixXi& F,
            const collision::TriangleBvh& bvh,
            double thickness,
            std::vector<collision::VertexTriangleContact>& out)
        {
            out.clear();
            if (X.rows() == 0 || F.rows() == 0)
                return false;

            const double thick = std::max(0.0, thickness);
            if (!(thick > 0.0))
                return true;

            const double thick2 = thick * thick;
            out.reserve((size_t)std::max<int>(16, (int)X.rows() / 8));

            for (int vi = 0; vi < X.rows(); ++vi)
            {
                int face = -1;
                Eigen::Vector3d closest;
                double dist2 = std::numeric_limits<double>::infinity();
                const Eigen::Vector3d p = X.row(vi);
                if (!bvh.closest_triangle(p, X, F, face, closest, dist2, thick2, vi))
                    continue;
                if (!(dist2 < thick2) || !std::isfinite(dist2))
                    continue;
                out.push_back(collision::VertexTriangleContact{ vi, face });
            }

            return true;
        }

        static inline void apply_self_collision_constraint(
            Eigen::MatrixXd& Xs,
            const Eigen::MatrixXi& F,
            const collision::VertexTriangleContact& c,
            const Eigen::VectorXd& inv_mass,
            int N,
            double thickness,
            double alpha,
            double& lambda)
        {
            const int vi = c.v;
            const int face = c.face;
            if (vi < 0 || vi >= N || face < 0 || face >= F.rows())
                return;

            const int i0 = F(face, 0);
            const int i1 = F(face, 1);
            const int i2 = F(face, 2);
            if (i0 < 0 || i1 < 0 || i2 < 0)
                return;
            if (i0 >= N || i1 >= N || i2 >= N)
                return;
            if (vi == i0 || vi == i1 || vi == i2)
                return;

            const double wv = inv_mass[vi];
            const double w0 = inv_mass[i0];
            const double w1 = inv_mass[i1];
            const double w2 = inv_mass[i2];
            if (!(wv > 0.0) && !(w0 > 0.0) && !(w1 > 0.0) && !(w2 > 0.0))
                return;

            const Eigen::Vector3d a = Xs.row(i0);
            const Eigen::Vector3d b = Xs.row(i1);
            const Eigen::Vector3d ctri = Xs.row(i2);
            const Eigen::Vector3d p = Xs.row(vi);

            Eigen::Vector3d bary;
            const Eigen::Vector3d q = core::closest_point_on_triangle(p, a, b, ctri, &bary);
            const Eigen::Vector3d pq = p - q;
            const double dist = pq.norm();

            Eigen::Vector3d grad = Eigen::Vector3d::Zero();
            if (dist > 1e-12 && std::isfinite(dist))
                grad = pq / dist;
            else
                grad = core::triangle_unit_normal(a, b, ctri);

            if (!(grad.squaredNorm() > 0.0) || !grad.allFinite() || !bary.allFinite())
                return;

            const double C = dist - thickness;
            if (!(C < 0.0) || !std::isfinite(C))
                return;

            const double b0 = std::clamp(bary[0], 0.0, 1.0);
            const double b1 = std::clamp(bary[1], 0.0, 1.0);
            const double b2 = std::clamp(bary[2], 0.0, 1.0);
            const double denom = wv + b0 * b0 * w0 + b1 * b1 * w1 + b2 * b2 * w2 + alpha;
            if (!(denom > kEps))
                return;

            const double dlambda = (-C - alpha * lambda) / denom;
            lambda += dlambda;

            const Eigen::Vector3d corr = dlambda * grad;
            if (wv > 0.0)
                Xs.row(vi) += (wv * corr).transpose();
            if (w0 > 0.0)
                Xs.row(i0) -= (w0 * b0 * corr).transpose();
            if (w1 > 0.0)
                Xs.row(i1) -= (w1 * b1 * corr).transpose();
            if (w2 > 0.0)
                Xs.row(i2) -= (w2 * b2 * corr).transpose();
        }

        static inline void damp_self_collision_velocity(
            Eigen::MatrixXd& Vv,
            const Eigen::MatrixXd& X,
            const Eigen::MatrixXi& F,
            const collision::VertexTriangleContact& c,
            const Eigen::VectorXd& inv_mass,
            int N)
        {
            const int vi = c.v;
            const int face = c.face;
            if (vi < 0 || vi >= N || face < 0 || face >= F.rows())
                return;

            const int i0 = F(face, 0);
            const int i1 = F(face, 1);
            const int i2 = F(face, 2);
            if (i0 < 0 || i1 < 0 || i2 < 0)
                return;
            if (i0 >= N || i1 >= N || i2 >= N)
                return;
            if (vi == i0 || vi == i1 || vi == i2)
                return;

            const Eigen::Vector3d a = X.row(i0);
            const Eigen::Vector3d b = X.row(i1);
            const Eigen::Vector3d ctri = X.row(i2);
            const Eigen::Vector3d p = X.row(vi);

            Eigen::Vector3d bary;
            const Eigen::Vector3d q = core::closest_point_on_triangle(p, a, b, ctri, &bary);
            Eigen::Vector3d n = p - q;
            const double nn = n.norm();
            if (nn > 1e-12 && std::isfinite(nn))
                n /= nn;
            else
                n = core::triangle_unit_normal(a, b, ctri);
            if (!(n.squaredNorm() > 0.0) || !n.allFinite() || !bary.allFinite())
                return;

            const double b0 = std::clamp(bary[0], 0.0, 1.0);
            const double b1 = std::clamp(bary[1], 0.0, 1.0);
            const double b2 = std::clamp(bary[2], 0.0, 1.0);

            const double wv = inv_mass[vi];
            const double w0 = inv_mass[i0];
            const double w1 = inv_mass[i1];
            const double w2 = inv_mass[i2];
            const double denom = wv + b0 * b0 * w0 + b1 * b1 * w1 + b2 * b2 * w2;
            if (!(denom > kEps))
                return;

            const Eigen::Vector3d rel_v =
                Vv.row(vi).transpose()
                - b0 * Vv.row(i0).transpose()
                - b1 * Vv.row(i1).transpose()
                - b2 * Vv.row(i2).transpose();
            const double vn = rel_v.dot(n);
            if (!(vn < 0.0))
                return;

            const double impulse = -vn / denom;
            const Eigen::Vector3d dv = impulse * n;
            if (wv > 0.0)
                Vv.row(vi) += (wv * dv).transpose();
            if (w0 > 0.0)
                Vv.row(i0) -= (w0 * b0 * dv).transpose();
            if (w1 > 0.0)
                Vv.row(i1) -= (w1 * b1 * dv).transpose();
            if (w2 > 0.0)
                Vv.row(i2) -= (w2 * b2 * dv).transpose();
        }
    }

    bool run_cloth_xpbd(
        const Eigen::MatrixXd& cloth_V0,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const ClothXpbdParams& params,
        ClothXpbdResult* out,
        std::string* error,
        std::atomic_bool* cancel,
        std::atomic_int* progress_steps_done,
        ClothXpbdOnFrameFn on_frame,
        void* on_frame_user)
    {
        if (out)
            *out = ClothXpbdResult{};

        using Clock = std::chrono::steady_clock;
        const auto t_total0 = Clock::now();
        const auto to_ms = [](Clock::duration d)
        {
            return std::chrono::duration<double, std::milli>(d).count();
        };

        double ms_bvh_build = 0.0;
        double ms_predict = 0.0;
        double ms_wind = 0.0;
        double ms_candidates = 0.0;
        double ms_stretch = 0.0;
        double ms_bend = 0.0;

        double ms_collision_bvh = 0.0;
        double ms_collision_narrow = 0.0;
        double ms_collision_broad = 0.0;
        double ms_collision_solve = 0.0;
        double ms_self_collision_bvh = 0.0;
        double ms_self_collision_solve = 0.0;
        double ms_friction = 0.0;

        if (cloth_V0.cols() != 3 || body_V.cols() != 3)
        {
            if (error)
                *error = "Expected 3D vertex positions (#V x 3).";
            return false;
        }
        if (cloth_F.rows() == 0 || cloth_F.cols() != 3)
        {
            if (error)
                *error = "Expected cloth faces to be triangles (#F x 3).";
            return false;
        }
        if (body_F.rows() == 0 || body_F.cols() != 3)
        {
            if (error)
                *error = "Expected body faces to be triangles (#F x 3).";
            return false;
        }
        if (cloth_V0.rows() == 0 || body_V.rows() == 0)
        {
            if (error)
                *error = "Missing cloth/body mesh data.";
            return false;
        }

        const double dt = params.dt;
        if (!(dt > 0.0) || !std::isfinite(dt))
        {
            if (error)
                *error = "dt must be finite and > 0.";
            return false;
        }
        if (!(params.duration_seconds >= 0.0) || !std::isfinite(params.duration_seconds))
        {
            if (error)
                *error = "duration_seconds must be finite and >= 0.";
            return false;
        }

        const int N = (int)cloth_V0.rows();
        const int max_frames = std::max(1, params.max_frames);
        const int steps_target = (params.duration_seconds <= 0.0) ? 0 : (int)std::ceil(params.duration_seconds / dt);
        const int steps = std::min(steps_target, std::max(0, max_frames - 1));

        if (progress_steps_done)
            progress_steps_done->store(0);

        std::vector<StretchEdge> stretch_edges;
        std::vector<BendEdge> bend_edges;
        build_stretch_edges(cloth_V0, cloth_F, stretch_edges);
        build_bend_edges(cloth_V0, cloth_F, bend_edges);

        Eigen::VectorXd mass;
        if (!compute_lumped_masses(cloth_V0, cloth_F, mass))
        {
            if (error)
                *error = "Failed to compute masses.";
            return false;
        }

        Eigen::VectorXd inv_mass = Eigen::VectorXd::Zero(N);
        for (int i = 0; i < N; ++i)
        {
            const double m = mass[i];
            inv_mass[i] = (m > 0.0 && std::isfinite(m)) ? (1.0 / m) : 0.0;
        }

        // Compliance alphas are scaled by sub_dt, not dt, so stiffness is invariant to substep count.
        const int substeps = std::max(1, params.substeps);
        const double sub_dt = dt / (double)substeps;

        const double stretch_alpha = compliance_alpha(params.stretch_compliance, sub_dt);
        const double bend_alpha = compliance_alpha(params.bend_compliance, sub_dt);
        const double col_alpha = compliance_alpha(params.collision_compliance, sub_dt);
        const double self_col_alpha = compliance_alpha(params.self_collision_compliance, sub_dt);

        const double thickness = std::max(0.0, params.collision_thickness);
        const double self_thickness = std::max(0.0, params.self_collision_thickness);

        collision::TriangleBvh bvh;
        // For static body (no sequence) build BVH once; dynamic body rebuilds per step in the loop.
        const Eigen::MatrixXd* p_body_V = &body_V;
        int bvh_body_idx = -1;
        if (params.body_sequence.empty())
        {
            const auto t0 = Clock::now();
            bvh.build(body_V, body_F);
            ms_bvh_build += to_ms(Clock::now() - t0);
        }

        collision::TriangleBvh self_bvh;
        if (params.self_collision_enabled && self_thickness > 0.0)
        {
            const auto t0 = Clock::now();
            self_bvh.build(cloth_V0, cloth_F);
            ms_bvh_build += to_ms(Clock::now() - t0);
        }

        const auto body_for_frame = [&](int frame_index) -> const Eigen::MatrixXd*
        {
            if (params.body_sequence.empty())
                return &body_V;
            const int idx = std::clamp(frame_index, 0, (int)params.body_sequence.size() - 1);
            return &params.body_sequence[(size_t)idx];
        };

        Eigen::MatrixXd X = cloth_V0;
        Eigen::MatrixXd Vv = Eigen::MatrixXd::Zero(N, 3);

        ClothXpbdResult result;
        result.frames.reserve((size_t)(steps + 1));
        result.frames.push_back(X);
        if (on_frame)
            on_frame(result.frames.back(), body_for_frame(result.stats.steps_simulated), result.stats.steps_simulated, on_frame_user);

        std::vector<collision::VertexTriangleContact> contacts;
        std::vector<collision::VertexTriangleContact> self_contacts;

        std::vector<double> lambda_stretch(stretch_edges.size(), 0.0);
        std::vector<double> lambda_bend(bend_edges.size(), 0.0);
        std::vector<double> lambda_col((size_t)N, 0.0);
        std::vector<double> lambda_self;
        std::vector<int> col_candidates;

        for (int step = 0; step < steps; ++step)
        {
            if (cancel && cancel->load())
                break;

            if (!params.body_sequence.empty())
            {
                const int idx = std::min((int)params.body_sequence.size() - 1, step);
                if (idx != bvh_body_idx)
                {
                    p_body_V = &params.body_sequence[(size_t)idx];
                    const auto t0 = Clock::now();
                    bvh.build(*p_body_V, body_F);
                    ms_bvh_build += to_ms(Clock::now() - t0);
                    bvh_body_idx = idx;
                }
            }
            const Eigen::MatrixXd& cur_body_V = *p_body_V;

            for (int sub = 0; sub < substeps; ++sub)
            {
                if (cancel && cancel->load())
                    break;

                const Eigen::MatrixXd X_sub_prev = X;

                Eigen::MatrixXd X_pred;
                {
                    const auto t0 = Clock::now();
                    X_pred = X + sub_dt * Vv;
                    X_pred.col(1).array() += (sub_dt * sub_dt) * params.gravity_y;
                    ms_predict += to_ms(Clock::now() - t0);
                }
                {
                    const auto t0 = Clock::now();
                    // Pass global substep index so wind time variation is continuous.
                    apply_wind_prediction(X_pred, X, step * substeps + sub, sub_dt, params);
                    ms_wind += to_ms(Clock::now() - t0);
                }

                Eigen::MatrixXd Xs = X_pred;

                col_candidates.clear();
                {
                    const auto t0 = Clock::now();
                    if (params.collision_enabled)
                    {
                        const core::Aabb bb = bvh.bounds();
                        const double margin = std::max(0.02, thickness * 4.0);
                        const core::Aabb expanded{ bb.min - Eigen::Vector3d::Constant(margin), bb.max + Eigen::Vector3d::Constant(margin) };

                        col_candidates.reserve((size_t)N);
                        for (int vi = 0; vi < N; ++vi)
                        {
                            const Eigen::Vector3d p = Xs.row(vi);
                            if ((p.array() >= expanded.min.array()).all() && (p.array() <= expanded.max.array()).all())
                                col_candidates.push_back(vi);
                        }
                    }
                    ms_candidates += to_ms(Clock::now() - t0);
                }

                std::fill(lambda_stretch.begin(), lambda_stretch.end(), 0.0);
                std::fill(lambda_bend.begin(), lambda_bend.end(), 0.0);
                std::fill(lambda_col.begin(), lambda_col.end(), 0.0);
                contacts.clear();
                self_contacts.clear();
                lambda_self.clear();

                const int iters = std::clamp(params.solver_iterations, 1, 500);
                const double base_stretch_mult = std::max(0.0, params.base_stretch_mult);

                int last_contact_count = 0;

                for (int it = 0; it < iters; ++it)
                {
                    if (cancel && cancel->load())
                        break;

                    if (params.stretch_enabled && stretch_alpha >= 0.0 && !stretch_edges.empty())
                    {
                        const auto t0 = Clock::now();
                        for (size_t ei = 0; ei < stretch_edges.size(); ++ei)
                        {
                            const StretchEdge& e = stretch_edges[ei];
                            const int i = e.i;
                            const int j = e.j;
                            if (i < 0 || j < 0 || i >= N || j >= N)
                                continue;

                            const double wi = inv_mass[i];
                            const double wj = inv_mass[j];
                            if (!(wi > 0.0) && !(wj > 0.0))
                                continue;

                            const Eigen::Vector3d xi = Xs.row(i);
                            const Eigen::Vector3d xj = Xs.row(j);
                            Eigen::Vector3d d = xi - xj;
                            const double len = d.norm();
                            if (!(len > 1e-20) || !std::isfinite(len))
                                continue;

                            const double rest = base_stretch_mult * e.rest_len;
                            const double C = len - rest;
                            const Eigen::Vector3d n = d / len;

                            const double denom = wi + wj + stretch_alpha;
                            if (!(denom > kEps))
                                continue;

                            double& lam = lambda_stretch[ei];
                            const double dlambda = (-C - stretch_alpha * lam) / denom;
                            lam += dlambda;

                            const Eigen::Vector3d corr = dlambda * n;
                            if (wi > 0.0)
                                Xs.row(i) += (wi * corr).transpose();
                            if (wj > 0.0)
                                Xs.row(j) -= (wj * corr).transpose();
                        }
                        ms_stretch += to_ms(Clock::now() - t0);
                    }

                    if (params.bend_enabled && bend_alpha >= 0.0 && !bend_edges.empty())
                    {
                        const auto t0 = Clock::now();
                        for (size_t bi = 0; bi < bend_edges.size(); ++bi)
                        {
                            const BendEdge& be = bend_edges[bi];
                            const int i = be.k0;
                            const int j = be.k1;
                            if (i < 0 || j < 0 || i >= N || j >= N)
                                continue;

                            const double wi = inv_mass[i];
                            const double wj = inv_mass[j];
                            if (!(wi > 0.0) && !(wj > 0.0))
                                continue;

                            const Eigen::Vector3d xi = Xs.row(i);
                            const Eigen::Vector3d xj = Xs.row(j);
                            Eigen::Vector3d d = xi - xj;
                            const double len = d.norm();
                            if (!(len > 1e-20) || !std::isfinite(len))
                                continue;

                            const double C = len - be.rest_len;
                            const Eigen::Vector3d n = d / len;

                            const double denom = wi + wj + bend_alpha;
                            if (!(denom > kEps))
                                continue;

                            double& lam = lambda_bend[bi];
                            const double dlambda = (-C - bend_alpha * lam) / denom;
                            lam += dlambda;

                            const Eigen::Vector3d corr = dlambda * n;
                            if (wi > 0.0)
                                Xs.row(i) += (wi * corr).transpose();
                            if (wj > 0.0)
                                Xs.row(j) -= (wj * corr).transpose();
                        }
                        ms_bend += to_ms(Clock::now() - t0);
                    }

                    // Refresh BVH contacts every 2 iterations to track moving vertices.
                    last_contact_count = 0;
                    if (params.collision_enabled && col_alpha >= 0.0 && !col_candidates.empty())
                    {
                        constexpr int kRefreshEvery = 2;
                        const bool refresh_contacts = (it % kRefreshEvery) == 0;
                        if (refresh_contacts)
                        {
                            collision::VertexTriangleContactQueryStats qs;
                            (void)collision::find_vertex_triangle_contacts(Xs, cur_body_V, body_F, bvh, thickness, col_candidates, contacts, &qs);
                            ms_collision_bvh += qs.ms_total;
                            ms_collision_narrow += qs.ms_bvh;
                            ms_collision_broad += std::max(0.0, qs.ms_total - qs.ms_bvh);
                        }

                        const auto t0 = Clock::now();
                        for (const auto& c : contacts)
                        {
                            apply_collision_constraint(Xs, cur_body_V, body_F, c, inv_mass, N, thickness, col_alpha, lambda_col);
                            last_contact_count++;
                        }
                        ms_collision_solve += to_ms(Clock::now() - t0);
                    }
                    else
                    {
                        contacts.clear();
                    }

                    last_contact_count = 0;
                    if (params.self_collision_enabled && self_col_alpha >= 0.0 && self_thickness > 0.0)
                    {
                        constexpr int kRefreshEvery = 2;
                        const bool refresh_contacts = (it % kRefreshEvery) == 0;
                        if (refresh_contacts)
                        {
                            const auto t0 = Clock::now();
                            self_bvh.refit(Xs, cloth_F);
                            (void)find_self_collision_contacts(Xs, cloth_F, self_bvh, self_thickness, self_contacts);
                            lambda_self.assign(self_contacts.size(), 0.0);
                            ms_self_collision_bvh += to_ms(Clock::now() - t0);
                        }

                        const auto t0 = Clock::now();
                        for (size_t ci = 0; ci < self_contacts.size(); ++ci)
                            apply_self_collision_constraint(Xs, cloth_F, self_contacts[ci], inv_mass, N, self_thickness, self_col_alpha, lambda_self[ci]);
                        ms_self_collision_solve += to_ms(Clock::now() - t0);
                    }
                    else
                    {
                        self_contacts.clear();
                        lambda_self.clear();
                    }
                }

                // Final collision refresh on converged positions + extra passes for stubborn penetrations.
                if (params.collision_enabled && col_alpha >= 0.0 && !col_candidates.empty())
                {
                    {
                        collision::VertexTriangleContactQueryStats qs;
                        (void)collision::find_vertex_triangle_contacts(Xs, cur_body_V, body_F, bvh, thickness, col_candidates, contacts, &qs);
                        ms_collision_bvh += qs.ms_total;
                        ms_collision_narrow += qs.ms_bvh;
                        ms_collision_broad += std::max(0.0, qs.ms_total - qs.ms_bvh);
                    }

                    result.stats.last_step_contacts = (int)contacts.size();

                    // Extra passes to clear remaining penetrations (e.g. in concave regions like armpits).
                    const int post_passes = 4;
                    for (int pass = 0; pass < post_passes; ++pass)
                    {
                        const auto t0 = Clock::now();
                        for (const auto& c : contacts)
                            apply_collision_constraint(Xs, cur_body_V, body_F, c, inv_mass, N, thickness, col_alpha, lambda_col);
                        ms_collision_solve += to_ms(Clock::now() - t0);
                    }
                }
                else
                {
                    contacts.clear();
                    result.stats.last_step_contacts = 0;
                }

                if (params.self_collision_enabled && self_col_alpha >= 0.0 && self_thickness > 0.0)
                {
                    {
                        const auto t0 = Clock::now();
                        self_bvh.refit(Xs, cloth_F);
                        (void)find_self_collision_contacts(Xs, cloth_F, self_bvh, self_thickness, self_contacts);
                        lambda_self.assign(self_contacts.size(), 0.0);
                        ms_self_collision_bvh += to_ms(Clock::now() - t0);
                    }

                    result.stats.last_step_self_contacts = (int)self_contacts.size();

                    const int post_passes = 2;
                    for (int pass = 0; pass < post_passes; ++pass)
                    {
                        const auto t0 = Clock::now();
                        for (size_t ci = 0; ci < self_contacts.size(); ++ci)
                            apply_self_collision_constraint(Xs, cloth_F, self_contacts[ci], inv_mass, N, self_thickness, self_col_alpha, lambda_self[ci]);
                        ms_self_collision_solve += to_ms(Clock::now() - t0);
                    }
                }
                else
                {
                    self_contacts.clear();
                    result.stats.last_step_self_contacts = 0;
                }

                X = Xs;

                Vv = (X - X_sub_prev) / sub_dt;

                const double damp = std::clamp(params.damping, 0.0, 0.99);
                Vv *= (1.0 - damp);

                // Zero inward normal velocity at contacts to prevent tunneling on subsequent substeps.
                if (params.collision_enabled && !contacts.empty())
                {
                    for (const auto& cc : contacts)
                    {
                        const int vi = cc.v;
                        const int face = cc.face;
                        if (vi < 0 || vi >= N || face < 0 || face >= body_F.rows())
                            continue;

                        const int i0 = body_F(face, 0);
                        const int i1 = body_F(face, 1);
                        const int i2 = body_F(face, 2);
                        if (i0 < 0 || i1 < 0 || i2 < 0)
                            continue;
                        if (i0 >= cur_body_V.rows() || i1 >= cur_body_V.rows() || i2 >= cur_body_V.rows())
                            continue;

                        const Eigen::Vector3d n = core::triangle_unit_normal(
                            cur_body_V.row(i0).transpose(),
                            cur_body_V.row(i1).transpose(),
                            cur_body_V.row(i2).transpose());
                        if (!(n.squaredNorm() > 0.0))
                            continue;

                        const double vn = Vv.row(vi).dot(n);
                        if (vn < 0.0)
                            Vv.row(vi) -= (vn * n).transpose();
                    }
                }

                if (params.self_collision_enabled && !self_contacts.empty())
                {
                    for (const auto& cc : self_contacts)
                        damp_self_collision_velocity(Vv, X, cloth_F, cc, inv_mass, N);
                }

                if (params.enable_friction && params.friction_mu > 0.0 && !contacts.empty())
                {
                    const auto t0 = Clock::now();
                    const double mu = std::max(0.0, params.friction_mu);
                    for (const auto& cc : contacts)
                    {
                        const int vi = cc.v;
                        const int face = cc.face;
                        if (vi < 0 || vi >= N)
                            continue;
                        if (face < 0 || face >= body_F.rows())
                            continue;

                        const int i0 = body_F(face, 0);
                        const int i1 = body_F(face, 1);
                        const int i2 = body_F(face, 2);
                        if (i0 < 0 || i1 < 0 || i2 < 0)
                            continue;
                        if (i0 >= cur_body_V.rows() || i1 >= cur_body_V.rows() || i2 >= cur_body_V.rows())
                            continue;

                        const Eigen::Vector3d n = core::triangle_unit_normal(
                            cur_body_V.row(i0).transpose(),
                            cur_body_V.row(i1).transpose(),
                            cur_body_V.row(i2).transpose());
                        if (!(n.squaredNorm() > 0.0))
                            continue;

                        const Eigen::Vector3d v = Vv.row(vi);
                        const double vn_mag = v.dot(n);
                        const Eigen::Vector3d v_n = vn_mag * n;
                        const Eigen::Vector3d v_t = v - v_n;
                        const double vt = v_t.norm();
                        if (!(vt > 1e-12))
                            continue;

                        const double max_t = std::max(0.0, vt - mu * std::abs(vn_mag));
                        const Eigen::Vector3d v_t_new = (max_t / vt) * v_t;
                        const Eigen::Vector3d v_new = v_n + v_t_new;
                        Vv.row(vi) = v_new.transpose();
                    }

                    X = X_sub_prev + sub_dt * Vv;
                    ms_friction += to_ms(Clock::now() - t0);

                    if (params.collision_enabled && col_alpha >= 0.0 && !col_candidates.empty())
                    {
                        {
                            collision::VertexTriangleContactQueryStats qs;
                            (void)collision::find_vertex_triangle_contacts(X, cur_body_V, body_F, bvh, thickness, col_candidates, contacts, &qs);
                            ms_collision_bvh += qs.ms_total;
                            ms_collision_narrow += qs.ms_bvh;
                            ms_collision_broad += std::max(0.0, qs.ms_total - qs.ms_bvh);
                        }
                        if (!contacts.empty())
                        {
                            const auto t1 = Clock::now();
                            for (const auto& c : contacts)
                                apply_collision_constraint(X, cur_body_V, body_F, c, inv_mass, N, thickness, col_alpha, lambda_col);
                            ms_collision_solve += to_ms(Clock::now() - t1);
                        }
                        Vv = (X - X_sub_prev) / sub_dt;
                    }

                    if (params.self_collision_enabled && self_col_alpha >= 0.0 && self_thickness > 0.0)
                    {
                        {
                            const auto t1 = Clock::now();
                            self_bvh.refit(X, cloth_F);
                            (void)find_self_collision_contacts(X, cloth_F, self_bvh, self_thickness, self_contacts);
                            lambda_self.assign(self_contacts.size(), 0.0);
                            ms_self_collision_bvh += to_ms(Clock::now() - t1);
                        }
                        if (!self_contacts.empty())
                        {
                            const auto t1 = Clock::now();
                            for (size_t ci = 0; ci < self_contacts.size(); ++ci)
                                apply_self_collision_constraint(X, cloth_F, self_contacts[ci], inv_mass, N, self_thickness, self_col_alpha, lambda_self[ci]);
                            ms_self_collision_solve += to_ms(Clock::now() - t1);
                            Vv = (X - X_sub_prev) / sub_dt;
                        }
                    }
                }
            } // end substep loop

            result.frames.push_back(X);
            result.stats.steps_simulated++;

            if (on_frame)
                on_frame(result.frames.back(), body_for_frame(result.stats.steps_simulated), result.stats.steps_simulated, on_frame_user);

            if (progress_steps_done)
                progress_steps_done->store(result.stats.steps_simulated);

            if ((int)result.frames.size() >= max_frames)
                break;
        }

        result.stats.frames_recorded = (int)result.frames.size();

        result.stats.ms_bvh_build = ms_bvh_build;
        result.stats.ms_predict = ms_predict;
        result.stats.ms_wind = ms_wind;
        result.stats.ms_candidates = ms_candidates;
        result.stats.ms_stretch = ms_stretch;
        result.stats.ms_bend = ms_bend;
        result.stats.ms_collision_bvh = ms_collision_bvh;
        result.stats.ms_collision_narrow = ms_collision_narrow;
        result.stats.ms_collision_broad = ms_collision_broad;
        result.stats.ms_collision_solve = ms_collision_solve;
        result.stats.ms_self_collision_bvh = ms_self_collision_bvh;
        result.stats.ms_self_collision_solve = ms_self_collision_solve;
        result.stats.ms_friction = ms_friction;
        result.stats.ms_total = to_ms(Clock::now() - t_total0);

        if (out)
            *out = std::move(result);

        return true;
    }
}
