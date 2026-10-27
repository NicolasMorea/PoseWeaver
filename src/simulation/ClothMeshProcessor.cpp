#include "simulation/ClothMeshProcessor.h"

#include <igl/signed_distance.h>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace poseweaver
{
    namespace
    {
        static inline long long edge_key(const int a, const int b)
        {
            const int lo = std::min(a, b);
            const int hi = std::max(a, b);
            return (static_cast<long long>(lo) << 32) | static_cast<unsigned int>(hi);
        }

        static inline Eigen::RowVector3d rotate_around_axis(
            const Eigen::RowVector3d& p,
            const Eigen::RowVector3d& axis_point,
            const Eigen::RowVector3d& axis_dir_unit,
            const double angle_rad)
        {
            // Rodrigues' rotation formula around an axis passing through axis_point.
            const Eigen::RowVector3d v = p - axis_point;
            const double c = std::cos(angle_rad);
            const double s = std::sin(angle_rad);
            const Eigen::RowVector3d k = axis_dir_unit;
            const Eigen::RowVector3d v_rot = v * c + (k.cross(v)) * s + k * (k.dot(v)) * (1.0 - c);
            return axis_point + v_rot;
        }

        static inline double wrap_angle_pi(const double a)
        {
            constexpr double kPi = 3.14159265358979323846264338327950288;
            double x = std::fmod(a + kPi, 2.0 * kPi);
            if (x < 0.0)
                x += 2.0 * kPi;
            return x - kPi;
        }

        struct EdgeFace
        {
            int v0 = -1;
            int v1 = -1;
            int opp = -1;
            int face = -1;
        };
    }

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

        constexpr int kSmoothIterations = 6;
        constexpr double kNeighborTotalFrac = 0.35;
        constexpr double kSmoothAlpha = 0.65;

        std::vector<std::vector<int>> adj;
        adj.resize((size_t)P.rows());
        if (cloth_F.rows() > 0 && cloth_F.cols() == 3)
        {
            for (int fi = 0; fi < cloth_F.rows(); ++fi)
            {
                const int a = cloth_F(fi, 0);
                const int b = cloth_F(fi, 1);
                const int c = cloth_F(fi, 2);
                if (a < 0 || b < 0 || c < 0)
                    continue;
                if (a >= P.rows() || b >= P.rows() || c >= P.rows())
                    continue;

                adj[(size_t)a].push_back(b);
                adj[(size_t)a].push_back(c);
                adj[(size_t)b].push_back(a);
                adj[(size_t)b].push_back(c);
                adj[(size_t)c].push_back(a);
                adj[(size_t)c].push_back(b);
            }
            for (auto& v : adj)
            {
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
            }
        }

        Eigen::MatrixXd moved_delta = Eigen::MatrixXd::Zero(P.rows(), 3);
        Eigen::VectorXd push_amount = Eigen::VectorXd::Zero(P.rows());
        std::vector<unsigned char> is_source((size_t)P.rows(), 0);

        Eigen::MatrixXd out_dir = Eigen::MatrixXd::Zero(P.rows(), 3);

        int moved = 0;
        for (int vi = 0; vi < P.rows(); ++vi)
        {
            const double sd = S[vi];
            if (!std::isfinite(sd))
                continue;

            Eigen::RowVector3d dir;
            double n = 0.0;

            if (sd >= 0.0)
            {
                dir = (P.row(vi) - C.row(vi));
                n = dir.norm();
            }
            else
            {
                dir = (C.row(vi) - P.row(vi));
                n = dir.norm();
            }

            if (!(n > 1e-12) || !std::isfinite(n))
            {
                dir = (P.row(vi) - body_centroid);
                n = dir.norm();
                if (!(n > 1e-12) || !std::isfinite(n))
                    continue;
            }
            dir /= n;

            out_dir.row(vi) = dir;

            if (!(sd < surface_offset))
                continue;

            const double delta = std::max(0.0, surface_offset - sd);
            const Eigen::RowVector3d d = delta * dir;
            if (!d.allFinite())
                continue;

            moved_delta.row(vi) = d;
            push_amount[vi] = delta;
            is_source[(size_t)vi] = 1;
            moved++;
        }

        Eigen::MatrixXd V_out = P + moved_delta;

        // Diffuse push-amount over the mesh graph to avoid contour-stacking artifacts.
        if (kSmoothIterations > 0 && kNeighborTotalFrac > 0.0 && kSmoothAlpha > 0.0 && !adj.empty())
        {
            Eigen::VectorXd seed = Eigen::VectorXd::Zero(P.rows());
            for (int vi = 0; vi < P.rows(); ++vi)
            {
                if (!is_source[(size_t)vi])
                    continue;
                const double a = push_amount[vi];
                if (!(a > 0.0) || !std::isfinite(a))
                    continue;
                seed[vi] = kNeighborTotalFrac * a;
            }

            Eigen::VectorXd g = seed;
            Eigen::VectorXd g_next = g;

            const int iters = std::clamp(kSmoothIterations, 1, 50);
            const double alpha = std::clamp(kSmoothAlpha, 0.0, 1.0);

            for (int it = 0; it < iters; ++it)
            {
                g_next = g;
                for (int vi = 0; vi < P.rows(); ++vi)
                {
                    if (is_source[(size_t)vi])
                        continue;

                    const auto& nb = adj[(size_t)vi];
                    if (nb.empty())
                    {
                        g_next[vi] = 0.0;
                        continue;
                    }

                    double sum = 0.0;
                    int cnt = 0;
                    for (const int j : nb)
                    {
                        if (j < 0 || j >= P.rows())
                            continue;
                        const double v = g[j];
                        if (!std::isfinite(v))
                            continue;
                        sum += v;
                        cnt++;
                    }

                    if (cnt <= 0)
                    {
                        g_next[vi] = 0.0;
                        continue;
                    }

                    const double avg = sum / (double)cnt;
                    g_next[vi] = alpha * avg;
                }

                for (int vi = 0; vi < P.rows(); ++vi)
                {
                    if (is_source[(size_t)vi])
                        g_next[vi] = seed[vi];
                }

                g.swap(g_next);
            }

            for (int vi = 0; vi < P.rows(); ++vi)
            {
                if (is_source[(size_t)vi])
                    continue;
                const double a = g[vi];
                if (!(a > 0.0) || !std::isfinite(a))
                    continue;
                const Eigen::RowVector3d dir = out_dir.row(vi);
                if (!(dir.squaredNorm() > 0.0) || !dir.allFinite())
                    continue;

                const Eigen::RowVector3d d = a * dir;
                if (!d.allFinite())
                    continue;
                V_out.row(vi) += d;
            }
        }

        cloth_V = std::move(V_out);

        if (result)
            result->moved_vertices = moved;
        return true;
    }

    bool ease_cloth_toward_rest_pose(
        Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& cloth_V_rest,
        const double amount,
        ClothEaseResult* result,
        std::string* error)
    {
        if (result)
            *result = ClothEaseResult{};

        if (cloth_V.cols() != 3 || cloth_V_rest.cols() != 3)
        {
            if (error)
                *error = "Expected 3D vertex positions (#V x 3).";
            return false;
        }
        if (cloth_V.rows() == 0 || cloth_F.rows() == 0)
        {
            if (error)
                *error = "Missing cloth mesh data.";
            return false;
        }
        if (cloth_V_rest.rows() != cloth_V.rows())
        {
            if (error)
                *error = "cloth_V_rest must have the same number of vertices as cloth_V.";
            return false;
        }
        if (cloth_F.cols() != 3)
        {
            if (error)
                *error = "Expected cloth faces to be triangles (#F x 3).";
            return false;
        }
        if (!(amount >= 0.0) || !std::isfinite(amount))
        {
            if (error)
                *error = "amount must be a finite non-negative value.";
            return false;
        }
        if (amount == 0.0)
            return true;

        const double step = std::min(1.0, amount);
        constexpr double kEps = 1e-12;

        // Build unique edges + edge->incident faces (for bending).
        struct EdgeInfo
        {
            int a = -1;
            int b = -1;
            double rest_len = 0.0;
            EdgeFace f[2];
            int face_count = 0;
        };

        std::unordered_map<long long, EdgeInfo> edges;
        edges.reserve((size_t)cloth_F.rows() * 2u);

        for (int fi = 0; fi < cloth_F.rows(); ++fi)
        {
            const int a = cloth_F(fi, 0);
            const int b = cloth_F(fi, 1);
            const int c = cloth_F(fi, 2);
            if (a < 0 || b < 0 || c < 0)
                continue;
            if (a >= cloth_V.rows() || b >= cloth_V.rows() || c >= cloth_V.rows())
                continue;

            const int v[3] = { a, b, c };
            const int opp[3] = { c, a, b }; // opposite for edges (a,b)->c, (b,c)->a, (c,a)->b

            for (int ei = 0; ei < 3; ++ei)
            {
                const int v0 = v[ei];
                const int v1 = v[(ei + 1) % 3];
                const int vo = opp[ei];

                const long long key = edge_key(v0, v1);
                auto it = edges.find(key);
                if (it == edges.end())
                {
                    EdgeInfo info;
                    info.a = std::min(v0, v1);
                    info.b = std::max(v0, v1);
                    info.rest_len = (cloth_V_rest.row(info.b) - cloth_V_rest.row(info.a)).norm();
                    info.face_count = 0;
                    edges.emplace(key, info);
                    it = edges.find(key);
                }

                if (it->second.face_count < 2)
                {
                    EdgeFace ef;
                    ef.v0 = v0;
                    ef.v1 = v1;
                    ef.opp = vo;
                    ef.face = fi;
                    it->second.f[it->second.face_count++] = ef;
                }
            }
        }

        Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(cloth_V.rows(), 3);
        int affected_edges = 0;

        for (const auto& kv : edges)
        {
            const EdgeInfo& e = kv.second;
            const int i = e.a;
            const int j = e.b;
            if (i < 0 || j < 0)
                continue;

            const Eigen::RowVector3d pi = cloth_V.row(i);
            const Eigen::RowVector3d pj = cloth_V.row(j);
            const Eigen::RowVector3d d = pj - pi;
            const double len = d.norm();
            const double rest = e.rest_len;

            if (!(len > kEps) || !(rest > kEps) || !std::isfinite(len) || !std::isfinite(rest))
                continue;

            const double diff = (len - rest);
            if (!std::isfinite(diff))
                continue;

            const Eigen::RowVector3d corr = (diff / len) * d;
            if (!corr.allFinite())
                continue;

            delta.row(i) += 0.5 * corr;
            delta.row(j) -= 0.5 * corr;
            affected_edges++;
        }

        constexpr double kStretch = 1.0;
        Eigen::MatrixXd V_new = cloth_V + (step * kStretch) * delta;

        int affected_bend = 0;
        constexpr double kBend = 0.5;

        for (const auto& kv : edges)
        {
            const EdgeInfo& e = kv.second;
            if (e.face_count != 2)
                continue;

            // Choose an orientation from face 0: (i,j,opp0)
            const int i = e.f[0].v0;
            const int j = e.f[0].v1;
            const int k0 = e.f[0].opp;
            const int k1 = e.f[1].opp;

            if (i < 0 || j < 0 || k0 < 0 || k1 < 0)
                continue;
            if (i >= V_new.rows() || j >= V_new.rows() || k0 >= V_new.rows() || k1 >= V_new.rows())
                continue;

            const Eigen::RowVector3d pi = V_new.row(i);
            const Eigen::RowVector3d pj = V_new.row(j);
            Eigen::RowVector3d u = pj - pi;
            const double ulen = u.norm();
            if (!(ulen > kEps) || !std::isfinite(ulen))
                continue;
            u /= ulen;

            auto face_normal = [](const Eigen::RowVector3d& a, const Eigen::RowVector3d& b, const Eigen::RowVector3d& c) -> Eigen::RowVector3d
            {
                Eigen::RowVector3d n = (b - a).cross(c - a);
                const double nn = n.norm();
                if (!(nn > 1e-20) || !std::isfinite(nn))
                    return Eigen::RowVector3d(0.0, 0.0, 0.0);
                n /= nn;
                return n;
            };

            const Eigen::RowVector3d n0 = face_normal(pi, pj, V_new.row(k0));
            const Eigen::RowVector3d n1 = face_normal(pj, pi, V_new.row(k1));
            if (!(n0.squaredNorm() > 0.0) || !(n1.squaredNorm() > 0.0))
                continue;

            const double dot01 = std::clamp(n0.dot(n1), -1.0, 1.0);
            const double s01 = u.dot(n0.cross(n1));
            const double theta = std::atan2(s01, dot01);
            if (!std::isfinite(theta))
                continue;

            const Eigen::RowVector3d ri = cloth_V_rest.row(i);
            const Eigen::RowVector3d rj = cloth_V_rest.row(j);
            const Eigen::RowVector3d rk0 = cloth_V_rest.row(k0);
            const Eigen::RowVector3d rk1 = cloth_V_rest.row(k1);

            Eigen::RowVector3d ru = rj - ri;
            const double rulen = ru.norm();
            if (!(rulen > kEps) || !std::isfinite(rulen))
                continue;
            ru /= rulen;

            const Eigen::RowVector3d rn0 = face_normal(ri, rj, rk0);
            const Eigen::RowVector3d rn1 = face_normal(rj, ri, rk1);
            if (!(rn0.squaredNorm() > 0.0) || !(rn1.squaredNorm() > 0.0))
                continue;

            const double rdot01 = std::clamp(rn0.dot(rn1), -1.0, 1.0);
            const double rs01 = ru.dot(rn0.cross(rn1));
            const double theta0 = std::atan2(rs01, rdot01);
            if (!std::isfinite(theta0))
                continue;

            double err = wrap_angle_pi(theta - theta0);
            if (!std::isfinite(err))
                continue;

            const double dtheta = (step * kBend) * err;
            if (std::abs(dtheta) < 1e-12)
                continue;

            const Eigen::RowVector3d pk0 = V_new.row(k0);
            const Eigen::RowVector3d pk1 = V_new.row(k1);

            const Eigen::RowVector3d pk0r = rotate_around_axis(pk0, pi, u, -0.5 * dtheta);
            const Eigen::RowVector3d pk1r = rotate_around_axis(pk1, pi, u, +0.5 * dtheta);
            if (!pk0r.allFinite() || !pk1r.allFinite())
                continue;

            V_new.row(k0) = pk0r;
            V_new.row(k1) = pk1r;
            affected_bend++;
        }

        double max_disp = 0.0;
        for (int vi = 0; vi < cloth_V.rows(); ++vi)
        {
            const double d = (V_new.row(vi) - cloth_V.row(vi)).norm();
            if (std::isfinite(d))
                max_disp = std::max(max_disp, d);
        }

        cloth_V = std::move(V_new);

        if (result)
        {
            result->affected_edges = affected_edges;
            result->affected_bend_edges = affected_bend;
            result->max_vertex_displacement = max_disp;
        }
        return true;
    }

    bool relax_cloth_preserve_stretch(
        Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& cloth_V_rest,
        int iterations,
        const double edge_alpha,
        const double smooth_lambda,
        ClothRelaxResult* result,
        std::string* error)
    {
        if (result)
            *result = ClothRelaxResult{};

        if (cloth_V.cols() != 3 || cloth_V_rest.cols() != 3)
        {
            if (error)
                *error = "Expected 3D vertex positions (#V x 3).";
            return false;
        }
        if (cloth_V.rows() == 0 || cloth_F.rows() == 0)
        {
            if (error)
                *error = "Missing cloth mesh data.";
            return false;
        }
        if (cloth_V_rest.rows() != cloth_V.rows())
        {
            if (error)
                *error = "cloth_V_rest must have the same number of vertices as cloth_V.";
            return false;
        }
        if (cloth_F.cols() != 3)
        {
            if (error)
                *error = "Expected cloth faces to be triangles (#F x 3).";
            return false;
        }
        if (iterations <= 0)
            return true;
        if (!(edge_alpha >= 0.0) || !std::isfinite(edge_alpha))
        {
            if (error)
                *error = "edge_alpha must be a finite non-negative value.";
            return false;
        }
        if (!(smooth_lambda >= 0.0) || !std::isfinite(smooth_lambda))
        {
            if (error)
                *error = "smooth_lambda must be a finite non-negative value.";
            return false;
        }

        struct Edge
        {
            int i = -1;
            int j = -1;
            double rest_len = 0.0;
        };

        std::vector<Edge> edges;
        edges.reserve((size_t)cloth_F.rows() * 2u);

        std::unordered_map<long long, int> seen;
        seen.reserve((size_t)cloth_F.rows() * 2u);

        std::vector<std::vector<int>> adj;
        adj.resize((size_t)cloth_V.rows());

        auto add_edge = [&](const int a, const int b)
        {
            if (a < 0 || b < 0)
                return;
            if (a >= cloth_V.rows() || b >= cloth_V.rows())
                return;

            const long long key = edge_key(a, b);
            if (seen.find(key) != seen.end())
                return;
            seen.emplace(key, 1);

            const int i = std::min(a, b);
            const int j = std::max(a, b);
            Edge e;
            e.i = i;
            e.j = j;
            e.rest_len = (cloth_V_rest.row(j) - cloth_V_rest.row(i)).norm();
            edges.push_back(e);
        };

        for (int fi = 0; fi < cloth_F.rows(); ++fi)
        {
            const int a = cloth_F(fi, 0);
            const int b = cloth_F(fi, 1);
            const int c = cloth_F(fi, 2);
            if (a < 0 || b < 0 || c < 0)
                continue;
            if (a >= cloth_V.rows() || b >= cloth_V.rows() || c >= cloth_V.rows())
                continue;

            add_edge(a, b);
            add_edge(b, c);
            add_edge(c, a);

            adj[(size_t)a].push_back(b);
            adj[(size_t)a].push_back(c);
            adj[(size_t)b].push_back(a);
            adj[(size_t)b].push_back(c);
            adj[(size_t)c].push_back(a);
            adj[(size_t)c].push_back(b);
        }

        for (auto& nb : adj)
        {
            std::sort(nb.begin(), nb.end());
            nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        }

        constexpr double kEps = 1e-12;

        const Eigen::MatrixXd V_start = cloth_V;
        int edges_applied_total = 0;

        const int iters = std::clamp(iterations, 1, 1000);
        for (int it = 0; it < iters; ++it)
        {
            auto apply_edge_pass = [&](const double alpha)
            {
                if (!(alpha > 0.0))
                    return;

                Eigen::MatrixXd dV = Eigen::MatrixXd::Zero(cloth_V.rows(), 3);
                std::vector<int> counts((size_t)cloth_V.rows(), 0);

                for (const Edge& e : edges)
                {
                    const int i = e.i;
                    const int j = e.j;
                    if (i < 0 || j < 0)
                        continue;

                    const Eigen::RowVector3d d = cloth_V.row(j) - cloth_V.row(i);
                    const double len = d.norm();
                    const double rest = e.rest_len;
                    if (!(len > kEps) || !(rest > kEps) || !std::isfinite(len) || !std::isfinite(rest))
                        continue;

                    const Eigen::RowVector3d corr = (0.5 * alpha * (len - rest) / len) * d;
                    if (!corr.allFinite())
                        continue;

                    dV.row(i) += corr;
                    dV.row(j) -= corr;
                    counts[(size_t)i]++;
                    counts[(size_t)j]++;
                    edges_applied_total++;
                }

                for (int vi = 0; vi < cloth_V.rows(); ++vi)
                {
                    const int c = counts[(size_t)vi];
                    if (c <= 0)
                        continue;
                    cloth_V.row(vi) += (1.0 / (double)c) * dV.row(vi);
                }
            };

            apply_edge_pass(edge_alpha);

            if (smooth_lambda > 0.0)
            {
                Eigen::MatrixXd V_new = cloth_V;
                for (int vi = 0; vi < cloth_V.rows(); ++vi)
                {
                    const auto& nb = adj[(size_t)vi];
                    if (nb.empty())
                        continue;

                    Eigen::RowVector3d avg(0.0, 0.0, 0.0);
                    for (const int j : nb)
                        avg += cloth_V.row(j);
                    avg *= (1.0 / (double)nb.size());

                    V_new.row(vi) = cloth_V.row(vi) + smooth_lambda * (avg - cloth_V.row(vi));
                }
                cloth_V = std::move(V_new);
            }

            apply_edge_pass(0.5 * edge_alpha); // counteracts smoothing shrinkage
        }

        double max_disp = 0.0;
        for (int vi = 0; vi < cloth_V.rows(); ++vi)
        {
            const double d = (cloth_V.row(vi) - V_start.row(vi)).norm();
            if (std::isfinite(d))
                max_disp = std::max(max_disp, d);
        }

        if (result)
        {
            result->iterations_done = iters;
            result->edges_applied = edges_applied_total;
            result->max_vertex_displacement = max_disp;
        }
        return true;
    }

    bool refine_cloth_mesh_for_render(
        const Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const double target_edge_length,
        Eigen::MatrixXd& out_V,
        Eigen::MatrixXi& out_F,
        ClothRenderRefineResult* result,
        std::string* error,
        const Eigen::MatrixXd* cloth_UV,
        Eigen::MatrixXd* out_UV,
        Eigen::VectorXi* out_source_faces)
    {
        if (result)
            *result = ClothRenderRefineResult{};
        out_V.resize(0, 3);
        out_F.resize(0, 3);
        if (out_UV)
            out_UV->resize(0, 2);
        if (out_source_faces)
            out_source_faces->resize(0);

        if (cloth_V.cols() != 3 || cloth_F.cols() != 3 || cloth_V.rows() <= 0 || cloth_F.rows() <= 0)
        {
            if (error)
                *error = "Expected a non-empty triangular cloth mesh.";
            return false;
        }
        if (!(target_edge_length > 0.0) || !std::isfinite(target_edge_length))
        {
            if (error)
                *error = "Target detail size must be a finite positive value.";
            return false;
        }
        if (!cloth_V.allFinite())
        {
            if (error)
                *error = "Cloth vertices contain non-finite values.";
            return false;
        }
        if (cloth_F.minCoeff() < 0 || cloth_F.maxCoeff() >= cloth_V.rows())
        {
            if (error)
                *error = "Cloth faces contain invalid vertex indices.";
            return false;
        }
        const bool refine_uv = cloth_UV != nullptr && out_UV != nullptr;
        if ((cloth_UV == nullptr) != (out_UV == nullptr)
            || (refine_uv && (cloth_UV->rows() != cloth_V.rows() || cloth_UV->cols() < 2 || !cloth_UV->allFinite())))
        {
            if (error)
                *error = "Render UVs must be finite and aligned with cloth vertices.";
            return false;
        }

        constexpr int kMaxPasses = 10;
        constexpr int kMaxVertices = 2000000;
        constexpr int kMaxFaces = 2000000;
        constexpr double kSplitTolerance = 1.001;

        Eigen::MatrixXd V = cloth_V;
        Eigen::MatrixXi F = cloth_F;
        Eigen::MatrixXd UV = refine_uv ? cloth_UV->leftCols(2).eval() : Eigen::MatrixXd{};
        Eigen::VectorXi source_faces = Eigen::VectorXi::LinSpaced(F.rows(), 0, F.rows() - 1);

        const auto max_edge_length = [](const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces)
        {
            double max_len = 0.0;
            for (int fi = 0; fi < faces.rows(); ++fi)
            {
                const int ids[3] = { faces(fi, 0), faces(fi, 1), faces(fi, 2) };
                for (int ei = 0; ei < 3; ++ei)
                {
                    const double len = (vertices.row(ids[(ei + 1) % 3]) - vertices.row(ids[ei])).norm();
                    if (std::isfinite(len))
                        max_len = std::max(max_len, len);
                }
            }
            return max_len;
        };

        const double max_before = max_edge_length(V, F);
        int passes = 0;
        int added_vertices = 0;

        for (; passes < kMaxPasses; ++passes)
        {
            Eigen::MatrixXd normals = Eigen::MatrixXd::Zero(V.rows(), 3);
            for (int fi = 0; fi < F.rows(); ++fi)
            {
                const int a = F(fi, 0);
                const int b = F(fi, 1);
                const int c = F(fi, 2);
                const Eigen::RowVector3d ab = V.row(b) - V.row(a);
                const Eigen::RowVector3d ac = V.row(c) - V.row(a);
                const Eigen::RowVector3d n = ab.cross(ac);
                if (!n.allFinite() || n.squaredNorm() <= 1e-24)
                    continue;
                normals.row(a) += n;
                normals.row(b) += n;
                normals.row(c) += n;
            }
            for (int vi = 0; vi < normals.rows(); ++vi)
            {
                const double len = normals.row(vi).norm();
                if (len > 1e-12 && std::isfinite(len))
                    normals.row(vi) /= len;
            }

            std::unordered_map<long long, int> midpoint_index;
            midpoint_index.reserve((size_t)F.rows());

            std::vector<Eigen::RowVector3d> vertices;
            vertices.reserve((size_t)V.rows() + (size_t)F.rows());
            for (int vi = 0; vi < V.rows(); ++vi)
                vertices.push_back(V.row(vi));
            std::vector<Eigen::RowVector2d> texture_coordinates;
            if (refine_uv)
            {
                texture_coordinates.reserve((size_t)V.rows() + (size_t)F.rows());
                for (int vi = 0; vi < UV.rows(); ++vi)
                    texture_coordinates.push_back(UV.row(vi));
            }

            const auto curved_midpoint = [&](const int i, const int j)
            {
                const Eigen::RowVector3d p0 = V.row(i);
                const Eigen::RowVector3d p1 = V.row(j);
                const Eigen::RowVector3d linear = 0.5 * (p0 + p1);
                Eigen::RowVector3d n0 = normals.row(i);
                Eigen::RowVector3d n1 = normals.row(j);
                if (n0.squaredNorm() <= 1e-20 || n1.squaredNorm() <= 1e-20)
                    return linear;

                const Eigen::RowVector3d d = p1 - p0;
                const Eigen::RowVector3d b1 = (2.0 * p0 + p1 - d.dot(n0) * n0) / 3.0;
                const Eigen::RowVector3d b2 = (2.0 * p1 + p0 + d.dot(n1) * n1) / 3.0;
                Eigen::RowVector3d curved = 0.125 * p0 + 0.375 * b1 + 0.375 * b2 + 0.125 * p1;

                const double edge_len = d.norm();
                Eigen::RowVector3d offset = curved - linear;
                const double offset_len = offset.norm();
                const double max_offset = 0.25 * edge_len;
                if (offset_len > max_offset && offset_len > 1e-12)
                    curved = linear + offset * (max_offset / offset_len);
                return curved.allFinite() ? curved : linear;
            };

            const auto get_midpoint = [&](const int i, const int j)
            {
                const long long key = edge_key(i, j);
                const auto found = midpoint_index.find(key);
                if (found != midpoint_index.end())
                    return found->second;

                const int index = (int)vertices.size();
                vertices.push_back(curved_midpoint(i, j));
                if (refine_uv)
                    texture_coordinates.push_back(0.5 * (UV.row(i) + UV.row(j)));
                midpoint_index.emplace(key, index);
                return index;
            };

            std::vector<Eigen::Vector3i> faces;
            faces.reserve((size_t)F.rows() * 2u);
            std::vector<int> next_source_faces;
            next_source_faces.reserve((size_t)F.rows() * 2u);
            bool split_any = false;

            for (int fi = 0; fi < F.rows(); ++fi)
            {
                const auto add_face = [&](const int i, const int j, const int k)
                {
                    faces.emplace_back(i, j, k);
                    next_source_faces.push_back(source_faces(fi));
                };
                const int a = F(fi, 0);
                const int b = F(fi, 1);
                const int c = F(fi, 2);
                const bool s0 = (V.row(b) - V.row(a)).norm() > target_edge_length * kSplitTolerance;
                const bool s1 = (V.row(c) - V.row(b)).norm() > target_edge_length * kSplitTolerance;
                const bool s2 = (V.row(a) - V.row(c)).norm() > target_edge_length * kSplitTolerance;
                const int split_count = (int)s0 + (int)s1 + (int)s2;

                if (split_count == 0)
                {
                    add_face(a, b, c);
                    continue;
                }

                split_any = true;
                const int m0 = s0 ? get_midpoint(a, b) : -1;
                const int m1 = s1 ? get_midpoint(b, c) : -1;
                const int m2 = s2 ? get_midpoint(c, a) : -1;

                if (split_count == 1)
                {
                    if (s0)
                    {
                        add_face(a, m0, c);
                        add_face(m0, b, c);
                    }
                    else if (s1)
                    {
                        add_face(b, m1, a);
                        add_face(m1, c, a);
                    }
                    else
                    {
                        add_face(c, m2, b);
                        add_face(m2, a, b);
                    }
                }
                else if (split_count == 2)
                {
                    if (s0 && s1)
                    {
                        add_face(b, m1, m0);
                        add_face(a, m0, m1);
                        add_face(a, m1, c);
                    }
                    else if (s1 && s2)
                    {
                        add_face(c, m2, m1);
                        add_face(b, m1, m2);
                        add_face(b, m2, a);
                    }
                    else
                    {
                        add_face(a, m0, m2);
                        add_face(c, m2, m0);
                        add_face(c, m0, b);
                    }
                }
                else
                {
                    add_face(a, m0, m2);
                    add_face(m0, b, m1);
                    add_face(m2, m1, c);
                    add_face(m0, m1, m2);
                }
            }

            if (!split_any)
                break;
            if ((int)vertices.size() > kMaxVertices || (int)faces.size() > kMaxFaces)
            {
                if (error)
                    *error = "Fine render detail size would exceed the 2,000,000 element mesh limit.";
                return false;
            }

            const int previous_vertex_count = V.rows();
            V.resize((int)vertices.size(), 3);
            for (int vi = 0; vi < (int)vertices.size(); ++vi)
                V.row(vi) = vertices[(size_t)vi];
            if (refine_uv)
            {
                UV.resize((int)texture_coordinates.size(), 2);
                for (int vi = 0; vi < (int)texture_coordinates.size(); ++vi)
                    UV.row(vi) = texture_coordinates[(size_t)vi];
            }

            F.resize((int)faces.size(), 3);
            for (int fi = 0; fi < (int)faces.size(); ++fi)
                F.row(fi) = faces[(size_t)fi];
            source_faces.resize((int)next_source_faces.size());
            for (int fi = 0; fi < (int)next_source_faces.size(); ++fi)
                source_faces(fi) = next_source_faces[(size_t)fi];

            added_vertices += V.rows() - previous_vertex_count;
        }

        const double max_after = max_edge_length(V, F);
        if (max_after > target_edge_length * kSplitTolerance)
        {
            if (error)
                *error = "Target detail size could not be reached within the fine-render mesh limits.";
            return false;
        }

        out_V = std::move(V);
        out_F = std::move(F);
        if (refine_uv)
            *out_UV = std::move(UV);
        if (out_source_faces)
            *out_source_faces = std::move(source_faces);
        if (result)
        {
            result->passes = passes;
            result->added_vertices = added_vertices;
            result->output_faces = out_F.rows();
            result->max_edge_before = max_before;
            result->max_edge_after = max_after;
        }
        return true;
    }
}
