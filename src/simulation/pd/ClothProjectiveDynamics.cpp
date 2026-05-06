#include "simulation/pd/ClothProjectiveDynamics.h"

#include "core/Geometry.h"
#include "simulation/collision/ClothContacts.h"
#include "simulation/collision/TriangleBvh.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

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

        static inline double clamp01(double x) { return std::max(0.0, std::min(1.0, x)); }

        static inline double wrap_angle_pi(double a)
        {
            constexpr double kPi = 3.14159265358979323846264338327950288;
            double x = std::fmod(a + kPi, 2.0 * kPi);
            if (x < 0.0)
                x += 2.0 * kPi;
            return x - kPi;
        }

        static inline Eigen::RowVector3d rotate_around_axis(
            const Eigen::RowVector3d& p,
            const Eigen::RowVector3d& axis_point,
            const Eigen::RowVector3d& axis_dir_unit,
            double angle_rad)
        {
            const Eigen::RowVector3d v = p - axis_point;
            const double c = std::cos(angle_rad);
            const double s = std::sin(angle_rad);
            const Eigen::RowVector3d k = axis_dir_unit;
            const Eigen::RowVector3d v_rot = v * c + (k.cross(v)) * s + k * (k.dot(v)) * (1.0 - c);
            return axis_point + v_rot;
        }

        static inline bool dihedral_angle(
            const Eigen::MatrixXd& V,
            int i,
            int j,
            int k0,
            int k1,
            double& out_theta)
        {
            if (i < 0 || j < 0 || k0 < 0 || k1 < 0)
                return false;
            if (i >= V.rows() || j >= V.rows() || k0 >= V.rows() || k1 >= V.rows())
                return false;

            const Eigen::RowVector3d pi = V.row(i);
            const Eigen::RowVector3d pj = V.row(j);
            const Eigen::RowVector3d pk0 = V.row(k0);
            const Eigen::RowVector3d pk1 = V.row(k1);

            Eigen::RowVector3d u = pj - pi;
            const double ulen = u.norm();
            if (!(ulen > 1e-20) || !std::isfinite(ulen))
                return false;
            u /= ulen;

            Eigen::RowVector3d n0, n1;
            if (!core::triangle_unit_normal(pi, pj, pk0, n0))
                return false;
            // Reverse the second triangle orientation along the shared edge.
            if (!core::triangle_unit_normal(pj, pi, pk1, n1))
                return false;

            const double dot01 = std::clamp(n0.dot(n1), -1.0, 1.0);
            const double s01 = u.dot(n0.cross(n1));
            const double theta = std::atan2(s01, dot01);
            if (!std::isfinite(theta))
                return false;
            out_theta = theta;
            return true;
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

        struct BendHinge
        {
            int i = -1;
            int j = -1;
            int k0 = -1;
            int k1 = -1;
            double rest_theta = 0.0;
        };

        // Isometric bending (PBD-style): distance constraint between opposite vertices across an interior edge.
        struct IsoBendEdge
        {
            int k0 = -1;
            int k1 = -1;
            double rest_len = 0.0;
        };

        using CollisionContact = collision::VertexTriangleContact;

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

        static void build_bend_hinges(
            const Eigen::MatrixXd& V_rest,
            const Eigen::MatrixXi& F,
            std::vector<BendHinge>& out_hinges)
        {
            out_hinges.clear();
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

            out_hinges.reserve(edge_faces.size());
            for (const auto& kv : edge_faces)
            {
                const EdgeInfo& e = kv.second;
                if (e.count != 2)
                    continue;

                // Use orientation from the first face and reverse orientation on the second.
                BendHinge h;
                h.i = e.f[0].v0;
                h.j = e.f[0].v1;
                h.k0 = e.f[0].opp;
                h.k1 = e.f[1].opp;

                double theta0 = 0.0;
                if (!dihedral_angle(V_rest, h.i, h.j, h.k0, h.k1, theta0))
                    continue;
                h.rest_theta = theta0;
                out_hinges.push_back(h);
            }
        }

        static void build_iso_bend_edges(
            const Eigen::MatrixXd& V_rest,
            const Eigen::MatrixXi& F,
            std::vector<IsoBendEdge>& out_edges)
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

                IsoBendEdge be;
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

            // Fallback for isolated vertices.
            const double mean = (out_mass.size() > 0) ? out_mass.mean() : 1.0;
            const double fallback = (mean > 0.0 && std::isfinite(mean)) ? mean : 1.0;
            for (int i = 0; i < out_mass.size(); ++i)
            {
                if (!(out_mass[i] > 0.0) || !std::isfinite(out_mass[i]))
                    out_mass[i] = fallback;
            }
            return true;
        }

        static inline void add_block_I3(std::vector<Eigen::Triplet<double>>& T, int i, int j, double w)
        {
            // Adds w*I3 to the 3x3 block (i,j) in a 3N x 3N matrix.
            for (int k = 0; k < 3; ++k)
                T.emplace_back(3 * i + k, 3 * j + k, w);
        }

        static inline void add_vec3(Eigen::VectorXd& b, int i, const Eigen::RowVector3d& v, double w)
        {
            b.segment<3>(3 * i) += w * v.transpose();
        }

    }

    bool run_cloth_projective_dynamics(
        const Eigen::MatrixXd& cloth_V0,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const ClothPdParams& params,
        ClothPdResult* out,
        std::string* error,
        std::atomic_bool* cancel,
        std::atomic_int* progress_steps_done,
        ClothPdOnFrameFn on_frame,
        void* on_frame_user)
    {
        if (out)
            *out = ClothPdResult{};

        using Clock = std::chrono::steady_clock;
        const auto t_total0 = Clock::now();
        const auto to_ms = [](Clock::duration d)
        {
            return std::chrono::duration<double, std::milli>(d).count();
        };

        double ms_bvh_build = 0.0;
        double ms_predict = 0.0;
        double ms_wind = 0.0;
        double ms_contacts_bvh = 0.0;
        double ms_contacts_query = 0.0;
        double ms_contacts_other = 0.0;
        double ms_assemble = 0.0;
        double ms_factor = 0.0;
        double ms_local_stretch = 0.0;
        double ms_local_bend = 0.0;
        double ms_local_collision = 0.0;
        double ms_global_solve = 0.0;
        double ms_post_pushout = 0.0;
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

        // Build constraints/topology from the initial mesh (treated as rest).
        std::vector<StretchEdge> edges;
        std::vector<BendHinge> hinges;
        std::vector<IsoBendEdge> iso_bend_edges;
        build_stretch_edges(cloth_V0, cloth_F, edges);
        build_bend_hinges(cloth_V0, cloth_F, hinges);
        build_iso_bend_edges(cloth_V0, cloth_F, iso_bend_edges);

        Eigen::VectorXd mass;
        if (!compute_lumped_masses(cloth_V0, cloth_F, mass))
        {
            if (error)
                *error = "Failed to compute masses.";
            return false;
        }

        const double m_avg = (mass.size() > 0) ? mass.mean() : 1.0;
        const double inv_dt2 = 1.0 / (dt * dt);

        const double stretch_stiff = std::max(0.0, params.stretch_stiffness);
        const double base_stretch_mult = std::max(0.0, params.base_stretch_mult);
        const double bend_stiff = std::max(0.0, params.bend_stiffness);
        const double iso_bend_stiff = std::max(0.0, params.iso_bend_stiffness);
        const double col_stiff = std::max(0.0, params.collision_stiffness);

        const double w_base = (m_avg > 0.0 && std::isfinite(m_avg)) ? (m_avg * inv_dt2) : inv_dt2;
        const double w_stretch = stretch_stiff * 10.0 * w_base;
        // Bending was previously far weaker than stretching; scale up so it meaningfully resists folding.
        const double w_bend = bend_stiff * 10.0 * w_base;
        const double w_iso_bend = iso_bend_stiff * 10.0 * w_base;
        const double w_col = col_stiff * 50.0 * w_base;

        const double thickness = std::max(0.0, params.collision_thickness);

        // Build BVH for the collision mesh (static during the run).
        collision::TriangleBvh bvh;
        {
            const auto t0 = Clock::now();
            bvh.build(body_V, body_F);
            ms_bvh_build += to_ms(Clock::now() - t0);
        }

        Eigen::MatrixXd X = cloth_V0;
        Eigen::MatrixXd X_prev = cloth_V0;
        Eigen::MatrixXd Vv = Eigen::MatrixXd::Zero(N, 3);

        ClothPdResult result;
        result.frames.reserve((size_t)(steps + 1));
        result.frames.push_back(X);
        if (on_frame)
            on_frame(result.frames.back(), result.stats.steps_simulated, on_frame_user);

        std::vector<CollisionContact> contacts;

        // Pre-allocate sparse system.
        using SpMat = Eigen::SparseMatrix<double>;
        using Triplet = Eigen::Triplet<double>;

        Eigen::VectorXd rhs(3 * N);
        Eigen::VectorXd xvec(3 * N);
        Eigen::VectorXd yvec(3 * N);

        auto pack = [&](const Eigen::MatrixXd& M, Eigen::VectorXd& outv)
        {
            outv.resize(3 * N);
            for (int i = 0; i < N; ++i)
                outv.segment<3>(3 * i) = M.row(i).transpose();
        };
        auto unpack = [&](const Eigen::VectorXd& v, Eigen::MatrixXd& outM)
        {
            outM.resize(N, 3);
            for (int i = 0; i < N; ++i)
                outM.row(i) = v.segment<3>(3 * i).transpose();
        };

        for (int step = 0; step < steps; ++step)
        {
            if (cancel && cancel->load())
                break;

            X_prev = X;

            // Inertial prediction.
            Eigen::MatrixXd Y;
            {
                const auto t0 = Clock::now();
                Y = X + dt * Vv;
                Y.col(1).array() += (dt * dt) * params.gravity_y;
                ms_predict += to_ms(Clock::now() - t0);
            }

            // Wind force: treated as additional acceleration during prediction.
            {
                const auto t0 = Clock::now();
                if (params.wind_force > 0.0)
                {
                    Eigen::Vector3d dir = params.wind_dir;
                    const double dn = dir.norm();
                    if (dn > 1e-12 && std::isfinite(dn))
                    {
                        dir /= dn;

                        // Smooth "random" variation. These are intentionally simple, deterministic, and cheap.
                        const double t = (double)step * dt;
                        const double w_time = 1.7; // rad/s
                        const Eigen::Vector3d k_space(12.9898, 78.233, 37.719);

                        const double time_term = (params.wind_random_time > 0.0)
                            ? std::sin(t * w_time)
                            : 0.0;

                        for (int i = 0; i < N; ++i)
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
                            Y.row(i) += ((dt * dt) * a).transpose();
                        }
                    }
                }
                ms_wind += to_ms(Clock::now() - t0);
            }

            // Broad+narrow phase (BVH) once per timestep to decide which vertices get collision constraints.
            // {
            //     const auto t0 = Clock::now();
            //     (void)collision::find_vertex_triangle_contacts(Y, body_V, body_F, bvh, thickness, contacts);
            //     ms_contacts_bvh += to_ms(Clock::now() - t0);
            // }

            // Assemble constant system matrix A for this timestep.
            const auto t_assemble0 = Clock::now();
            std::vector<Triplet> T;
            T.reserve((size_t)N * 3u + edges.size() * 24u + hinges.size() * 6u + iso_bend_edges.size() * 24u + contacts.size() * 6u);

            // Mass term.
            for (int i = 0; i < N; ++i)
            {
                const double m = mass[i] * inv_dt2;
                for (int k = 0; k < 3; ++k)
                    T.emplace_back(3 * i + k, 3 * i + k, m);
            }

            // Stretch (edge) term: w * ||(xi-xj) - p||^2 => adds Laplacian blocks.
            if (w_stretch > 0.0)
            {
                for (const StretchEdge& e : edges)
                {
                    if (e.i < 0 || e.j < 0)
                        continue;
                    add_block_I3(T, e.i, e.i, w_stretch);
                    add_block_I3(T, e.j, e.j, w_stretch);
                    add_block_I3(T, e.i, e.j, -w_stretch);
                    add_block_I3(T, e.j, e.i, -w_stretch);
                }
            }

            // Isometric bending term: w * ||(xk0-xk1) - p||^2 => adds Laplacian blocks.
            if (w_iso_bend > 0.0)
            {
                for (const IsoBendEdge& e : iso_bend_edges)
                {
                    if (e.k0 < 0 || e.k1 < 0)
                        continue;
                    add_block_I3(T, e.k0, e.k0, w_iso_bend);
                    add_block_I3(T, e.k1, e.k1, w_iso_bend);
                    add_block_I3(T, e.k0, e.k1, -w_iso_bend);
                    add_block_I3(T, e.k1, e.k0, -w_iso_bend);
                }
            }

            // Dihedral bending term: diagonal positional constraints on opposite vertices (k0,k1).
            if (w_bend > 0.0)
            {
                for (const BendHinge& h : hinges)
                {
                    if (h.k0 >= 0)
                        add_block_I3(T, h.k0, h.k0, w_bend);
                    if (h.k1 >= 0)
                        add_block_I3(T, h.k1, h.k1, w_bend);
                }
            }

            // Collision term: diagonal positional constraints on colliding vertices.
            // if (w_col > 0.0)
            // {
            //     for (const CollisionContact& c : contacts)
            //     {
            //         if (c.v < 0)
            //             continue;
            //         add_block_I3(T, c.v, c.v, w_col);
            //     }
            // }

            SpMat A(3 * N, 3 * N);
            A.setFromTriplets(T.begin(), T.end());
            ms_assemble += to_ms(Clock::now() - t_assemble0);

            Eigen::SimplicialLDLT<SpMat> solver;
            {
                const auto t0 = Clock::now();
                solver.compute(A);
                ms_factor += to_ms(Clock::now() - t0);
            }
            if (solver.info() != Eigen::Success)
            {
                if (error)
                    *error = "Failed to factorize the PD system matrix.";
                return false;
            }

            pack(X, xvec);
            pack(Y, yvec);

            // Local-global iterations.
            const int iters = std::clamp(params.local_global_iterations, 1, 200);
            for (int it = 0; it < iters; ++it)
            {
                if (cancel && cancel->load())
                    break;

                Eigen::MatrixXd Xit;
                unpack(xvec, Xit);

                rhs.setZero();

                // Mass RHS.
                for (int i = 0; i < N; ++i)
                {
                    const double m = mass[i] * inv_dt2;
                    rhs.segment<3>(3 * i) += m * yvec.segment<3>(3 * i);
                }

                // Stretch projections.
                if (w_stretch > 0.0)
                {
                    const auto t0 = Clock::now();
                    for (const StretchEdge& e : edges)
                    {
                        if (e.i < 0 || e.j < 0)
                            continue;

                        const Eigen::RowVector3d xi = Xit.row(e.i);
                        const Eigen::RowVector3d xj = Xit.row(e.j);
                        // NOTE: the system matrix is assembled for the term ||(xi - xj) - p||^2.
                        // Therefore the projected edge vector p must match the (xi - xj) orientation.
                        const Eigen::RowVector3d d = xi - xj;
                        const double len = d.norm();
                        const double rest = base_stretch_mult * e.rest_len;
                        Eigen::RowVector3d p = Eigen::RowVector3d::Zero();
                        if (len > kEps)
                            p = (rest / len) * d; // projected edge vector

                        add_vec3(rhs, e.i, p, w_stretch);
                        add_vec3(rhs, e.j, -p, w_stretch);
                    }
                    ms_local_stretch += to_ms(Clock::now() - t0);
                }

                // Isometric bending projections (distance between opposite vertices across a hinge).
                if (w_iso_bend > 0.0)
                {
                    const auto t0 = Clock::now();
                    for (const IsoBendEdge& e : iso_bend_edges)
                    {
                        if (e.k0 < 0 || e.k1 < 0)
                            continue;

                        const Eigen::RowVector3d x0 = Xit.row(e.k0);
                        const Eigen::RowVector3d x1 = Xit.row(e.k1);
                        // NOTE: the system matrix is assembled for the term ||(x0 - x1) - p||^2.
                        // Therefore the projected vector p must match the (x0 - x1) orientation.
                        const Eigen::RowVector3d d = x0 - x1;
                        const double len = d.norm();
                        const double rest = e.rest_len;
                        Eigen::RowVector3d p = Eigen::RowVector3d::Zero();
                        if (len > kEps)
                            p = (rest / len) * d;

                        add_vec3(rhs, e.k0, p, w_iso_bend);
                        add_vec3(rhs, e.k1, -p, w_iso_bend);
                    }
                    ms_local_bend += to_ms(Clock::now() - t0);
                }

                // Bending projections (dihedral) -> positional targets for opposite vertices.
                if (w_bend > 0.0)
                {
                    const auto t0 = Clock::now();
                    for (const BendHinge& h : hinges)
                    {
                        double theta = 0.0;
                        if (!dihedral_angle(Xit, h.i, h.j, h.k0, h.k1, theta))
                            continue;

                        const double err = wrap_angle_pi(theta - h.rest_theta);
                        if (!std::isfinite(err))
                            continue;

                        const Eigen::RowVector3d pi = Xit.row(h.i);
                        const Eigen::RowVector3d pj = Xit.row(h.j);
                        Eigen::RowVector3d axis = pj - pi;
                        const double alen = axis.norm();
                        if (!(alen > kEps))
                            continue;
                        axis /= alen;

                        const Eigen::RowVector3d pk0 = Xit.row(h.k0);
                        const Eigen::RowVector3d pk1 = Xit.row(h.k1);

                        Eigen::RowVector3d pk0p = pk0;
                        Eigen::RowVector3d pk1p = pk1;

                        // Full correction in projection; stiffness handled by weight.
                        // NOTE: err = theta - rest; the correction must be applied with the opposite sign.
                        const double dtheta = std::clamp(-err, -0.75, 0.75);
                        if (std::abs(dtheta) >= 1e-12)
                        {
                            pk0p = rotate_around_axis(pk0, pi, axis, -0.5 * dtheta);
                            pk1p = rotate_around_axis(pk1, pi, axis, +0.5 * dtheta);
                        }

                        if (!pk0p.allFinite() || !pk1p.allFinite())
                            continue;

                        add_vec3(rhs, h.k0, pk0p, w_bend);
                        add_vec3(rhs, h.k1, pk1p, w_bend);
                    }
                    ms_local_bend += to_ms(Clock::now() - t0);
                }

                // Collision projections -> positional targets for vertices.
                // int contact_count = 0;
                // if (w_col > 0.0)
                // {
                //     const auto t0 = Clock::now();
                //     for (const CollisionContact& c : contacts)
                //     {
                //         const int vi = c.v;
                //         const int face = c.face;
                //         if (vi < 0 || vi >= Xit.rows())
                //             continue;
                //         if (face < 0 || face >= body_F.rows())
                //             continue;

                //         const int i0 = body_F(face, 0);
                //         const int i1 = body_F(face, 1);
                //         const int i2 = body_F(face, 2);
                //         if (i0 < 0 || i1 < 0 || i2 < 0)
                //             continue;
                //         if (i0 >= body_V.rows() || i1 >= body_V.rows() || i2 >= body_V.rows())
                //             continue;

                //         const Eigen::Vector3d a = body_V.row(i0);
                //         const Eigen::Vector3d b = body_V.row(i1);
                //         const Eigen::Vector3d ctri = body_V.row(i2);
                //         const Eigen::Vector3d p = Xit.row(vi);

                //         const Eigen::Vector3d q = core::closest_point_on_triangle(p, a, b, ctri, nullptr);
                //         const Eigen::Vector3d n = core::triangle_unit_normal(a, b, ctri);
                //         if (!(n.squaredNorm() > 0.0))
                //             continue;

                //         // Signed distance based on closest-point distance magnitude, with the triangle normal
                //         // providing the one-sided orientation.
                //         const Eigen::Vector3d pq = p - q;
                //         const double dist = pq.norm();
                //         const double sd_plane = n.dot(pq);
                //         const double sgn = (sd_plane >= 0.0) ? 1.0 : -1.0;

                //         double sd = sd_plane;
                //         Eigen::Vector3d grad = n;
                //         if (dist > 1e-12 && std::isfinite(dist))
                //         {
                //             sd = sgn * dist;
                //             grad = (sgn / dist) * pq;
                //         }

                //         const double depth = thickness - sd;
                //         Eigen::RowVector3d target = p.transpose();
                //         if (depth > 0.0 && std::isfinite(depth) && grad.allFinite())
                //         {
                //             target = (p + depth * grad).transpose();
                //             contact_count++;
                //         }
                //         add_vec3(rhs, vi, target, w_col);
                //     }
                //     ms_local_collision += to_ms(Clock::now() - t0);
                // }

                // Global solve.
                Eigen::VectorXd x_new;
                {
                    const auto t0 = Clock::now();
                    x_new = solver.solve(rhs);
                    ms_global_solve += to_ms(Clock::now() - t0);
                }
                if (solver.info() != Eigen::Success)
                {
                    if (error)
                        *error = "PD linear solve failed.";
                    return false;
                }

                xvec = std::move(x_new);
                // result.stats.last_step_contacts = contact_count;
            }

            unpack(xvec, X);

            // Update contact set on the converged positions and apply a final push-out pass.
            // This helps catch contacts that were not present in the initial broad+narrow phase.
            if (w_col > 0.0)
            {
                {
                    collision::VertexTriangleContactQueryStats qs;
                    (void)collision::find_vertex_triangle_contacts(X, body_V, body_F, bvh, thickness, contacts, &qs);
                    ms_contacts_bvh += qs.ms_total;
                    ms_contacts_query += qs.ms_bvh;
                    ms_contacts_other += std::max(0.0, qs.ms_total - qs.ms_bvh);
                }
                {
                    const auto t0 = Clock::now();
                    (void)collision::apply_vertex_triangle_pushout(X, body_V, body_F, contacts, thickness, 2);
                    ms_post_pushout += to_ms(Clock::now() - t0);
                }
            }
            else
            {
                contacts.clear();
            }

            // Velocity update.
            Vv = (X - X_prev) / dt;

            // Damping.
            const double damp = std::clamp(params.damping, 0.0, 0.99);
            Vv *= (1.0 - damp);

            // Optional friction (Coulomb-like, velocity-level) on contact vertices.
            if (params.enable_friction && params.friction_mu > 0.0 && !contacts.empty())
            {
                {
                    const auto t0 = Clock::now();
                    const double mu = std::max(0.0, params.friction_mu);
                    for (const CollisionContact& cc : contacts)
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
                        if (i0 >= body_V.rows() || i1 >= body_V.rows() || i2 >= body_V.rows())
                            continue;

                        const Eigen::Vector3d n = core::triangle_unit_normal(
                            body_V.row(i0).transpose(),
                            body_V.row(i1).transpose(),
                            body_V.row(i2).transpose());
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

                    // Re-apply position update to match modified velocities.
                    X = X_prev + dt * Vv;
                    ms_friction += to_ms(Clock::now() - t0);
                }

                // Ensure we didn't reintroduce penetration with the position-level velocity update.
                if (w_col > 0.0)
                {
                    {
                        collision::VertexTriangleContactQueryStats qs;
                        (void)collision::find_vertex_triangle_contacts(X, body_V, body_F, bvh, thickness, contacts, &qs);
                        ms_contacts_bvh += qs.ms_total;
                        ms_contacts_query += qs.ms_bvh;
                        ms_contacts_other += std::max(0.0, qs.ms_total - qs.ms_bvh);
                    }
                    {
                        const auto t0 = Clock::now();
                        (void)collision::apply_vertex_triangle_pushout(X, body_V, body_F, contacts, thickness, 1);
                        ms_post_pushout += to_ms(Clock::now() - t0);
                    }
                    Vv = (X - X_prev) / dt;
                }
            }

            result.frames.push_back(X);
            result.stats.steps_simulated++;

            if (on_frame)
                on_frame(result.frames.back(), result.stats.steps_simulated, on_frame_user);

            if (progress_steps_done)
                progress_steps_done->store(result.stats.steps_simulated);

            if ((int)result.frames.size() >= max_frames)
                break;
        }

        result.stats.frames_recorded = (int)result.frames.size();

        result.stats.ms_total = to_ms(Clock::now() - t_total0);
        result.stats.ms_bvh_build = ms_bvh_build;
        result.stats.ms_predict = ms_predict;
        result.stats.ms_wind = ms_wind;
        result.stats.ms_contacts_bvh = ms_contacts_bvh;
        result.stats.ms_contacts_query = ms_contacts_query;
        result.stats.ms_contacts_other = ms_contacts_other;
        result.stats.ms_assemble = ms_assemble;
        result.stats.ms_factor = ms_factor;
        result.stats.ms_local_stretch = ms_local_stretch;
        result.stats.ms_local_bend = ms_local_bend;
        result.stats.ms_local_collision = ms_local_collision;
        result.stats.ms_global_solve = ms_global_solve;
        result.stats.ms_post_pushout = ms_post_pushout;
        result.stats.ms_friction = ms_friction;

        if (out)
            *out = std::move(result);
        return true;
    }
}
