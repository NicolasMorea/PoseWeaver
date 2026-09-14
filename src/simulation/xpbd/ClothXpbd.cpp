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
            int i = -1;
            int j = -1;
            int k0 = -1;
            int k1 = -1;
            double rest_len = 0.0;
            double rest_angle = 0.0;
            double angle_scale = 1.0;
            double max_correction = 0.0;
        };

        static double dihedral_angle(
            const Eigen::Vector3d& xi,
            const Eigen::Vector3d& xj,
            const Eigen::Vector3d& xk0,
            const Eigen::Vector3d& xk1)
        {
            const Eigen::Vector3d edge = xj - xi;
            const Eigen::Vector3d n0 = edge.cross(xk0 - xi);
            const Eigen::Vector3d n1 = (xk1 - xi).cross(edge);
            const double edge_len = edge.norm();
            const double n0_len = n0.norm();
            const double n1_len = n1.norm();
            if (!(edge_len > 1e-12) || !(n0_len > 1e-12) || !(n1_len > 1e-12))
                return 0.0;

            const Eigen::Vector3d edge_dir = edge / edge_len;
            const Eigen::Vector3d n0_dir = n0 / n0_len;
            const Eigen::Vector3d n1_dir = n1 / n1_len;
            return std::atan2(edge_dir.dot(n0_dir.cross(n1_dir)),
                std::clamp(n0_dir.dot(n1_dir), -1.0, 1.0));
        }

        static double wrapped_angle_difference(double angle, double rest_angle)
        {
            constexpr double two_pi = 6.28318530717958647692;
            return std::remainder(angle - rest_angle, two_pi);
        }

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
                be.i = (int)(kv.first >> 32);
                be.j = (int)(kv.first & 0xffffffffu);
                be.k1 = k1;
                be.rest_len = (V_rest.row(k0) - V_rest.row(k1)).norm();
                be.rest_angle = dihedral_angle(
                    V_rest.row(be.i), V_rest.row(be.j), V_rest.row(k0), V_rest.row(k1));

                // Discrete-shell bending energy is proportional to l^2 / (A0 + A1).
                // Applying its square root to both the angle error and gradient makes
                // compliance independent of tessellation density.  The rest altitude
                // also provides a conservative trust region for badly folded hinges.
                const Eigen::Vector3d xi = V_rest.row(be.i);
                const Eigen::Vector3d xj = V_rest.row(be.j);
                const Eigen::Vector3d xk0 = V_rest.row(k0);
                const Eigen::Vector3d xk1 = V_rest.row(k1);
                const Eigen::Vector3d edge = xj - xi;
                const double edge_len = edge.norm();
                const double twice_area0 = edge.cross(xk0 - xi).norm();
                const double twice_area1 = (xk1 - xi).cross(edge).norm();
                const double area_sum = 0.5 * (twice_area0 + twice_area1);
                if (!(edge_len > 1e-12) || !(area_sum > 1e-16))
                    continue;
                be.angle_scale = std::sqrt(3.0 * edge_len * edge_len / area_sum);
                const double min_altitude = std::min(twice_area0, twice_area1) / edge_len;
                be.max_correction = 0.2 * min_altitude;
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

        static inline double wind_lerp(double a, double b, double t)
        {
            return a + (b - a) * t;
        }

        static inline double wind_fade(double t)
        {
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        }

        static inline std::uint32_t wind_hash_combine(std::uint32_t h, std::uint32_t v)
        {
            h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= h >> 16;
            h *= 0x7feb352du;
            h ^= h >> 15;
            h *= 0x846ca68bu;
            h ^= h >> 16;
            return h;
        }

        static double wind_hash_noise(int x, int y, int z, int w, std::uint32_t seed)
        {
            std::uint32_t h = wind_hash_combine(seed, static_cast<std::uint32_t>(x));
            h = wind_hash_combine(h, static_cast<std::uint32_t>(y));
            h = wind_hash_combine(h, static_cast<std::uint32_t>(z));
            h = wind_hash_combine(h, static_cast<std::uint32_t>(w));
            return (static_cast<double>(h & 0x00ffffffu) / 8388607.5) - 1.0;
        }

        static double wind_value_noise4(double x, double y, double z, double w, std::uint32_t seed)
        {
            const int ix = (int)std::floor(x);
            const int iy = (int)std::floor(y);
            const int iz = (int)std::floor(z);
            const int iw = (int)std::floor(w);
            const double fx = x - (double)ix;
            const double fy = y - (double)iy;
            const double fz = z - (double)iz;
            const double fw = w - (double)iw;
            const double sx = wind_fade(fx);
            const double sy = wind_fade(fy);
            const double sz = wind_fade(fz);
            const double sw = wind_fade(fw);

            double zw[2];
            for (int dw = 0; dw <= 1; ++dw)
            {
                double zv[2];
                for (int dz = 0; dz <= 1; ++dz)
                {
                    double yv[2];
                    for (int dy = 0; dy <= 1; ++dy)
                    {
                        const double a = wind_hash_noise(ix, iy + dy, iz + dz, iw + dw, seed);
                        const double b = wind_hash_noise(ix + 1, iy + dy, iz + dz, iw + dw, seed);
                        yv[dy] = wind_lerp(a, b, sx);
                    }
                    zv[dz] = wind_lerp(yv[0], yv[1], sy);
                }
                zw[dw] = wind_lerp(zv[0], zv[1], sz);
            }
            return wind_lerp(zw[0], zw[1], sw);
        }

        static double wind_fbm4(double x, double y, double z, double w, std::uint32_t seed)
        {
            double value = 0.0;
            double amplitude = 0.5;
            double norm = 0.0;
            double frequency = 1.0;
            for (int octave = 0; octave < 4; ++octave)
            {
                value += amplitude * wind_value_noise4(
                    x * frequency, y * frequency, z * frequency, w * frequency,
                    seed + (std::uint32_t)(octave * 1013));
                norm += amplitude;
                amplitude *= 0.5;
                frequency *= 2.03;
            }
            return (norm > 0.0) ? std::clamp(value / norm, -1.0, 1.0) : 0.0;
        }

        static Eigen::Vector3d random_wind_at(
            const Eigen::Vector3d& pos,
            double t,
            const ClothXpbdParams& params)
        {
            const double max_magnitude = std::max(0.0, params.wind_random_magnitude);
            if (!(max_magnitude > 0.0))
                return Eigen::Vector3d::Zero();

            const double spatial_scale = std::max(0.0, params.wind_random_spatial_scale);
            const double time_scale = std::max(0.0, params.wind_random_time_scale);
            const double x = pos.x() * spatial_scale;
            const double y = pos.y() * spatial_scale;
            const double z = pos.z() * spatial_scale;
            const double w = t * time_scale;

            Eigen::Vector3d raw(
                wind_fbm4(x + 17.31, y - 9.42, z + 3.77, w, 0x6a09e667u),
                wind_fbm4(x - 31.11, y + 5.13, z + 23.9, w + 41.0, 0xbb67ae85u),
                wind_fbm4(x + 7.49, y + 29.3, z - 13.7, w - 19.0, 0x3c6ef372u));
            const double len = raw.norm();
            if (!(len > 1e-9) || !std::isfinite(len))
                return Eigen::Vector3d::Zero();

            const double gust_noise = 0.5 + 0.5 * wind_fbm4(
                x * 0.63 + 4.1, y * 0.63 - 8.7, z * 0.63 + 12.2, w * 0.71 + 5.0, 0xa54ff53au);
            const double gust = std::pow(std::clamp(gust_noise, 0.0, 1.0), 2.35);
            const double magnitude = max_magnitude * std::clamp(0.08 + 0.92 * gust, 0.0, 1.0);
            return (magnitude / len) * raw;
        }

        static void apply_wind_prediction(
            Eigen::MatrixXd& X_pred,
            const Eigen::MatrixXd& X,
            int step,
            double dt,
            const ClothXpbdParams& params)
        {
            const double constant_magnitude = std::max(0.0, params.wind_force);
            const double random_magnitude = std::max(0.0, params.wind_random_magnitude);
            if (!(constant_magnitude > 0.0) && !(random_magnitude > 0.0))
                return;

            const double t = (double)step * dt;
            Eigen::Vector3d constant_wind = Eigen::Vector3d::Zero();
            if (constant_magnitude > 0.0)
            {
                Eigen::Vector3d dir = params.wind_dir;
                const double dn = dir.norm();
                if (dn > 1e-12 && std::isfinite(dn))
                    constant_wind = constant_magnitude * (dir / dn);
            }

            for (int i = 0; i < X.rows(); ++i)
            {
                const Eigen::Vector3d pos = X.row(i);
                const Eigen::Vector3d a = constant_wind + random_wind_at(pos, t, params);
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

        struct VertexCapsuleContact
        {
            int v = -1;
            int capsule = -1;
        };

        struct RuntimeCapsule
        {
            Eigen::Vector3d a = Eigen::Vector3d::Zero();
            Eigen::Vector3d b = Eigen::Vector3d::Zero();
            Eigen::Vector3d u = Eigen::Vector3d::UnitX();
            Eigen::Vector3d v = Eigen::Vector3d::UnitZ();
            Eigen::Vector3d w = Eigen::Vector3d::UnitY();
            double radius_u = 0.03;
            double radius_v = 0.03;
            double radius_w = 0.03;
            double bound_radius = 0.03;
        };

        static inline Eigen::Vector3d capsule_fallback_normal(const Eigen::Vector3d& a, const Eigen::Vector3d& b);

        static inline bool capsule_row(
            const Eigen::MatrixXd& capsules,
            int ci,
            RuntimeCapsule& out)
        {
            if (ci < 0 || ci >= capsules.rows() || capsules.cols() < 7)
                return false;
            out.a = capsules.block<1, 3>(ci, 0).transpose();
            out.b = capsules.block<1, 3>(ci, 3).transpose();
            out.radius_u = capsules(ci, 6);
            out.radius_v = (capsules.cols() >= 8) ? capsules(ci, 7) : out.radius_u;
            out.radius_w = (capsules.cols() >= 15)
                ? capsules(ci, 8)
                : 0.5 * (out.radius_u + out.radius_v);
            out.bound_radius = std::max({ out.radius_u, out.radius_v, out.radius_w });

            const Eigen::Vector3d axis = out.b - out.a;
            const double axis_len = axis.norm();
            out.w = (axis_len > 1e-12 && std::isfinite(axis_len))
                ? axis / axis_len
                : Eigen::Vector3d(0.0, 1.0, 0.0);

            const int frame_offset = capsules.cols() >= 15 ? 9 : 8;
            if (capsules.cols() >= frame_offset + 6)
            {
                out.u = capsules.block<1, 3>(ci, frame_offset).transpose();
                out.v = capsules.block<1, 3>(ci, frame_offset + 3).transpose();
                out.u -= out.u.dot(out.w) * out.w;
                const double un = out.u.norm();
                if (un > 1e-12 && std::isfinite(un))
                    out.u /= un;
                else
                    out.u = capsule_fallback_normal(out.a, out.b);
                out.v -= out.v.dot(out.w) * out.w;
                out.v -= out.v.dot(out.u) * out.u;
                const double vn = out.v.norm();
                if (vn > 1e-12 && std::isfinite(vn))
                    out.v /= vn;
                else
                {
                    out.v = out.w.cross(out.u);
                    const double fallback_vn = out.v.norm();
                    if (fallback_vn > 1e-12 && std::isfinite(fallback_vn))
                        out.v /= fallback_vn;
                    else
                        out.v = Eigen::Vector3d(0.0, 0.0, 1.0);
                }
            }
            else
            {
                out.u = capsule_fallback_normal(out.a, out.b);
                out.v = out.w.cross(out.u);
                const double vn = out.v.norm();
                if (vn > 1e-12 && std::isfinite(vn))
                    out.v /= vn;
                else
                    out.v = Eigen::Vector3d(0.0, 0.0, 1.0);
            }

            return out.a.allFinite() && out.b.allFinite()
                && out.u.allFinite() && out.v.allFinite() && out.w.allFinite()
                && out.radius_u > 0.0 && out.radius_v > 0.0 && out.radius_w > 0.0
                && std::isfinite(out.radius_u) && std::isfinite(out.radius_v)
                && std::isfinite(out.radius_w);
        }

        static inline Eigen::Vector3d capsule_fallback_normal(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
        {
            Eigen::Vector3d axis = b - a;
            const double an = axis.norm();
            if (an > 1e-12 && std::isfinite(an))
                axis /= an;
            else
                axis = Eigen::Vector3d(0.0, 1.0, 0.0);

            Eigen::Vector3d seed = (std::abs(axis.y()) < 0.9)
                ? Eigen::Vector3d(0.0, 1.0, 0.0)
                : Eigen::Vector3d(1.0, 0.0, 0.0);
            Eigen::Vector3d n = axis.cross(seed);
            if (n.squaredNorm() < 1e-12)
                n = axis.cross(Eigen::Vector3d(0.0, 0.0, 1.0));
            if (n.squaredNorm() < 1e-12)
                return Eigen::Vector3d(1.0, 0.0, 0.0);
            n.normalize();
            return n;
        }

        static inline double signed_distance_to_capsule(
            const Eigen::Vector3d& p,
            const RuntimeCapsule& capsule,
            Eigen::Vector3d* out_grad = nullptr)
        {
            const Eigen::Vector3d ab = capsule.b - capsule.a;
            const double len2 = ab.squaredNorm();
            double t = 0.0;
            if (len2 > 1e-12 && std::isfinite(len2))
                t = std::clamp((p - capsule.a).dot(ab) / len2, 0.0, 1.0);
            const Eigen::Vector3d q = capsule.a + t * ab;
            const Eigen::Vector3d d = p - q;
            const double dist = d.norm();
            Eigen::Vector3d grad = Eigen::Vector3d::Zero();
            if (out_grad)
            {
                if (dist > 1e-12 && std::isfinite(dist))
                    grad = d / dist;
                else
                    grad = capsule_fallback_normal(capsule.a, capsule.b);
                *out_grad = grad;
            }

            if (dist > 1e-12 && std::isfinite(dist))
                grad = d / dist;
            else
                grad = capsule_fallback_normal(capsule.a, capsule.b);

            const double cu = grad.dot(capsule.u);
            const double cv = grad.dot(capsule.v);
            const double cw = grad.dot(capsule.w);
            const double denom = std::sqrt(
                (cu * cu) / (capsule.radius_u * capsule.radius_u)
                + (cv * cv) / (capsule.radius_v * capsule.radius_v)
                + (cw * cw) / (capsule.radius_w * capsule.radius_w));
            const double effective_radius = (denom > 1e-12 && std::isfinite(denom))
                ? (1.0 / denom)
                : capsule.bound_radius;
            return dist - effective_radius;
        }

        static core::Aabb capsule_bounds(const Eigen::MatrixXd& capsules, double margin)
        {
            core::Aabb bb;
            for (int ci = 0; ci < capsules.rows(); ++ci)
            {
                RuntimeCapsule c;
                if (!capsule_row(capsules, ci, c))
                    continue;
                const double grow = c.bound_radius + margin;
                const Eigen::Vector3d mn = c.a.cwiseMin(c.b) - Eigen::Vector3d::Constant(grow);
                const Eigen::Vector3d mx = c.a.cwiseMax(c.b) + Eigen::Vector3d::Constant(grow);
                bb.expand(mn);
                bb.expand(mx);
            }
            return bb;
        }

        static bool find_vertex_capsule_contacts(
            const Eigen::MatrixXd& cloth_V,
            const Eigen::MatrixXd& capsules,
            double thickness,
            const std::vector<int>& vertex_indices,
            std::vector<VertexCapsuleContact>& out)
        {
            out.clear();
            if (vertex_indices.empty())
                return true;
            if (cloth_V.rows() == 0 || capsules.rows() == 0 || capsules.cols() < 7)
                return false;

            out.reserve(vertex_indices.size());
            for (int vi : vertex_indices)
            {
                if (vi < 0 || vi >= cloth_V.rows())
                    continue;

                const Eigen::Vector3d p = cloth_V.row(vi);
                int best_capsule = -1;
                double best_C = std::numeric_limits<double>::infinity();
                for (int ci = 0; ci < capsules.rows(); ++ci)
                {
                    RuntimeCapsule c;
                    if (!capsule_row(capsules, ci, c))
                        continue;
                    const double C = signed_distance_to_capsule(p, c) - thickness;
                    if (C < best_C)
                    {
                        best_C = C;
                        best_capsule = ci;
                    }
                }
                if (best_capsule >= 0 && best_C < 0.0 && std::isfinite(best_C))
                    out.push_back(VertexCapsuleContact{ vi, best_capsule });
            }

            return true;
        }

        static inline void apply_capsule_collision_constraint(
            Eigen::MatrixXd& Xs,
            const Eigen::MatrixXd& capsules,
            const VertexCapsuleContact& c,
            const Eigen::VectorXd& inv_mass,
            int N,
            double thickness,
            double col_alpha,
            std::vector<double>& lambda_col)
        {
            const int vi = c.v;
            if (vi < 0 || vi >= N)
                return;

            const double w = inv_mass[vi];
            if (!(w > 0.0))
                return;

            RuntimeCapsule capsule;
            if (!capsule_row(capsules, c.capsule, capsule))
                return;

            Eigen::Vector3d grad;
            const double C = signed_distance_to_capsule(Xs.row(vi).transpose(), capsule, &grad) - thickness;
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

        static inline Eigen::Vector3d capsule_contact_normal(
            const Eigen::MatrixXd& X,
            const Eigen::MatrixXd& capsules,
            const VertexCapsuleContact& c)
        {
            RuntimeCapsule capsule;
            Eigen::Vector3d grad = Eigen::Vector3d::Zero();
            if (!capsule_row(capsules, c.capsule, capsule))
                return grad;
            (void)signed_distance_to_capsule(X.row(c.v).transpose(), capsule, &grad);
            return grad;
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
        const bool use_capsule_collision =
            params.body_capsules.rows() > 0 && params.body_capsules.cols() >= 7;

        if (!use_capsule_collision && (body_F.rows() == 0 || body_F.cols() != 3))
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
        const Eigen::MatrixXd& cloth_reference =
            (params.reference_positions.rows() == cloth_V0.rows()
                && params.reference_positions.cols() == 3)
                ? params.reference_positions
                : cloth_V0;

        if (progress_steps_done)
            progress_steps_done->store(0);

        std::vector<StretchEdge> stretch_edges;
        std::vector<BendEdge> bend_edges;
        build_stretch_edges(cloth_reference, cloth_F, stretch_edges);
        build_bend_edges(cloth_reference, cloth_F, bend_edges);

        Eigen::VectorXd mass;
        if (!compute_lumped_masses(cloth_reference, cloth_F, mass))
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

        std::vector<std::pair<int, int>> fixed_entries;
        const bool have_fixed_targets =
            !params.fixed_vertex_indices.empty() &&
            !params.fixed_vertex_target_sequence.empty() &&
            params.fixed_vertex_target_sequence.front().cols() == 3 &&
            params.fixed_vertex_target_sequence.front().rows() == (int)params.fixed_vertex_indices.size();
        if (have_fixed_targets)
        {
            fixed_entries.reserve(params.fixed_vertex_indices.size());
            for (int ri = 0; ri < (int)params.fixed_vertex_indices.size(); ++ri)
            {
                const int vi = params.fixed_vertex_indices[(size_t)ri];
                if (vi >= 0 && vi < N)
                {
                    fixed_entries.emplace_back(vi, ri);
                    inv_mass[vi] = 0.0;
                }
            }
        }

        const auto fixed_targets_for_frame = [&](int frame_index) -> const Eigen::MatrixXd*
        {
            if (fixed_entries.empty())
                return nullptr;
            const int idx = std::clamp(frame_index, 0, (int)params.fixed_vertex_target_sequence.size() - 1);
            const Eigen::MatrixXd& targets = params.fixed_vertex_target_sequence[(size_t)idx];
            if (targets.cols() != 3 || targets.rows() != (int)params.fixed_vertex_indices.size())
                return nullptr;
            return &targets;
        };

        const auto apply_fixed_positions = [&](Eigen::MatrixXd& Xfix, int frame_index)
        {
            const Eigen::MatrixXd* targets = fixed_targets_for_frame(frame_index);
            if (!targets || Xfix.rows() != N || Xfix.cols() != 3)
                return;
            for (const auto& fr : fixed_entries)
                if (targets->row(fr.second).allFinite())
                    Xfix.row(fr.first) = targets->row(fr.second);
        };

        // Compliance alphas are scaled by sub_dt, not dt, so stiffness is invariant to substep count.
        const int substeps = std::max(1, params.substeps);
        const double sub_dt = dt / (double)substeps;

        const double stretch_alpha = compliance_alpha(params.stretch_compliance, sub_dt);
        const double bend_compliance = params.bending_method == BendingMethod::DihedralAngle
            ? params.dihedral_compliance
            : params.bend_compliance;
        const double bend_alpha = compliance_alpha(bend_compliance, sub_dt);
        const double col_alpha = compliance_alpha(params.collision_compliance, sub_dt);
        const double self_col_alpha = compliance_alpha(params.self_collision_compliance, sub_dt);

        const double thickness = std::max(0.0, params.collision_thickness);
        const double self_thickness = std::max(0.0, params.self_collision_thickness);

        collision::TriangleBvh bvh;
        // For static body (no sequence) build BVH once; dynamic body rebuilds per step in the loop.
        const Eigen::MatrixXd* p_body_V = &body_V;
        const Eigen::MatrixXd* p_body_capsules = use_capsule_collision ? &params.body_capsules : nullptr;
        int bvh_body_idx = -1;
        if (!use_capsule_collision && params.body_sequence.empty())
        {
            const auto t0 = Clock::now();
            bvh.build(body_V, body_F);
            ms_bvh_build += to_ms(Clock::now() - t0);
        }

        collision::TriangleBvh self_bvh;
        if (params.self_collision_enabled && self_thickness > 0.0)
        {
            const auto t0 = Clock::now();
            self_bvh.build(cloth_reference, cloth_F);
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
        apply_fixed_positions(X, 0);
        Eigen::MatrixXd Vv = Eigen::MatrixXd::Zero(N, 3);

        ClothXpbdResult result;
        result.frames.reserve((size_t)(steps + 1));
        result.frames.push_back(X);
        if (on_frame)
            on_frame(result.frames.back(), body_for_frame(result.stats.steps_simulated), result.stats.steps_simulated, on_frame_user);

        std::vector<collision::VertexTriangleContact> contacts;
        std::vector<VertexCapsuleContact> capsule_contacts;
        std::vector<collision::VertexTriangleContact> self_contacts;

        std::vector<double> lambda_stretch(stretch_edges.size(), 0.0);
        std::vector<double> lambda_bend(bend_edges.size(), 0.0);
        std::vector<double> lambda_bend_orientation(bend_edges.size(), 0.0);
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
                    if (use_capsule_collision)
                    {
                        if (!params.body_capsule_sequence.empty())
                        {
                            const int capsule_idx = std::min((int)params.body_capsule_sequence.size() - 1, idx);
                            p_body_capsules = &params.body_capsule_sequence[(size_t)capsule_idx];
                        }
                    }
                    else
                    {
                        const auto t0 = Clock::now();
                        bvh.build(*p_body_V, body_F);
                        ms_bvh_build += to_ms(Clock::now() - t0);
                    }
                    bvh_body_idx = idx;
                }
            }
            const Eigen::MatrixXd& cur_body_V = *p_body_V;
            const Eigen::MatrixXd* cur_body_capsules = p_body_capsules;
            const int fixed_frame = std::min(step + 1, std::max(0, max_frames - 1));

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
                apply_fixed_positions(X_pred, fixed_frame);

                Eigen::MatrixXd Xs = X_pred;
                apply_fixed_positions(Xs, fixed_frame);

                col_candidates.clear();
                {
                    const auto t0 = Clock::now();
                    if (params.collision_enabled)
                    {
                        const double margin = std::max(0.02, thickness * 4.0);
                        core::Aabb expanded;
                        if (use_capsule_collision && cur_body_capsules)
                            expanded = capsule_bounds(*cur_body_capsules, margin);
                        else
                        {
                            const core::Aabb bb = bvh.bounds();
                            expanded = core::Aabb{ bb.min - Eigen::Vector3d::Constant(margin), bb.max + Eigen::Vector3d::Constant(margin) };
                        }

                        if (expanded.valid())
                        {
                            col_candidates.reserve((size_t)N);
                            for (int vi = 0; vi < N; ++vi)
                            {
                                const Eigen::Vector3d p = Xs.row(vi);
                                if ((p.array() >= expanded.min.array()).all() && (p.array() <= expanded.max.array()).all())
                                    col_candidates.push_back(vi);
                            }
                        }
                    }
                    ms_candidates += to_ms(Clock::now() - t0);
                }

                std::fill(lambda_stretch.begin(), lambda_stretch.end(), 0.0);
                std::fill(lambda_bend.begin(), lambda_bend.end(), 0.0);
                std::fill(lambda_bend_orientation.begin(), lambda_bend_orientation.end(), 0.0);
                std::fill(lambda_col.begin(), lambda_col.end(), 0.0);
                contacts.clear();
                capsule_contacts.clear();
                self_contacts.clear();
                lambda_self.clear();

                const int iters = std::clamp(params.solver_iterations, 1, 500);
                const double base_stretch_mult = std::max(0.0, params.base_stretch_mult);

                int last_contact_count = 0;

                for (int it = 0; it < iters; ++it)
                {
                    if (cancel && cancel->load())
                        break;

                    apply_fixed_positions(Xs, fixed_frame);

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
                        const auto solve_dihedral = [&](const BendEdge& be, double& lambda, bool flipped_only)
                        {
                            const int indices[4] = { be.i, be.j, be.k0, be.k1 };
                            if (indices[0] < 0 || indices[1] < 0 || indices[2] < 0 || indices[3] < 0
                                || indices[0] >= N || indices[1] >= N || indices[2] >= N || indices[3] >= N)
                                return;

                            const Eigen::Vector3d xi = Xs.row(be.i);
                            const Eigen::Vector3d xj = Xs.row(be.j);
                            const Eigen::Vector3d xk0 = Xs.row(be.k0);
                            const Eigen::Vector3d xk1 = Xs.row(be.k1);
                            const Eigen::Vector3d edge = xj - xi;
                            const Eigen::Vector3d n0 = edge.cross(xk0 - xi);
                            const Eigen::Vector3d n1 = (xk1 - xi).cross(edge);
                            const double edge_len_sq = edge.squaredNorm();
                            const double n0_len_sq = n0.squaredNorm();
                            const double n1_len_sq = n1.squaredNorm();
                            if (!(edge_len_sq > 1e-20) || !(n0_len_sq > 1e-20) || !(n1_len_sq > 1e-20))
                                return;

                            const double current_angle = dihedral_angle(xi, xj, xk0, xk1);
                            if (flipped_only)
                            {
                                const double rest_side = std::sin(be.rest_angle);
                                if (std::abs(rest_side) <= 1e-6
                                    || std::sin(current_angle) * rest_side > 0.0)
                                    return;
                            }

                            const double edge_len = std::sqrt(edge_len_sq);
                            const Eigen::Vector3d qk0 = -(edge_len / n0_len_sq) * n0;
                            const Eigen::Vector3d qk1 = -(edge_len / n1_len_sq) * n1;
                            const Eigen::Vector3d qi =
                                ((xk0 - xj).dot(edge) / edge_len_sq) * qk0
                                + ((xk1 - xj).dot(edge) / edge_len_sq) * qk1;
                            const Eigen::Vector3d qj =
                                ((xi - xk0).dot(edge) / edge_len_sq) * qk0
                                + ((xi - xk1).dot(edge) / edge_len_sq) * qk1;
                            const Eigen::Vector3d gradients[4] = {
                                be.angle_scale * qi,
                                be.angle_scale * qj,
                                be.angle_scale * qk0,
                                be.angle_scale * qk1
                            };

                            double denominator = bend_alpha;
                            for (int vi = 0; vi < 4; ++vi)
                                denominator += inv_mass[indices[vi]] * gradients[vi].squaredNorm();
                            if (!(denominator > kEps))
                                return;

                            const double C = be.angle_scale * wrapped_angle_difference(
                                current_angle, be.rest_angle);
                            double dlambda = (-C - bend_alpha * lambda) / denominator;

                            double largest_response = 0.0;
                            for (int vi = 0; vi < 4; ++vi)
                                largest_response = std::max(largest_response,
                                    inv_mass[indices[vi]] * gradients[vi].norm());
                            if (largest_response > kEps && be.max_correction > 0.0)
                            {
                                const double max_dlambda = be.max_correction / largest_response;
                                dlambda = std::clamp(dlambda, -max_dlambda, max_dlambda);
                            }
                            lambda += dlambda;
                            for (int vi = 0; vi < 4; ++vi)
                            {
                                const double weight = inv_mass[indices[vi]];
                                if (weight > 0.0)
                                    Xs.row(indices[vi]) += (weight * dlambda * gradients[vi]).transpose();
                            }
                        };

                        for (size_t bi = 0; bi < bend_edges.size(); ++bi)
                        {
                            const BendEdge& be = bend_edges[bi];
                            if (params.bending_method == BendingMethod::DihedralAngle)
                            {
                                solve_dihedral(be, lambda_bend[bi], false);
                                continue;
                            }
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

                            // Opposite-vertex distance encodes the magnitude of a fold but
                            // not its side. Keep curved rest hinges on their original side
                            // so adjacent triangles cannot settle into alternating spikes.
                            solve_dihedral(be, lambda_bend_orientation[bi], true);
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
                            if (use_capsule_collision && cur_body_capsules)
                            {
                                const auto tq = Clock::now();
                                (void)find_vertex_capsule_contacts(Xs, *cur_body_capsules, thickness, col_candidates, capsule_contacts);
                                const double elapsed = to_ms(Clock::now() - tq);
                                ms_collision_bvh += elapsed;
                                ms_collision_narrow += elapsed;
                            }
                            else
                            {
                                collision::VertexTriangleContactQueryStats qs;
                                (void)collision::find_vertex_triangle_contacts(Xs, cur_body_V, body_F, bvh, thickness, col_candidates, contacts, &qs);
                                ms_collision_bvh += qs.ms_total;
                                ms_collision_narrow += qs.ms_bvh;
                                ms_collision_broad += std::max(0.0, qs.ms_total - qs.ms_bvh);
                            }
                        }

                        const auto t0 = Clock::now();
                        if (use_capsule_collision && cur_body_capsules)
                        {
                            for (const auto& c : capsule_contacts)
                            {
                                apply_capsule_collision_constraint(Xs, *cur_body_capsules, c, inv_mass, N, thickness, col_alpha, lambda_col);
                                last_contact_count++;
                            }
                        }
                        else
                        {
                            for (const auto& c : contacts)
                            {
                                apply_collision_constraint(Xs, cur_body_V, body_F, c, inv_mass, N, thickness, col_alpha, lambda_col);
                                last_contact_count++;
                            }
                        }
                        ms_collision_solve += to_ms(Clock::now() - t0);
                    }
                    else
                    {
                        contacts.clear();
                        capsule_contacts.clear();
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

                    apply_fixed_positions(Xs, fixed_frame);
                }

                // Final collision refresh on converged positions + extra passes for stubborn penetrations.
                if (params.collision_enabled && col_alpha >= 0.0 && !col_candidates.empty())
                {
                    {
                        if (use_capsule_collision && cur_body_capsules)
                        {
                            const auto tq = Clock::now();
                            (void)find_vertex_capsule_contacts(Xs, *cur_body_capsules, thickness, col_candidates, capsule_contacts);
                            const double elapsed = to_ms(Clock::now() - tq);
                            ms_collision_bvh += elapsed;
                            ms_collision_narrow += elapsed;
                        }
                        else
                        {
                            collision::VertexTriangleContactQueryStats qs;
                            (void)collision::find_vertex_triangle_contacts(Xs, cur_body_V, body_F, bvh, thickness, col_candidates, contacts, &qs);
                            ms_collision_bvh += qs.ms_total;
                            ms_collision_narrow += qs.ms_bvh;
                            ms_collision_broad += std::max(0.0, qs.ms_total - qs.ms_bvh);
                        }
                    }

                    result.stats.last_step_contacts = use_capsule_collision
                        ? (int)capsule_contacts.size()
                        : (int)contacts.size();

                    // Extra passes to clear remaining penetrations (e.g. in concave regions like armpits).
                    const int post_passes = 4;
                    for (int pass = 0; pass < post_passes; ++pass)
                    {
                        const auto t0 = Clock::now();
                        if (use_capsule_collision && cur_body_capsules)
                        {
                            for (const auto& c : capsule_contacts)
                                apply_capsule_collision_constraint(Xs, *cur_body_capsules, c, inv_mass, N, thickness, col_alpha, lambda_col);
                        }
                        else
                        {
                            for (const auto& c : contacts)
                                apply_collision_constraint(Xs, cur_body_V, body_F, c, inv_mass, N, thickness, col_alpha, lambda_col);
                        }
                        apply_fixed_positions(Xs, fixed_frame);
                        ms_collision_solve += to_ms(Clock::now() - t0);
                    }
                }
                else
                {
                    contacts.clear();
                    capsule_contacts.clear();
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
                        apply_fixed_positions(Xs, fixed_frame);
                        ms_self_collision_solve += to_ms(Clock::now() - t0);
                    }
                }
                else
                {
                    self_contacts.clear();
                    result.stats.last_step_self_contacts = 0;
                }

                X = Xs;
                apply_fixed_positions(X, fixed_frame);

                Vv = (X - X_sub_prev) / sub_dt;
                for (const auto& fr : fixed_entries)
                    Vv.row(fr.first).setZero();

                const double damp = std::clamp(params.damping, 0.0, 0.99);
                Vv *= (1.0 - damp);

                // Zero inward normal velocity at contacts to prevent tunneling on subsequent substeps.
                if (params.collision_enabled && use_capsule_collision && cur_body_capsules && !capsule_contacts.empty())
                {
                    for (const auto& cc : capsule_contacts)
                    {
                        const int vi = cc.v;
                        if (vi < 0 || vi >= N)
                            continue;
                        const Eigen::Vector3d n = capsule_contact_normal(X, *cur_body_capsules, cc);
                        if (!(n.squaredNorm() > 0.0) || !n.allFinite())
                            continue;

                        const double vn = Vv.row(vi).dot(n);
                        if (vn < 0.0)
                            Vv.row(vi) -= (vn * n).transpose();
                    }
                }
                else if (params.collision_enabled && !contacts.empty())
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

                const bool have_body_contacts = use_capsule_collision ? !capsule_contacts.empty() : !contacts.empty();
                if (params.enable_friction && params.friction_mu > 0.0 && have_body_contacts)
                {
                    const auto t0 = Clock::now();
                    const double mu = std::max(0.0, params.friction_mu);
                    if (use_capsule_collision && cur_body_capsules)
                    {
                        for (const auto& cc : capsule_contacts)
                        {
                            const int vi = cc.v;
                            if (vi < 0 || vi >= N)
                                continue;

                            const Eigen::Vector3d n = capsule_contact_normal(X, *cur_body_capsules, cc);
                            if (!(n.squaredNorm() > 0.0) || !n.allFinite())
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
                    }
                    else
                    {
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
                    }

                    X = X_sub_prev + sub_dt * Vv;
                    apply_fixed_positions(X, fixed_frame);
                    ms_friction += to_ms(Clock::now() - t0);

                    if (params.collision_enabled && col_alpha >= 0.0 && !col_candidates.empty())
                    {
                        {
                            if (use_capsule_collision && cur_body_capsules)
                            {
                                const auto tq = Clock::now();
                                (void)find_vertex_capsule_contacts(X, *cur_body_capsules, thickness, col_candidates, capsule_contacts);
                                const double elapsed = to_ms(Clock::now() - tq);
                                ms_collision_bvh += elapsed;
                                ms_collision_narrow += elapsed;
                            }
                            else
                            {
                                collision::VertexTriangleContactQueryStats qs;
                                (void)collision::find_vertex_triangle_contacts(X, cur_body_V, body_F, bvh, thickness, col_candidates, contacts, &qs);
                                ms_collision_bvh += qs.ms_total;
                                ms_collision_narrow += qs.ms_bvh;
                                ms_collision_broad += std::max(0.0, qs.ms_total - qs.ms_bvh);
                            }
                        }
                        if (use_capsule_collision && cur_body_capsules && !capsule_contacts.empty())
                        {
                            const auto t1 = Clock::now();
                            for (const auto& c : capsule_contacts)
                                apply_capsule_collision_constraint(X, *cur_body_capsules, c, inv_mass, N, thickness, col_alpha, lambda_col);
                            apply_fixed_positions(X, fixed_frame);
                            ms_collision_solve += to_ms(Clock::now() - t1);
                        }
                        else if (!contacts.empty())
                        {
                            const auto t1 = Clock::now();
                            for (const auto& c : contacts)
                                apply_collision_constraint(X, cur_body_V, body_F, c, inv_mass, N, thickness, col_alpha, lambda_col);
                            apply_fixed_positions(X, fixed_frame);
                            ms_collision_solve += to_ms(Clock::now() - t1);
                        }
                        Vv = (X - X_sub_prev) / sub_dt;
                        for (const auto& fr : fixed_entries)
                            Vv.row(fr.first).setZero();
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
                            apply_fixed_positions(X, fixed_frame);
                            ms_self_collision_solve += to_ms(Clock::now() - t1);
                            Vv = (X - X_sub_prev) / sub_dt;
                            for (const auto& fr : fixed_entries)
                                Vv.row(fr.first).setZero();
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
