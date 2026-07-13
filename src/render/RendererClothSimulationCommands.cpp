#include "render/Renderer.h"

#include "core/Geometry.h"
#include "simulation/ClothMeshProcessor.h"
#include "simulation/SmplCollisionProxy.h"
#include "simulation/SmplSimplifier.h"
#include "simulation/collision/TriangleBvh.h"
#include "simulation/xpbd/ClothXpbd.h"
#include "simulation/gpu/ClothCudaJacobi.h"

#include <igl/signed_distance.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace poseweaver
{
    namespace
    {
        template <typename T>
        static bool future_ready(std::future<T>& f)
        {
            return f.valid() && (f.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
        }

        static inline int clamp_frame(int i, int count)
        {
            if (count <= 0)
                return 0;
            return std::clamp(i, 0, count - 1);
        }

        static Eigen::MatrixXd compute_cloth_smpl_targets(
            const std::vector<int>& cloth_indices,
            const Eigen::MatrixXd& fallback_cloth_V,
            const ClothToSmplMapping& mapping,
            const Eigen::MatrixXd& body_V,
            const Eigen::MatrixXi& body_F,
            const Eigen::MatrixXd& body_rest_V)
        {
            Eigen::MatrixXd targets((int)cloth_indices.size(), 3);
            targets.setZero();

            for (int ri = 0; ri < (int)cloth_indices.size(); ++ri)
            {
                const int vi = cloth_indices[(size_t)ri];
                if (vi < 0 || vi >= fallback_cloth_V.rows())
                    continue;

                targets.row(ri) = fallback_cloth_V.row(vi);
                if (!mapping.valid() || mapping.face_index.size() <= vi)
                    continue;

                const int fi = mapping.face_index[vi];
                if (fi < 0 || fi >= body_F.rows())
                    continue;

                const int ia = body_F(fi, 0);
                const int ib = body_F(fi, 1);
                const int ic = body_F(fi, 2);
                if (ia < 0 || ib < 0 || ic < 0 ||
                    ia >= body_V.rows() || ib >= body_V.rows() || ic >= body_V.rows())
                {
                    continue;
                }

                const Eigen::RowVector3d bc = mapping.barycentric.row(vi);
                const Eigen::RowVector3d p_def =
                    bc[0] * body_V.row(ia) + bc[1] * body_V.row(ib) + bc[2] * body_V.row(ic);

                Eigen::RowVector3d T_def, B_def, N_def;
                if (!core::triangle_frame(body_V.row(ia), body_V.row(ib), body_V.row(ic), T_def, B_def, N_def))
                {
                    if (!core::triangle_frame(
                            body_rest_V.row(ia),
                            body_rest_V.row(ib),
                            body_rest_V.row(ic),
                            T_def,
                            B_def,
                            N_def))
                    {
                        continue;
                    }
                }

                const Eigen::RowVector3d off_local = mapping.offset_local.row(vi);
                targets.row(ri) = p_def + off_local[0] * T_def + off_local[1] * B_def + off_local[2] * N_def;
            }

            return targets;
        }

        struct ConstraintSkinProjectionResult
        {
            int corrected_vertices = 0;
            int pushed_out_vertices = 0;
            int pulled_in_vertices = 0;
        };

        static bool project_constraints_to_skin_margin(
            Eigen::MatrixXd& cloth_V,
            ClothToSmplMapping& mapping,
            const std::vector<int>& cloth_indices,
            const Eigen::MatrixXd& body_V,
            const Eigen::MatrixXi& body_F,
            double surface_offset,
            ConstraintSkinProjectionResult* result,
            std::string* error)
        {
            if (result)
                *result = ConstraintSkinProjectionResult{};

            if (cloth_indices.empty())
                return true;
            if (cloth_V.cols() != 3 || body_V.cols() != 3)
            {
                if (error)
                    *error = "Expected 3D vertex positions (#V x 3).";
                return false;
            }
            if (body_F.cols() != 3)
            {
                if (error)
                    *error = "Expected body faces to be triangles (#F x 3).";
                return false;
            }
            if (!mapping.valid() || mapping.face_index.size() != cloth_V.rows())
            {
                if (error)
                    *error = "Cloth-to-SMPL mapping is not valid for constrained projection.";
                return false;
            }
            if (body_V.rows() == 0 || body_F.rows() == 0)
            {
                if (error)
                    *error = "Missing body mesh data.";
                return false;
            }
            if (!(surface_offset >= 0.0) || !std::isfinite(surface_offset))
            {
                if (error)
                    *error = "surface_offset must be a finite non-negative value.";
                return false;
            }

            Eigen::MatrixXd P((int)cloth_indices.size(), 3);
            std::vector<int> source_vertices;
            source_vertices.reserve(cloth_indices.size());
            for (int ri = 0; ri < (int)cloth_indices.size(); ++ri)
            {
                const int vi = cloth_indices[(size_t)ri];
                if (vi < 0 || vi >= cloth_V.rows())
                {
                    if (error)
                        *error = "Constrained cloth vertex index is out of range.";
                    return false;
                }
                P.row(ri) = cloth_V.row(vi);
                source_vertices.push_back(vi);
            }

            Eigen::VectorXd S;
            Eigen::VectorXi I;
            Eigen::MatrixXd C;
            Eigen::MatrixXd N_unused;
            igl::signed_distance(P, body_V, body_F, igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, S, I, C, N_unused);

            if (S.size() != P.rows() || I.size() != P.rows() || C.rows() != P.rows() || C.cols() != 3)
            {
                if (error)
                    *error = "Signed-distance query returned unexpected sizes.";
                return false;
            }

            const Eigen::RowVector3d body_centroid = body_V.colwise().mean();
            constexpr double kEps = 1e-9;

            int corrected = 0;
            int pushed_out = 0;
            int pulled_in = 0;

            for (int ri = 0; ri < P.rows(); ++ri)
            {
                const int vi = source_vertices[(size_t)ri];
                const int fi = I[ri];
                if (fi < 0 || fi >= body_F.rows() || !std::isfinite(S[ri]))
                    continue;

                const int ia = body_F(fi, 0);
                const int ib = body_F(fi, 1);
                const int ic = body_F(fi, 2);
                if (ia < 0 || ib < 0 || ic < 0 ||
                    ia >= body_V.rows() || ib >= body_V.rows() || ic >= body_V.rows())
                {
                    continue;
                }

                Eigen::RowVector3d dir = (S[ri] >= 0.0) ? (P.row(ri) - C.row(ri)) : (C.row(ri) - P.row(ri));
                double dir_norm = dir.norm();
                if (!(dir_norm > 1e-12) || !std::isfinite(dir_norm))
                {
                    const Eigen::RowVector3d a = body_V.row(ia);
                    const Eigen::RowVector3d b = body_V.row(ib);
                    const Eigen::RowVector3d c = body_V.row(ic);
                    dir = (b - a).cross(c - a);
                    dir_norm = dir.norm();
                    if (!(dir_norm > 1e-12) || !std::isfinite(dir_norm))
                        dir = C.row(ri) - body_centroid;
                    else if (dir.dot(((a + b + c) / 3.0) - body_centroid) < 0.0)
                        dir = -dir;
                    dir_norm = dir.norm();
                }
                if (!(dir_norm > 1e-12) || !std::isfinite(dir_norm))
                    continue;
                dir /= dir_norm;

                const Eigen::RowVector3d corrected_p = C.row(ri) + surface_offset * dir;
                if (!corrected_p.allFinite())
                    continue;

                if (S[ri] < surface_offset - kEps)
                    pushed_out++;
                else if (S[ri] > surface_offset + kEps)
                    pulled_in++;

                const Eigen::RowVector3d a = body_V.row(ia);
                const Eigen::RowVector3d b = body_V.row(ib);
                const Eigen::RowVector3d c = body_V.row(ic);
                const Eigen::RowVector3d bc = core::barycentric_on_triangle(C.row(ri), a, b, c);
                if (!bc.allFinite())
                    continue;
                const Eigen::RowVector3d off = corrected_p - C.row(ri);

                Eigen::RowVector3d T, B, N;
                if (!core::triangle_frame(a, b, c, T, B, N))
                {
                    mapping.offset_local.row(vi) = off;
                }
                else
                {
                    mapping.offset_local(vi, 0) = off.dot(T);
                    mapping.offset_local(vi, 1) = off.dot(B);
                    mapping.offset_local(vi, 2) = off.dot(N);
                }

                cloth_V.row(vi) = corrected_p;
                mapping.face_index[vi] = fi;
                mapping.barycentric.row(vi) = bc;
                mapping.offset.row(vi) = off;
                corrected++;
            }

            if (corrected != P.rows())
            {
                if (error)
                {
                    *error = "Could only project " + std::to_string(corrected)
                        + " of " + std::to_string((int)P.rows())
                        + " constrained vertices to the skin margin.";
                }
                return false;
            }

            if (result)
            {
                result->corrected_vertices = corrected;
                result->pushed_out_vertices = pushed_out;
                result->pulled_in_vertices = pulled_in;
            }
            return true;
        }

        struct BodyProxyPrepStats
        {
            int frames_processed = 0;
            int self_collision_contacts = 0;
            double inflation_margin = 0.0;
            double self_clearance = 0.0;
        };

        static Eigen::MatrixXd body_proxy_vertex_normals(
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F)
        {
            Eigen::MatrixXd N = Eigen::MatrixXd::Zero(V.rows(), 3);
            if (V.rows() == 0 || V.cols() != 3 || F.cols() != 3)
                return N;

            const Eigen::RowVector3d centroid = V.colwise().mean();
            for (int fi = 0; fi < F.rows(); ++fi)
            {
                const int i0 = F(fi, 0);
                const int i1 = F(fi, 1);
                const int i2 = F(fi, 2);
                if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows())
                    continue;

                const Eigen::RowVector3d a = V.row(i0);
                const Eigen::RowVector3d b = V.row(i1);
                const Eigen::RowVector3d c = V.row(i2);
                Eigen::RowVector3d n = (b - a).cross(c - a);
                const double nn = n.norm();
                if (!(nn > 1e-20) || !std::isfinite(nn))
                    continue;

                const Eigen::RowVector3d fc = (a + b + c) / 3.0;
                if (n.dot(fc - centroid) < 0.0)
                    n = -n;

                N.row(i0) += n;
                N.row(i1) += n;
                N.row(i2) += n;
            }

            for (int vi = 0; vi < N.rows(); ++vi)
            {
                Eigen::RowVector3d n = N.row(vi);
                double nn = n.norm();
                if (!(nn > 1e-20) || !std::isfinite(nn))
                {
                    n = V.row(vi) - centroid;
                    nn = n.norm();
                }
                if (!(nn > 1e-20) || !std::isfinite(nn))
                    n = Eigen::RowVector3d(0.0, 1.0, 0.0);
                else
                    n /= nn;
                N.row(vi) = n;
            }

            return N;
        }

        static void inflate_body_proxy(
            Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            double margin)
        {
            if (!(margin > 0.0) || V.rows() == 0 || V.cols() != 3)
                return;

            const Eigen::MatrixXd N = body_proxy_vertex_normals(V, F);
            if (N.rows() != V.rows())
                return;

            V += margin * N;
        }

        static bool body_proxy_face_is_local_to_vertex(
            int vi,
            int face,
            const Eigen::MatrixXi& F,
            const std::vector<std::vector<int>>& one_ring)
        {
            if (vi < 0 || face < 0 || face >= F.rows())
                return true;

            for (int k = 0; k < 3; ++k)
            {
                const int fv = F(face, k);
                if (fv == vi)
                    return true;
                if (vi >= 0 && vi < (int)one_ring.size())
                {
                    const auto& ring = one_ring[(size_t)vi];
                    if (std::find(ring.begin(), ring.end(), fv) != ring.end())
                        return true;
                }
            }

            return false;
        }

        static std::vector<std::vector<int>> build_body_proxy_one_ring(
            int vertex_count,
            const Eigen::MatrixXi& F)
        {
            std::vector<std::vector<int>> one_ring((size_t)std::max(0, vertex_count));
            const auto add_edge = [&](int a, int b)
            {
                if (a < 0 || b < 0 || a >= vertex_count || b >= vertex_count || a == b)
                    return;
                auto& ring = one_ring[(size_t)a];
                if (std::find(ring.begin(), ring.end(), b) == ring.end())
                    ring.push_back(b);
            };

            for (int fi = 0; fi < F.rows(); ++fi)
            {
                const int i0 = F(fi, 0);
                const int i1 = F(fi, 1);
                const int i2 = F(fi, 2);
                add_edge(i0, i1);
                add_edge(i1, i0);
                add_edge(i1, i2);
                add_edge(i2, i1);
                add_edge(i2, i0);
                add_edge(i0, i2);
            }

            return one_ring;
        }

        static int relax_body_proxy_self_collisions(
            Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            double clearance,
            int passes)
        {
            if (!(clearance > 0.0) || passes <= 0 || V.rows() == 0 || F.rows() == 0 || V.cols() != 3 || F.cols() != 3)
                return 0;

            const std::vector<std::vector<int>> one_ring = build_body_proxy_one_ring((int)V.rows(), F);
            const double search_radius = std::max(0.02, clearance * 5.0);
            const double max_dist2 = search_radius * search_radius;

            int contacts = 0;
            for (int pass = 0; pass < passes; ++pass)
            {
                collision::TriangleBvh bvh;
                bvh.build(V, F);
                if (!bvh.bounds().valid())
                    break;

                const Eigen::RowVector3d centroid = V.colwise().mean();
                int pass_contacts = 0;

                for (int vi = 0; vi < V.rows(); ++vi)
                {
                    const Eigen::Vector3d p = V.row(vi);
                    int face = -1;
                    Eigen::Vector3d q;
                    double dist2 = std::numeric_limits<double>::infinity();
                    if (!bvh.closest_triangle(p, V, F, face, q, dist2, max_dist2, vi))
                        continue;
                    if (body_proxy_face_is_local_to_vertex(vi, face, F, one_ring))
                        continue;

                    const int i0 = F(face, 0);
                    const int i1 = F(face, 1);
                    const int i2 = F(face, 2);
                    if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows())
                        continue;

                    const Eigen::Vector3d a = V.row(i0);
                    const Eigen::Vector3d b = V.row(i1);
                    const Eigen::Vector3d c = V.row(i2);

                    Eigen::Vector3d bary;
                    q = core::closest_point_on_triangle(p, a, b, c, &bary);
                    Eigen::Vector3d n = (b - a).cross(c - a);
                    const double nn = n.norm();
                    if (!(nn > 1e-20) || !std::isfinite(nn) || !bary.allFinite())
                        continue;
                    n /= nn;

                    const Eigen::Vector3d face_centroid = (a + b + c) / 3.0;
                    if (n.dot(face_centroid - centroid.transpose()) < 0.0)
                        n = -n;

                    const double signed_distance = n.dot(p - q);
                    if (!(signed_distance < clearance) || !std::isfinite(signed_distance))
                        continue;

                    const double b0 = std::clamp(bary[0], 0.0, 1.0);
                    const double b1 = std::clamp(bary[1], 0.0, 1.0);
                    const double b2 = std::clamp(bary[2], 0.0, 1.0);
                    const double denom = 1.0 + b0 * b0 + b1 * b1 + b2 * b2;
                    if (!(denom > 1e-12))
                        continue;

                    const double depth = std::min(clearance - signed_distance, 0.5 * search_radius);
                    const Eigen::Vector3d corr = (depth / denom) * n;
                    if (!corr.allFinite())
                        continue;

                    V.row(vi) += corr.transpose();
                    V.row(i0) -= (b0 * corr).transpose();
                    V.row(i1) -= (b1 * corr).transpose();
                    V.row(i2) -= (b2 * corr).transpose();

                    pass_contacts++;
                }

                contacts += pass_contacts;
                if (pass_contacts == 0)
                    break;
            }

            return contacts;
        }

        static BodyProxyPrepStats prepare_xpbd_body_proxy_frame(
            Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            double inflation_margin,
            double self_clearance)
        {
            BodyProxyPrepStats stats;
            stats.frames_processed = 1;
            stats.inflation_margin = inflation_margin;
            stats.self_clearance = self_clearance;

            inflate_body_proxy(V, F, inflation_margin);
            stats.self_collision_contacts = relax_body_proxy_self_collisions(V, F, self_clearance, 4);
            return stats;
        }
    }

    void Renderer::process_cloth_clear_simulation(AppState& state)
    {
        m_cloth_sim_frames.clear();
        m_smpl_sim_frames.clear();
        m_collision_proxy_sim_frames.clear();
        m_collision_proxy_sim_F.resize(0, 3);
        m_cloth_sim_restart_pending = false;
        m_cloth_sim_reset_frame0_valid = false;
        m_cloth_sim_preview_state.reset();
        m_cloth_sim_preview_enabled.reset();
        m_cloth_sim_preview_last_seq_applied = 0;
        m_smplh_sim_preview_mesh_active = false;
        m_smplh_mesh_name.clear();
        state.controls.frame_count = 0;
        state.controls.current_frame = 0;
        state.controls.playing = false;
        state.controls.sim_steps_done = 0;
        state.controls.sim_steps_total = 0;
        state.controls.sim_pipeline_stage = 0;
        state.controls.sim_sequence_steps_done = 0;
        state.controls.sim_sequence_steps_total = 0;
        state.controls.sim_proxy_steps_done = 0;
        state.controls.sim_proxy_steps_total = 0;
        state.controls.sim_solver_steps_done = 0;
        state.controls.sim_solver_steps_total = 0;
        state.controls.sim_sequence_seconds = 0.0;
        state.controls.sim_proxy_seconds = 0.0;
        state.controls.sim_solver_seconds = 0.0;
        state.controls.sim_pipeline_seconds = 0.0;
        state.controls.sim_sequence_timing_details.clear();
        state.controls.sim_proxy_timing_details.clear();
        state.controls.sim_solver_timing_details.clear();
        state.controls.sim_stage.clear();
        console_add_line(state, "Cleared cloth simulation frames");
    }

    void Renderer::process_cloth_cancel_simulation(AppState& state)
    {
        if (m_cloth_sim_job_running && m_cloth_sim_cancel_flag)
        {
            m_cloth_sim_cancel_flag->store(true);
            state.controls.sim_stage = "Stopping";
            console_add_line(state, "Cancel requested for cloth simulation");
        }
    }

    void Renderer::process_cloth_simulate(AppState& state)
    {
        state.controls.sim_stage = "Preparing";

        if (m_cloth_sim_job_running)
        {
            m_cloth_sim_restart_pending = true;
            process_cloth_cancel_simulation(state);
            console_add_line(state, "Simulation restart requested; cancel requested");
            return;
        }

        state.controls.sim_pipeline_stage = 0;
        state.controls.sim_steps_done = 0;
        state.controls.sim_steps_total = 0;
        state.controls.sim_sequence_steps_done = 0;
        state.controls.sim_sequence_steps_total = 0;
        state.controls.sim_proxy_steps_done = 0;
        state.controls.sim_proxy_steps_total = 0;
        state.controls.sim_solver_steps_done = 0;
        state.controls.sim_solver_steps_total = 0;
        state.controls.sim_sequence_seconds = 0.0;
        state.controls.sim_proxy_seconds = 0.0;
        state.controls.sim_solver_seconds = 0.0;
        state.controls.sim_pipeline_seconds = 0.0;
        state.controls.sim_sequence_timing_details.clear();
        state.controls.sim_proxy_timing_details.clear();
        state.controls.sim_solver_timing_details.clear();

        m_cloth_sim_restart_pending = false;
        m_smpl_sim_frames.clear();

        if (m_garment_mesh_name.empty() || !state.show_garment_mesh)
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Simulate requested but no garment is loaded/visible");
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() == 0 || garment.F.rows() == 0)
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Simulate requested but garment mesh is empty");
            return;
        }

        ensure_smplh_mesh_loaded(state);
        if (!m_smplh_loaded || m_smplh_vertices.rows() == 0 || m_smplh_model.F.rows() == 0)
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Simulate requested but SMPL body is not available");
            return;
        }

        ensure_garment_to_smpl_mapping(state);
        if (!m_garment_to_smpl.valid()
            || m_garment_to_smpl.face_index.size() != m_garment_rest_vertices.rows())
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Simulate requested but cloth-to-SMPL mapping is not available");
            return;
        }

        Eigen::MatrixXi cloth_F = garment.F;
        Eigen::MatrixXi body_F = m_smplh_model.F;
        if (m_smplh_loaded_pose_mirror_x && body_F.cols() >= 3)
            body_F.col(1).swap(body_F.col(2));

        const int max_frames_for_progress = std::max(1, state.controls.max_frames);
        const double dt_for_steps = (double)state.xpbd_params.dt;
        const double duration_for_steps = (double)state.xpbd_params.duration_seconds;
        const int steps_target = (duration_for_steps <= 0.0) ? 0 : (int)std::ceil(duration_for_steps / dt_for_steps);
        const int steps = std::min(steps_target, std::max(0, max_frames_for_progress - 1));

        const int n_joints = m_smplh_model.joint_count();
        const int dof = 3 * n_joints;
        const bool smooth_sequence_pose_ready =
            state.controls.smooth_sequence
            && state.selection.pose_index >= 0
            && m_smplh_loaded
            && n_joints > 0
            && (int)m_smplh_loaded_pose_axis_angle.size() == dof
            && m_garment_rest_vertices.rows() > 0;
        const float smooth_sequence_frac = std::clamp(state.controls.smooth_sequence_length, 0.0f, 1.0f);
        const int smooth_sequence_steps = smooth_sequence_pose_ready
            ? static_cast<int>(std::round(smooth_sequence_frac * static_cast<float>(steps)))
            : 0;
        const bool use_smooth_sequence_start_pose = smooth_sequence_steps > 0;

        SmplhParams tpose_params;
        tpose_params.betas = m_smplh_params.betas;
        tpose_params.pose_axis_angle = Eigen::VectorXd::Zero(dof);
        if (use_smooth_sequence_start_pose)
            tpose_params.pose_axis_angle.head<3>() = m_smplh_loaded_pose_axis_angle.head<3>();

        MatrixX3dRM body_tpose_rm;
        std::string tpose_err;
        if (!smplh_evaluate_lbs(m_smplh_model, tpose_params, body_tpose_rm, &tpose_err))
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: Failed to build T-pose body for simulation: " + tpose_err);
            return;
        }

        Eigen::MatrixXd body_tpose = body_tpose_rm;
        Eigen::MatrixXd cloth_V0(m_garment_rest_vertices.rows(), 3);
        const auto& F_body_map = m_smplh_model.F;
        for (int vi = 0; vi < cloth_V0.rows(); ++vi)
        {
            const int fi = m_garment_to_smpl.face_index[vi];
            if (fi < 0 || fi >= F_body_map.rows())
            {
                cloth_V0.row(vi) = m_garment_rest_vertices.row(vi);
                continue;
            }

            const int ia = F_body_map(fi, 0);
            const int ib = F_body_map(fi, 1);
            const int ic = F_body_map(fi, 2);
            if (ia < 0 || ib < 0 || ic < 0
                || ia >= body_tpose.rows() || ib >= body_tpose.rows() || ic >= body_tpose.rows())
            {
                cloth_V0.row(vi) = m_garment_rest_vertices.row(vi);
                continue;
            }

            const Eigen::RowVector3d bc = m_garment_to_smpl.barycentric.row(vi);
            const Eigen::RowVector3d p_def =
                bc[0] * body_tpose.row(ia) + bc[1] * body_tpose.row(ib) + bc[2] * body_tpose.row(ic);

            Eigen::RowVector3d T_def, B_def, N_def;
            if (!core::triangle_frame(body_tpose.row(ia), body_tpose.row(ib), body_tpose.row(ic), T_def, B_def, N_def))
            {
                if (!core::triangle_frame(
                        m_smplh_model.v_template.row(ia),
                        m_smplh_model.v_template.row(ib),
                        m_smplh_model.v_template.row(ic),
                        T_def,
                        B_def,
                        N_def))
                {
                    cloth_V0.row(vi) = m_garment_rest_vertices.row(vi);
                    continue;
                }
            }

            const Eigen::RowVector3d off_local = m_garment_to_smpl.offset_local.row(vi);
            const Eigen::RowVector3d off_def = off_local[0] * T_def + off_local[1] * B_def + off_local[2] * N_def;
            cloth_V0.row(vi) = p_def + off_def;
        }

        if (m_smplh_loaded_pose_mirror_x)
        {
            body_tpose.col(0) *= -1.0;
            cloth_V0.col(0) *= -1.0;
        }

        apply_smpl_scene_transform(state, body_tpose);
        apply_smpl_scene_transform(state, cloth_V0);

        {
            ClothPushOutResult push_res;
            std::string push_err;
            const double offset = std::max(0.0, (double)state.garment_min_cloth_skin_distance);
            const bool push_ok = push_cloth_vertices_outside_body(
                cloth_V0,
                cloth_F,
                body_tpose,
                body_F,
                offset,
                &push_res,
                &push_err);
            if (!push_ok)
            {
                state.controls.simulating = false;
                state.controls.sim_stage.clear();
                console_add_line(state, "ERROR: Simulation T-pose cloth push-out failed: "
                    + (push_err.empty() ? std::string("(unknown error)") : push_err));
                return;
            }
            console_add_line(state,
                std::string("Simulation starts from ")
                    + (use_smooth_sequence_start_pose ? "root-aligned T-pose cloth" : "T-pose cloth")
                    + " with betas; push-out moved "
                    + std::to_string(push_res.moved_vertices) + " vertices");
        }

        std::vector<int> fixed_vertex_indices;
        if ((int)m_garment_constraints.size() == cloth_V0.rows())
        {
            fixed_vertex_indices.reserve(m_garment_constraints.size());
            for (int vi = 0; vi < (int)m_garment_constraints.size(); ++vi)
                if (m_garment_constraints[(size_t)vi])
                    fixed_vertex_indices.push_back(vi);
        }

        ClothToSmplMapping constraint_attachment_mapping = m_garment_to_smpl;
        if (!fixed_vertex_indices.empty())
        {
            ConstraintSkinProjectionResult constraint_res;
            std::string constraint_err;
            const double offset = std::max(0.0, (double)state.garment_min_cloth_skin_distance);
            const bool constraint_ok = project_constraints_to_skin_margin(
                cloth_V0,
                constraint_attachment_mapping,
                fixed_vertex_indices,
                body_tpose,
                body_F,
                offset,
                &constraint_res,
                &constraint_err);
            if (!constraint_ok)
            {
                state.controls.simulating = false;
                state.controls.sim_stage.clear();
                console_add_line(state, "ERROR: Simulation constrained cloth projection failed: "
                    + (constraint_err.empty() ? std::string("(unknown error)") : constraint_err));
                return;
            }
            console_add_line(state,
                "Constrained T-pose vertices corrected to skin margin: "
                + std::to_string(constraint_res.corrected_vertices)
                + " updated, pushed_out=" + std::to_string(constraint_res.pushed_out_vertices)
                + ", pushed_in=" + std::to_string(constraint_res.pulled_in_vertices));
        }

        Eigen::MatrixXd body_V = m_smplh_vertices;

        const auto wind_dir_from_angles = [](double yaw_deg, double pitch_deg)
        {
            constexpr double kPi = 3.14159265358979323846264338327950288;
            const double yaw = yaw_deg * (kPi / 180.0);
            const double pitch = pitch_deg * (kPi / 180.0);
            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            const double cp = std::cos(pitch);
            const double sp = std::sin(pitch);
            // Yaw about +Y, pitch about +X' (after yaw). Base forward is +Z.
            return Eigen::Vector3d(sy * cp, -sp, cy * cp);
        };

        ClothXpbdParams xpbd_params;
        xpbd_params.dt = (double)state.xpbd_params.dt;
        xpbd_params.duration_seconds = (double)state.xpbd_params.duration_seconds;
        xpbd_params.solver_iterations = state.xpbd_params.solver_iterations;
        xpbd_params.gravity_y = (double)state.xpbd_params.gravity;
        xpbd_params.damping = (double)state.xpbd_params.damping;

        xpbd_params.stretch_enabled = state.xpbd_params.stretch_enabled;
        xpbd_params.stretch_compliance = std::max(0.0, (double)state.xpbd_params.stretch_compliance);
        xpbd_params.base_stretch_mult = (double)state.xpbd_params.base_stretch_mult;
        xpbd_params.substeps = std::max(1, state.xpbd_params.substeps);
        xpbd_params.bend_enabled = state.xpbd_params.bend_enabled;
        xpbd_params.bend_compliance = std::max(0.0, (double)state.xpbd_params.bend_compliance);
        xpbd_params.collision_enabled = state.xpbd_params.collision_enabled;
        xpbd_params.collision_compliance = std::max(0.0, (double)state.xpbd_params.collision_compliance);
        xpbd_params.collision_thickness = (double)state.xpbd_params.collision_thickness;
        xpbd_params.self_collision_enabled = state.xpbd_params.self_collision_enabled;
        xpbd_params.self_collision_compliance = std::max(0.0, (double)state.xpbd_params.self_collision_compliance);
        xpbd_params.self_collision_thickness = (double)state.xpbd_params.self_collision_thickness;

        xpbd_params.enable_friction = state.xpbd_params.enable_friction;
        xpbd_params.friction_mu = (double)state.xpbd_params.friction_mu;

        xpbd_params.wind_dir = wind_dir_from_angles((double)state.xpbd_params.wind_yaw_deg, (double)state.xpbd_params.wind_pitch_deg);
        xpbd_params.wind_force = std::max(0.0, (double)state.xpbd_params.wind_force);
        xpbd_params.wind_random_space = std::max(0.0, (double)state.xpbd_params.wind_random_space);
        xpbd_params.wind_random_time = std::max(0.0, (double)state.xpbd_params.wind_random_time);

        xpbd_params.max_frames = max_frames_for_progress;

        const bool use_gpu_simulation = state.xpbd_params.device == SimulationDevice::Gpu;
        if (use_gpu_simulation && !state.gpu_simulation_available)
        {
            state.controls.simulating = false;
            state.controls.sim_stage.clear();
            console_add_line(state, "ERROR: CUDA simulation is unavailable.");
            return;
        }

        const bool save_smpl_sequence = state.controls.save_smpl_sequence;
        const CollisionGeometryType collision_geometry = state.xpbd_params.collision_geometry;
        const bool simplify_smpl = collision_geometry == CollisionGeometryType::SimplifiedSmpl;
        const bool use_simple_proxy = collision_geometry == CollisionGeometryType::SimpleProxy;
        const bool prepare_collision_proxy = simplify_smpl || use_simple_proxy;
        const double keep_percent = std::clamp((double)state.xpbd_params.smpl_geometry_percent, 1.0, 100.0);
        const int simplification_units = simplify_smpl && keep_percent < 99.5 ? 1 : 0;
        const int target_units = fixed_vertex_indices.empty() ? 0 : std::max(1, smooth_sequence_steps);
        const int proxy_units = xpbd_params.collision_enabled && prepare_collision_proxy ? smooth_sequence_steps + 1 : 0;
        const int sequence_units = smooth_sequence_steps + target_units + simplification_units;
        const int preparation_units = sequence_units + proxy_units;

        m_cloth_sim_progress_steps_total = preparation_units + steps;
        state.controls.sim_steps_total = m_cloth_sim_progress_steps_total;
        state.controls.sim_steps_done = 0;
        state.controls.sim_sequence_steps_total = sequence_units;
        state.controls.sim_sequence_steps_done = 0;
        state.controls.sim_proxy_steps_total = proxy_units;
        state.controls.sim_proxy_steps_done = 0;
        state.controls.sim_solver_steps_total = steps;
        state.controls.sim_solver_steps_done = 0;

        m_cloth_sim_frames.clear();
        state.controls.frame_count = 0;
        m_collision_proxy_sim_frames.clear();
        m_collision_proxy_sim_F.resize(0, 3);
        state.controls.current_frame = 0;
        state.controls.playing = false;

        // Snapshot frame 0 so preview updates don't corrupt the restart baseline.
        m_cloth_sim_reset_frame0 = cloth_V0;
        m_cloth_sim_reset_frame0_valid = true;
        m_cloth_sim_reset_pose_version = m_pose_version;
        m_cloth_sim_reset_garment_version = m_garment_version;

        m_cloth_sim_pose_version = m_pose_version;
        m_cloth_sim_garment_version = m_garment_version;

        m_cloth_sim_preview_state = std::make_shared<ClothSimPreviewState>();
        m_cloth_sim_preview_last_seq_applied = 0;
        m_cloth_sim_preview_enabled = std::make_shared<std::atomic_bool>(state.controls.preview);
        const auto preview_state = m_cloth_sim_preview_state;
        const auto preview_enabled = m_cloth_sim_preview_enabled;

        m_cloth_sim_cancel_flag = std::make_shared<std::atomic_bool>(false);
        const auto cancel_flag = m_cloth_sim_cancel_flag;

        m_cloth_sim_progress_steps_done = std::make_shared<std::atomic_int>(0);
        const auto progress_steps_done = m_cloth_sim_progress_steps_done;
        const int initial_progress_stage = sequence_units > 0 ? 0 : (proxy_units > 0 ? 3 : 4);
        m_cloth_sim_progress_stage = std::make_shared<std::atomic_int>(initial_progress_stage);
        const auto progress_stage = m_cloth_sim_progress_stage;

        const Eigen::RowVector3d display_scale(
            (double)state.scene_view_smpl_fit_scale[0],
            (double)state.scene_view_smpl_fit_scale[1],
            (double)state.scene_view_smpl_fit_scale[2]);
        const Eigen::RowVector3d display_translation(
            (double)state.scene_view_smpl_fit_translation[0],
            (double)state.scene_view_smpl_fit_translation[1],
            (double)state.scene_view_smpl_fit_translation[2]);
        const bool apply_display_transform = state.scene_view_smpl_fit_valid
            && display_scale.allFinite() && display_translation.allFinite();

        m_cloth_sim_job_running = true;
        state.controls.simulating = true;
        state.controls.sim_pipeline_stage = initial_progress_stage < 3 ? 0 : (initial_progress_stage == 3 ? 1 : 2);
        state.controls.sim_stage = initial_progress_stage < 3
            ? "Building SMPL sequence"
            : (initial_progress_stage == 3 ? "Preparing SMPL proxy" : "Simulating XPBD");
        console_add_line(state, use_gpu_simulation
            ? "Starting cloth simulation (CUDA Jacobi XPBD)..."
            : "Starting cloth simulation (CPU XPBD)...");

        m_cloth_sim_future = std::async(std::launch::async,
            [cloth_V0 = std::move(cloth_V0),
                cloth_F = std::move(cloth_F),
                body_V = std::move(body_V),
                body_F = std::move(body_F),
                xpbd_params,
                body_tpose = std::move(body_tpose),
                fixed_vertex_indices = std::move(fixed_vertex_indices),
                constraint_attachment_mapping = std::move(constraint_attachment_mapping),
                smplh_model = m_smplh_model,
                target_aa = m_smplh_loaded_pose_axis_angle,
                smpl_betas = m_smplh_params.betas,
                mirror_display_x = m_smplh_loaded_pose_mirror_x,
                display_scale,
                display_translation,
                apply_display_transform,
                smooth_sequence_steps,
                n_joints,
                dof,
                save_smpl_sequence,
                simplify_smpl,
                use_simple_proxy,
                prepare_collision_proxy,
                keep_percent,
                use_gpu_simulation,
                preparation_units,
                cancel_flag,
                progress_steps_done,
                progress_stage,
                preview_state,
                preview_enabled]() mutable -> ClothSimJobResult
            {
                ClothSimJobResult jr;
                std::string err;
                ClothXpbdResult xpbd_res;
                const auto job_start = std::chrono::steady_clock::now();

                const auto seconds_since = [](std::chrono::steady_clock::time_point start)
                {
                    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                };
                const auto append_timing = [](std::vector<std::string>& out, const char* label, double seconds)
                {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << label << ": " << seconds << "s";
                    out.push_back(oss.str());
                };

                const auto advance_progress = [&]() { progress_steps_done->fetch_add(1); };

                if (smooth_sequence_steps > 0)
                {
                    const auto step_start = std::chrono::steady_clock::now();
                    progress_stage->store(0);
                    const auto aa_to_quat = [](const Eigen::Vector3d& value)
                    {
                        const double angle = value.norm();
                        return angle < 1e-12
                            ? Eigen::Quaterniond::Identity()
                            : Eigen::Quaterniond(Eigen::AngleAxisd(angle, value / angle));
                    };
                    const Eigen::Quaterniond root_target = aa_to_quat(target_aa.segment<3>(0));
                    SmplhParams interp_params;
                    interp_params.betas = smpl_betas;
                    interp_params.pose_axis_angle.resize(dof);
                    xpbd_params.body_sequence.reserve((size_t)smooth_sequence_steps);
                    if (save_smpl_sequence)
                        jr.smpl_frames.reserve((size_t)smooth_sequence_steps);

                    for (int i = 0; i < smooth_sequence_steps; ++i)
                    {
                        if (cancel_flag->load())
                        {
                            jr.error = "Simulation preparation cancelled";
                            return jr;
                        }
                        const double t = smooth_sequence_steps == 1
                            ? 1.0 : (double)i / (double)(smooth_sequence_steps - 1);
                        for (int j = 0; j < n_joints; ++j)
                        {
                            const Eigen::Quaterniond target = aa_to_quat(target_aa.segment<3>(3 * j));
                            const Eigen::Quaterniond start = j == 0 ? root_target : Eigen::Quaterniond::Identity();
                            const Eigen::AngleAxisd result(start.slerp(t, target));
                            interp_params.pose_axis_angle.segment<3>(3 * j) = result.angle() * result.axis();
                        }

                        MatrixX3dRM skinned;
                        if (!smplh_evaluate_lbs(smplh_model, interp_params, skinned, &err))
                        {
                            jr.error = "LBS failed for smooth sequence step " + std::to_string(i) + ": " + err;
                            return jr;
                        }
                        Eigen::MatrixXd frame = skinned;
                        if (mirror_display_x)
                            frame.col(0) *= -1.0;
                        if (apply_display_transform)
                        {
                            frame.array().rowwise() *= display_scale.array();
                            frame.rowwise() += display_translation;
                        }
                        if (save_smpl_sequence)
                            jr.smpl_frames.push_back(frame);
                        xpbd_params.body_sequence.push_back(std::move(frame));
                        advance_progress();
                    }
                    const double seconds = seconds_since(step_start);
                    jr.sequence_seconds += seconds;
                    append_timing(jr.sequence_timing_details, "Smooth SMPL sequence", seconds);
                }

                if (!fixed_vertex_indices.empty())
                {
                    const auto step_start = std::chrono::steady_clock::now();
                    progress_stage->store(1);
                    Eigen::MatrixXd rest_body = smplh_model.v_template;
                    if (mirror_display_x)
                        rest_body.col(0) *= -1.0;
                    xpbd_params.fixed_vertex_indices = fixed_vertex_indices;
                    auto& targets = xpbd_params.fixed_vertex_target_sequence;
                    if (xpbd_params.body_sequence.empty())
                    {
                        targets.push_back(compute_cloth_smpl_targets(
                            fixed_vertex_indices, cloth_V0, constraint_attachment_mapping,
                            body_V, body_F, rest_body));
                        advance_progress();
                    }
                    else
                    {
                        targets.reserve(xpbd_params.body_sequence.size());
                        for (const Eigen::MatrixXd& frame : xpbd_params.body_sequence)
                        {
                            targets.push_back(compute_cloth_smpl_targets(
                                fixed_vertex_indices, cloth_V0, constraint_attachment_mapping,
                                frame, body_F, rest_body));
                            advance_progress();
                        }
                    }
                    const double seconds = seconds_since(step_start);
                    jr.sequence_seconds += seconds;
                    append_timing(jr.sequence_timing_details, "Cloth constraint targets", seconds);
                }

                if (simplify_smpl && keep_percent < 99.5)
                {
                    const auto step_start = std::chrono::steady_clock::now();
                    progress_stage->store(2);
                    SmplSimplification simplification;
                    Eigen::MatrixXd simplified_body;
                    bool simpl_ok = build_smpl_simplification(
                        body_tpose, body_F, smplh_model.weights, keep_percent,
                        simplification, &err)
                        && apply_smpl_simplification(body_V, simplification, simplified_body, &err);
                    std::vector<Eigen::MatrixXd> sequence;
                    if (simpl_ok)
                    {
                        sequence.reserve(xpbd_params.body_sequence.size());
                        for (const Eigen::MatrixXd& frame : xpbd_params.body_sequence)
                        {
                            Eigen::MatrixXd simplified_frame;
                            if (!apply_smpl_simplification(frame, simplification, simplified_frame, &err))
                            {
                                simpl_ok = false;
                                break;
                            }
                            sequence.push_back(std::move(simplified_frame));
                        }
                    }
                    if (simpl_ok)
                    {
                        body_V = std::move(simplified_body);
                        body_F = simplification.F;
                        xpbd_params.body_sequence = std::move(sequence);
                    }
                    err.clear();
                    advance_progress();
                    const double seconds = seconds_since(step_start);
                    jr.sequence_seconds += seconds;
                    append_timing(jr.sequence_timing_details, "SMPL simplification", seconds);
                }

                if (xpbd_params.collision_enabled && prepare_collision_proxy && body_V.rows() > 0 && body_F.rows() > 0)
                {
                    const auto step_start = std::chrono::steady_clock::now();
                    progress_stage->store(3);
                    if (use_simple_proxy)
                    {
                        SmplCollisionProxy proxy;
                        Eigen::MatrixXd proxy_body;
                        Eigen::MatrixXd proxy_capsules;
                        Eigen::MatrixXi proxy_F;
                        if (!build_smpl_collision_proxy(body_tpose, smplh_model, proxy, &err)
                            || !apply_smpl_collision_proxy(proxy, body_V, smplh_model, proxy_body, proxy_F, &err)
                            || !apply_smpl_collision_proxy_capsules(proxy, body_V, smplh_model, proxy_capsules, &err))
                        {
                            jr.error = "Simple collision proxy failed: " + err;
                            return jr;
                        }
                        body_V = std::move(proxy_body);
                        body_F = proxy_F;
                        xpbd_params.body_capsules = std::move(proxy_capsules);
                        jr.collision_proxy_F = proxy_F;
                        jr.collision_proxy_frames.push_back(body_V);
                        advance_progress();

                        std::vector<Eigen::MatrixXd> proxy_sequence;
                        std::vector<Eigen::MatrixXd> capsule_sequence;
                        proxy_sequence.reserve(xpbd_params.body_sequence.size());
                        capsule_sequence.reserve(xpbd_params.body_sequence.size());
                        jr.collision_proxy_frames.reserve(xpbd_params.body_sequence.size() + 1u);
                        for (const Eigen::MatrixXd& frame : xpbd_params.body_sequence)
                        {
                            if (cancel_flag->load())
                            {
                                jr.error = "Simulation preparation cancelled";
                                return jr;
                            }
                            Eigen::MatrixXd proxy_frame;
                            Eigen::MatrixXd capsule_frame;
                            Eigen::MatrixXi frame_F;
                            if (!apply_smpl_collision_proxy(proxy, frame, smplh_model, proxy_frame, frame_F, &err)
                                || !apply_smpl_collision_proxy_capsules(proxy, frame, smplh_model, capsule_frame, &err))
                            {
                                jr.error = "Simple collision proxy frame failed: " + err;
                                return jr;
                            }
                            proxy_sequence.push_back(proxy_frame);
                            capsule_sequence.push_back(std::move(capsule_frame));
                            jr.collision_proxy_frames.push_back(std::move(proxy_frame));
                            advance_progress();
                        }
                        xpbd_params.body_sequence = std::move(proxy_sequence);
                        xpbd_params.body_capsule_sequence = std::move(capsule_sequence);
                        append_timing(jr.proxy_timing_details, "Build simple capsule proxy", seconds_since(step_start));
                    }
                    else
                    {
                        const double inflation = std::clamp(
                            std::max(0.0015, 0.5 * std::max(0.0, xpbd_params.collision_thickness)),
                            0.0015, 0.008);
                        const double clearance = std::clamp(std::max(0.0020, 1.25 * inflation), 0.0020, 0.012);
                        prepare_xpbd_body_proxy_frame(body_V, body_F, inflation, clearance);
                        advance_progress();
                        for (Eigen::MatrixXd& frame : xpbd_params.body_sequence)
                        {
                            if (cancel_flag->load())
                            {
                                jr.error = "Simulation preparation cancelled";
                                return jr;
                            }
                            prepare_xpbd_body_proxy_frame(frame, body_F, inflation, clearance);
                            advance_progress();
                        }
                        append_timing(jr.proxy_timing_details, "Inflate and relax SMPL proxy", seconds_since(step_start));
                    }
                    jr.proxy_seconds += seconds_since(step_start);
                }

                progress_stage->store(4);
                progress_steps_done->store(preparation_units);

                struct PreviewCb
                {
                    std::shared_ptr<ClothSimPreviewState> state;
                    std::shared_ptr<std::atomic_bool> enabled;
                    std::shared_ptr<std::atomic_int> progress;
                    int progress_offset = 0;
                    Eigen::MatrixXi body_F;
                    std::chrono::steady_clock::time_point last;
                    std::chrono::milliseconds min_interval{ 100 }; // low-FPS preview
                };

                auto on_frame_xpbd = [](const Eigen::MatrixXd& frame_V, const Eigen::MatrixXd* body_V, int steps_simulated, void* user)
                {
                    auto* pcb = static_cast<PreviewCb*>(user);
                    if (!pcb || !pcb->state || !pcb->enabled)
                        return;
                    if (pcb->progress)
                        pcb->progress->store(pcb->progress_offset + steps_simulated);
                    if (!pcb->enabled->load())
                        return;

                    const auto now = std::chrono::steady_clock::now();
                    if (pcb->last.time_since_epoch().count() != 0 && (now - pcb->last) < pcb->min_interval)
                        return;
                    pcb->last = now;

                    {
                        std::lock_guard<std::mutex> lk(pcb->state->mutex);
                        pcb->state->V = frame_V;
                        pcb->state->valid = true;
                        if (body_V)
                        {
                            pcb->state->body_V = *body_V;
                            pcb->state->body_F = pcb->body_F;
                            pcb->state->body_valid = true;
                            pcb->state->body_topology_valid = (pcb->body_F.rows() > 0 && pcb->body_F.cols() == 3);
                        }
                        else
                        {
                            pcb->state->body_valid = false;
                            pcb->state->body_topology_valid = false;
                        }
                        pcb->state->seq.fetch_add(1, std::memory_order_release);
                    }
                };

                PreviewCb preview_cb;
                preview_cb.state = preview_state;
                preview_cb.enabled = preview_enabled;
                preview_cb.progress = progress_steps_done;
                preview_cb.progress_offset = preparation_units;
                preview_cb.body_F = body_F;

                const auto solver_start = std::chrono::steady_clock::now();
                bool ok = use_gpu_simulation
                    ? run_cloth_cuda_jacobi(
                        cloth_V0, cloth_F,
                        body_V, body_F,
                        xpbd_params,
                        &xpbd_res,
                        &err,
                        cancel_flag.get(),
                        nullptr,
                        on_frame_xpbd,
                        &preview_cb)
                    : run_cloth_xpbd(
                        cloth_V0, cloth_F,
                        body_V, body_F,
                        xpbd_params,
                        &xpbd_res,
                        &err,
                        cancel_flag.get(),
                        nullptr,
                        on_frame_xpbd,
                        &preview_cb);
                jr.solver_seconds = seconds_since(solver_start);
                if (ok)
                {
                    jr.frames = std::move(xpbd_res.frames);
                    jr.steps_simulated = xpbd_res.stats.steps_simulated;
                    jr.last_step_contacts = xpbd_res.stats.last_step_contacts + xpbd_res.stats.last_step_self_contacts;

                    const ClothXpbdStats& s = xpbd_res.stats;
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2);
                    const auto pct = [&](double ms)
                    {
                        return (s.ms_total > 0.0) ? (100.0 * ms / s.ms_total) : 0.0;
                    };

                    const double ms_collision_broadphase = s.ms_candidates + s.ms_collision_broad;
                    const double ms_collision_narrowphase = s.ms_collision_narrow;

                    oss << (use_gpu_simulation
                        ? "CUDA Jacobi XPBD timing breakdown (ms)\n"
                        : "CPU XPBD timing breakdown (ms)\n");
                    oss << "  total: " << s.ms_total << "\n";
                    oss << "  bvh build: " << s.ms_bvh_build << " (" << pct(s.ms_bvh_build) << "%)\n";
                    oss << "  predict: " << s.ms_predict << " (" << pct(s.ms_predict) << "%)\n";
                    oss << "  wind: " << s.ms_wind << " (" << pct(s.ms_wind) << "%)\n";
                    oss << "  stretch: " << s.ms_stretch << " (" << pct(s.ms_stretch) << "%)\n";
                    oss << "  bend: " << s.ms_bend << " (" << pct(s.ms_bend) << "%)\n";
                    oss << "  collision contacts total: " << s.ms_collision_bvh << " (" << pct(s.ms_collision_bvh) << "%)\n";
                    oss << "    broad phase: " << ms_collision_broadphase << " (" << pct(ms_collision_broadphase) << "%)\n";
                    oss << "      candidates (AABB): " << s.ms_candidates << " (" << pct(s.ms_candidates) << "%)\n";
                    oss << "      filters/bookkeeping: " << s.ms_collision_broad << " (" << pct(s.ms_collision_broad) << "%)\n";
                    oss << "    narrow phase (" << (use_simple_proxy ? "capsule query" : "BVH query") << "): "
                        << ms_collision_narrowphase << " (" << pct(ms_collision_narrowphase) << "%)\n";
                    oss << "  collision solve: " << s.ms_collision_solve << " (" << pct(s.ms_collision_solve) << "%)\n";
                    oss << "  self collision contacts: " << s.ms_self_collision_bvh << " (" << pct(s.ms_self_collision_bvh) << "%)\n";
                    oss << "  self collision solve: " << s.ms_self_collision_solve << " (" << pct(s.ms_self_collision_solve) << "%)\n";
                    oss << "  friction: " << s.ms_friction << " (" << pct(s.ms_friction) << "%)";
                    jr.timing_report = oss.str();

                    append_timing(jr.solver_timing_details, "XPBD wall time", jr.solver_seconds);
                    const auto append_ms = [&](const char* label, double ms)
                    {
                        std::ostringstream detail;
                        detail << std::fixed << std::setprecision(2)
                            << label << ": " << (ms / 1000.0) << "s (" << pct(ms) << "%)";
                        jr.solver_timing_details.push_back(detail.str());
                    };
                    append_ms("BVH build", s.ms_bvh_build);
                    append_ms("Predict", s.ms_predict);
                    append_ms("Wind", s.ms_wind);
                    append_ms("Stretch", s.ms_stretch);
                    append_ms("Bend", s.ms_bend);
                    append_ms("Collision contacts", s.ms_collision_bvh);
                    append_ms("Collision solve", s.ms_collision_solve);
                    append_ms("Self collision contacts", s.ms_self_collision_bvh);
                    append_ms("Self collision solve", s.ms_self_collision_solve);
                    append_ms("Friction", s.ms_friction);
                }

                jr.pipeline_seconds = seconds_since(job_start);
                jr.ok = ok;
                jr.error = std::move(err);
                return jr;
            });
    }

    void Renderer::update_cloth_simulation_job(AppState& state)
    {
        if (!m_cloth_sim_job_running)
            return;

        if (m_cloth_sim_preview_enabled)
            m_cloth_sim_preview_enabled->store(state.controls.preview);

        if (m_cloth_sim_progress_steps_done)
        {
            const int done = std::max(0, m_cloth_sim_progress_steps_done->load());
            state.controls.sim_steps_done = done;
            state.controls.sim_steps_total = std::max(0, m_cloth_sim_progress_steps_total);
            state.controls.sim_sequence_steps_done = std::clamp(
                done, 0, state.controls.sim_sequence_steps_total);
            state.controls.sim_proxy_steps_done = std::clamp(
                done - state.controls.sim_sequence_steps_total,
                0,
                state.controls.sim_proxy_steps_total);
            state.controls.sim_solver_steps_done = std::clamp(
                done - state.controls.sim_sequence_steps_total - state.controls.sim_proxy_steps_total,
                0,
                state.controls.sim_solver_steps_total);
        }
        if (m_cloth_sim_progress_stage)
        {
            static constexpr const char* stages[] = {
                "Building smooth SMPL sequence",
                "Building cloth constraints",
                "Simplifying SMPL",
                "Preparing SMPL proxy",
                "Simulating XPBD"
            };
            const int stage = std::clamp(m_cloth_sim_progress_stage->load(), 0, 4);
            state.controls.sim_stage = stages[stage];
            state.controls.sim_pipeline_stage = stage < 3 ? 0 : (stage == 3 ? 1 : 2);
        }

        if (!future_ready(m_cloth_sim_future))
            return;

        ClothSimJobResult jr = m_cloth_sim_future.get();
        m_cloth_sim_job_running = false;
        state.controls.simulating = false;

        const bool was_cancelled = (m_cloth_sim_cancel_flag && m_cloth_sim_cancel_flag->load());
        if (!was_cancelled)
        {
            state.controls.sim_steps_done = std::max(state.controls.sim_steps_done, state.controls.sim_steps_total);
            state.controls.sim_sequence_steps_done = state.controls.sim_sequence_steps_total;
            state.controls.sim_proxy_steps_done = state.controls.sim_proxy_steps_total;
            state.controls.sim_solver_steps_done = state.controls.sim_solver_steps_total;
        }

        m_cloth_sim_preview_state.reset();
        m_cloth_sim_preview_enabled.reset();
        m_cloth_sim_preview_last_seq_applied = 0;

        if (!jr.ok)
        {
            state.controls.sim_stage.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            m_cloth_sim_last_error = jr.error;
            console_add_line(state, std::string("ERROR: Cloth simulation failed: ") + (jr.error.empty() ? "(unknown error)" : jr.error));
            return;
        }

        if (m_cloth_sim_restart_pending)
        {
            // Discard result and restart immediately to avoid a transient "stopped" state.
            m_cloth_sim_frames.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            state.controls.frame_count = 0;
            state.controls.playing = false;
            state.controls.current_frame = 0;

            m_cloth_sim_restart_pending = false;
            console_add_line(state, "Restarting cloth simulation…");
            process_cloth_simulate(state);
            return;
        }

        if (m_cloth_sim_pose_version != m_pose_version || m_cloth_sim_garment_version != m_garment_version)
        {
            console_add_line(state, "Cloth simulation finished, but selection changed; result discarded");
            m_cloth_sim_frames.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            state.controls.frame_count = 0;
            state.controls.sim_stage.clear();
            return;
        }

        m_cloth_sim_frames = std::move(jr.frames);
        m_smpl_sim_frames = std::move(jr.smpl_frames);
        m_collision_proxy_sim_frames = std::move(jr.collision_proxy_frames);
        m_collision_proxy_sim_F = std::move(jr.collision_proxy_F);
        state.controls.frame_count = (int)m_cloth_sim_frames.size();
        state.controls.sim_sequence_seconds = jr.sequence_seconds;
        state.controls.sim_proxy_seconds = jr.proxy_seconds;
        state.controls.sim_solver_seconds = jr.solver_seconds;
        state.controls.sim_pipeline_seconds = jr.pipeline_seconds;
        state.controls.sim_sequence_timing_details = std::move(jr.sequence_timing_details);
        state.controls.sim_proxy_timing_details = std::move(jr.proxy_timing_details);
        state.controls.sim_solver_timing_details = std::move(jr.solver_timing_details);

        if (was_cancelled)
        {
            state.controls.current_frame = std::max(0, state.controls.frame_count - 1);
            state.controls.playing = false;
        }
        else
        {
            state.controls.current_frame = 0;
            state.controls.playing = (state.controls.frame_count > 1);
        }

        console_add_line(state,
            std::string(was_cancelled ? "Cloth simulation stopped: " : "Cloth simulation ready: ")
                + std::to_string(state.controls.frame_count) + " frames");
        state.controls.sim_stage = was_cancelled ? "Stopped" : "Ready";

        if (!jr.timing_report.empty())
        {
            std::istringstream iss(jr.timing_report);
            std::string line;
            while (std::getline(iss, line))
            {
                if (!line.empty())
                    console_add_line(state, line);
            }
        }
    }

    void Renderer::apply_cloth_simulation_frame(AppState& state)
    {
        if (m_garment_mesh_name.empty())
            return;
        if (m_cloth_sim_job_running)
        {
            if (!state.show_garment_mesh || !state.controls.preview || !m_cloth_sim_preview_state)
                return;

            if (m_cloth_sim_pose_version != m_pose_version || m_cloth_sim_garment_version != m_garment_version)
                return;

            const std::uint64_t seq = m_cloth_sim_preview_state->seq.load(std::memory_order_acquire);
            if (seq == m_cloth_sim_preview_last_seq_applied)
                return;

            auto& garment = m_viewer.data(m_garment_mesh_index);
            auto& body = m_viewer.data(m_smplh_mesh_index);

            Eigen::MatrixXd garment_V;
            bool want_body_update = false;
            Eigen::MatrixXd body_V;
            Eigen::MatrixXi body_F;
            bool body_topology_valid = false;
            {
                std::lock_guard<std::mutex> lk(m_cloth_sim_preview_state->mutex);
                if (!m_cloth_sim_preview_state->valid)
                    return;
                if (m_cloth_sim_preview_state->V.rows() != garment.V.rows())
                    return;
                garment_V = m_cloth_sim_preview_state->V;

                if (m_cloth_sim_preview_state->body_valid)
                {
                    want_body_update = true;
                    body_V = m_cloth_sim_preview_state->body_V;
                    body_topology_valid = m_cloth_sim_preview_state->body_topology_valid;
                    if (body_topology_valid)
                        body_F = m_cloth_sim_preview_state->body_F;
                }
                m_cloth_sim_preview_last_seq_applied = seq;
            }
            garment.set_vertices(garment_V);
            garment.compute_normals();

            if (want_body_update)
            {
                if (body_topology_valid
                    && (!m_smplh_sim_preview_mesh_active
                        || body.V.rows() != body_V.rows()
                        || body.F.rows() != body_F.rows()))
                {
                    body.clear();
                    body.set_mesh(body_V, body_F);
                    body.show_faces = 1;
                    body.show_lines = 0;
                    body.show_overlay = 0;
                    body.set_colors(Eigen::RowVector3d(0.85, 0.86, 0.88));
                    m_smplh_sim_preview_mesh_active = true;
                }
                else if (body.V.rows() == body_V.rows())
                {
                    body.set_vertices(body_V);
                }
                body.compute_normals();
                body.set_visible(true);
            }
            return;
        }

        if (m_cloth_sim_frames.empty())
            return;

        if (m_cloth_sim_pose_version != m_pose_version || m_cloth_sim_garment_version != m_garment_version)
        {
            m_cloth_sim_frames.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            state.controls.frame_count = 0;
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        const int frame_count = (int)m_cloth_sim_frames.size();
        state.controls.frame_count = frame_count;
        const int fi = clamp_frame(state.controls.current_frame, frame_count);

        if (fi < 0 || fi >= frame_count)
            return;
        if (m_cloth_sim_frames[(size_t)fi].rows() != garment.V.rows())
        {
            m_cloth_sim_frames.clear();
            m_smpl_sim_frames.clear();
            m_collision_proxy_sim_frames.clear();
            m_collision_proxy_sim_F.resize(0, 3);
            state.controls.frame_count = 0;
            return;
        }

        if (state.show_garment_mesh)
        {
            garment.set_vertices(m_cloth_sim_frames[(size_t)fi]);
            garment.compute_normals();
        }

        const bool want_proxy_sequence =
            state.controls.save_smpl_sequence
            && state.controls.show_collision_proxy_sequence
            && !m_collision_proxy_sim_frames.empty()
            && m_collision_proxy_sim_F.rows() > 0;

        if (want_proxy_sequence)
        {
            const int body_fi = std::min(fi, (int)m_collision_proxy_sim_frames.size() - 1);
            const Eigen::MatrixXd& body_V = m_collision_proxy_sim_frames[(size_t)body_fi];
            const Eigen::MatrixXi& body_F = m_collision_proxy_sim_F;
            auto& body = m_viewer.data(m_smplh_mesh_index);

            if (body_V.rows() > 0 && body_F.rows() > 0)
            {
                if (!m_smplh_sim_preview_mesh_active
                    || body.V.rows() != body_V.rows()
                    || body.F.rows() != body_F.rows())
                {
                    body.clear();
                    body.set_mesh(body_V, body_F);
                    body.show_faces = 1;
                    body.show_lines = 0;
                    body.show_overlay = 0;
                    body.set_colors(Eigen::RowVector3d(0.72, 0.80, 0.92));
                    m_smplh_sim_preview_mesh_active = true;
                }
                else
                {
                    body.set_vertices(body_V);
                }
                body.compute_normals();
                body.set_visible(true);
            }
        }
        else if (!m_smpl_sim_frames.empty())
        {
            const int body_fi = std::min(fi, (int)m_smpl_sim_frames.size() - 1);
            const Eigen::MatrixXd& body_V = m_smpl_sim_frames[(size_t)body_fi];
            auto& body = m_viewer.data(m_smplh_mesh_index);

            if (body_V.rows() == m_smplh_model.v_template.rows())
            {
                if (m_smplh_sim_preview_mesh_active
                    || body.V.rows() != body_V.rows()
                    || body.F.rows() != m_smplh_model.F.rows())
                {
                    Eigen::MatrixXi body_F = m_smplh_model.F;
                    if (m_smplh_loaded_pose_mirror_x && body_F.cols() >= 3)
                        body_F.col(1).swap(body_F.col(2));
                    body.clear();
                    body.set_mesh(body_V, body_F);
                    body.show_faces = 1;
                    body.show_lines = 0;
                    body.show_overlay = 0;
                    body.set_colors(Eigen::RowVector3d(0.85, 0.86, 0.88));
                    m_smplh_sim_preview_mesh_active = false;
                }
                else
                {
                    body.set_vertices(body_V);
                }
                body.compute_normals();
                body.set_visible(true);
            }
        }
    }
}
