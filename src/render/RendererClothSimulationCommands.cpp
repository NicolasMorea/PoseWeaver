#include "render/Renderer.h"

#include "core/Geometry.h"
#include "simulation/ClothMeshProcessor.h"
#include "simulation/SmplSimplifier.h"
#include "simulation/pd/ClothProjectiveDynamics.h"
#include "simulation/xpbd/ClothXpbd.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <optional>
#include <sstream>
#include <utility>

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
    }

    void Renderer::process_cloth_clear_simulation(AppState& state)
    {
        m_cloth_sim_frames.clear();
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

        m_cloth_sim_restart_pending = false;

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

        SmplhParams tpose_params;
        tpose_params.betas = m_smplh_params.betas;
        tpose_params.pose_axis_angle = Eigen::VectorXd::Zero(3 * m_smplh_model.joint_count());

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

        if (state.scene_view_raw_rig_root_valid
            && m_smplh_model.J_regressor.rows() > 0
            && m_smplh_model.J_regressor.cols() == body_tpose.rows())
        {
            const Eigen::MatrixXd tpose_joints = m_smplh_model.J_regressor * body_tpose;
            if (tpose_joints.rows() > 0 && tpose_joints.cols() == 3)
            {
                const Eigen::RowVector3d current_root = tpose_joints.row(0);
                const Eigen::RowVector3d target_root(
                    (double)state.scene_view_raw_rig_root[0],
                    (double)state.scene_view_raw_rig_root[1],
                    (double)state.scene_view_raw_rig_root[2]);
                body_tpose.rowwise() -= current_root;
                body_tpose.rowwise() += target_root;
                cloth_V0.rowwise() -= current_root;
                cloth_V0.rowwise() += target_root;
            }
        }

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
                std::string("Simulation starts from T-pose cloth with betas; push-out moved ")
                    + std::to_string(push_res.moved_vertices) + " vertices");
        }

        Eigen::MatrixXd body_V = m_smplh_vertices;

        const SimulationSolver solver = state.controls.solver;

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

        std::optional<ClothPdParams> pd_params;
        std::optional<ClothXpbdParams> xpbd_params;

        double dt_for_progress = 1.0 / 60.0;
        double duration_for_progress = 0.0;
        int max_frames_for_progress = std::max(1, state.controls.max_frames);

        if (solver == SimulationSolver::ProjectiveDynamics)
        {
            ClothPdParams p;
            p.dt = (double)state.params.dt;
            p.duration_seconds = (double)state.params.duration_seconds;
            p.local_global_iterations = state.params.solver_iterations;
            p.gravity_y = (double)state.params.gravity;
            p.stretch_stiffness = state.params.stretch_enabled ? (double)state.params.stretch_stiffness : 0.0;
            p.base_stretch_mult = (double)state.params.base_stretch_mult;
            p.bend_stiffness = state.params.bend_enabled ? (double)state.params.bend_stiffness : 0.0;
            p.iso_bend_stiffness = (state.params.bend_enabled && state.params.iso_bend_enabled) ? (double)state.params.iso_bend_stiffness : 0.0;
            p.collision_stiffness = state.params.collision_enabled ? (double)state.params.collision_stiffness : 0.0;
            p.collision_thickness = (double)state.params.collision_thickness;
            p.enable_friction = state.params.enable_friction;
            p.friction_mu = (double)state.params.friction_mu;
            p.damping = (double)state.params.damping;
            p.max_frames = max_frames_for_progress;

            p.wind_dir = wind_dir_from_angles((double)state.params.wind_yaw_deg, (double)state.params.wind_pitch_deg);
            p.wind_force = std::max(0.0, (double)state.params.wind_force);
            p.wind_random_space = std::max(0.0, (double)state.params.wind_random_space);
            p.wind_random_time = std::max(0.0, (double)state.params.wind_random_time);

            dt_for_progress = p.dt;
            duration_for_progress = p.duration_seconds;
            pd_params = p;
        }
        else
        {
            ClothXpbdParams p;
            p.dt = (double)state.xpbd_params.dt;
            p.duration_seconds = (double)state.xpbd_params.duration_seconds;
            p.solver_iterations = state.xpbd_params.solver_iterations;
            p.gravity_y = (double)state.xpbd_params.gravity;
            p.damping = (double)state.xpbd_params.damping;

            p.stretch_enabled = state.xpbd_params.stretch_enabled;
            p.stretch_compliance = std::max(0.0, (double)state.xpbd_params.stretch_compliance);
            p.base_stretch_mult = (double)state.xpbd_params.base_stretch_mult;
            p.substeps = std::max(1, state.xpbd_params.substeps);
            p.bend_enabled = state.xpbd_params.bend_enabled;
            p.bend_compliance = std::max(0.0, (double)state.xpbd_params.bend_compliance);
            p.collision_enabled = state.xpbd_params.collision_enabled;
            p.collision_compliance = std::max(0.0, (double)state.xpbd_params.collision_compliance);
            p.collision_thickness = (double)state.xpbd_params.collision_thickness;
            p.self_collision_enabled = state.xpbd_params.self_collision_enabled;
            p.self_collision_compliance = std::max(0.0, (double)state.xpbd_params.self_collision_compliance);
            p.self_collision_thickness = (double)state.xpbd_params.self_collision_thickness;

            p.enable_friction = state.xpbd_params.enable_friction;
            p.friction_mu = (double)state.xpbd_params.friction_mu;

            p.wind_dir = wind_dir_from_angles((double)state.xpbd_params.wind_yaw_deg, (double)state.xpbd_params.wind_pitch_deg);
            p.wind_force = std::max(0.0, (double)state.xpbd_params.wind_force);
            p.wind_random_space = std::max(0.0, (double)state.xpbd_params.wind_random_space);
            p.wind_random_time = std::max(0.0, (double)state.xpbd_params.wind_random_time);

            p.max_frames = max_frames_for_progress;

            dt_for_progress = p.dt;
            duration_for_progress = p.duration_seconds;
            xpbd_params = p;
        }

        const double dt = dt_for_progress;
        const int max_frames = std::max(1, max_frames_for_progress);
        const int steps_target = (duration_for_progress <= 0.0) ? 0 : (int)std::ceil(duration_for_progress / dt);
        const int steps = std::min(steps_target, std::max(0, max_frames - 1));

        if (state.controls.smooth_sequence
            && xpbd_params.has_value()
            && state.selection.pose_index >= 0
            && m_smplh_loaded
            && m_smplh_model.joint_count() > 0
            && (int)m_smplh_loaded_pose_axis_angle.size() == 3 * m_smplh_model.joint_count()
            && m_garment_rest_vertices.rows() > 0
            && m_garment_rest_vertices.rows() == cloth_V0.rows())
        {
            const int n_joints = m_smplh_model.joint_count();
            const int dof = 3 * n_joints;
            const Eigen::VectorXd& target_aa = m_smplh_loaded_pose_axis_angle;

            const auto aa_to_quat = [](const Eigen::Vector3d& aa_vec)
            {
                const double angle = aa_vec.norm();
                if (angle < 1e-12)
                    return Eigen::Quaterniond::Identity();
                return Eigen::Quaterniond(Eigen::AngleAxisd(angle, aa_vec / angle));
            };

            const Eigen::Quaterniond q_root_target = aa_to_quat(target_aa.segment<3>(0));

            const bool mirror_display_x = m_smplh_loaded_pose_mirror_x;
            const bool align_display_root =
                state.scene_view_raw_rig_root_valid
                && m_smplh_model.J_regressor.rows() > 0
                && m_smplh_model.J_regressor.cols() == m_smplh_model.vertex_count();
            const Eigen::RowVector3d display_root_target(
                (double)state.scene_view_raw_rig_root[0],
                (double)state.scene_view_raw_rig_root[1],
                (double)state.scene_view_raw_rig_root[2]);

            const auto to_display_space_body = [&](Eigen::MatrixXd& V_body)
            {
                if (mirror_display_x && V_body.cols() >= 1)
                    V_body.col(0) *= -1.0;
                if (align_display_root)
                {
                    const Eigen::MatrixXd joints = m_smplh_model.J_regressor * V_body;
                    if (joints.rows() > 0 && joints.cols() == 3)
                    {
                        const Eigen::RowVector3d current_root = joints.row(0);
                        V_body.rowwise() -= current_root;
                        V_body.rowwise() += display_root_target;
                    }
                }
            };

            const float seq_frac = std::clamp(state.controls.smooth_sequence_length, 0.0f, 1.0f);
            const int seq_steps = static_cast<int>(std::round(seq_frac * static_cast<float>(steps)));

            if (seq_steps > 0)
            {
                console_add_line(state,
                    std::string("Smooth sequence: computing ") + std::to_string(seq_steps) +
                    "/" + std::to_string(steps) + " interpolated body frames...");

                SmplhParams interp_params;
                interp_params.betas = m_smplh_params.betas;
                interp_params.pose_axis_angle.resize(dof);

                auto& body_seq = xpbd_params->body_sequence;
                body_seq.reserve(static_cast<size_t>(seq_steps));

                bool seq_ok = true;
                for (int i = 0; i < seq_steps && seq_ok; ++i)
                {
                    const double t = (seq_steps == 1) ? 1.0 : static_cast<double>(i) / static_cast<double>(seq_steps - 1);

                    for (int j = 0; j < n_joints; ++j)
                    {
                        const Eigen::Quaterniond q_target = aa_to_quat(target_aa.segment<3>(3 * j));
                        const Eigen::Quaterniond q_start = (j == 0) ? q_root_target : Eigen::Quaterniond::Identity();
                        const Eigen::Quaterniond q_interp = q_start.slerp(t, q_target);

                        const Eigen::AngleAxisd aa_out(q_interp);
                        interp_params.pose_axis_angle.segment<3>(3 * j) = aa_out.angle() * aa_out.axis();
                    }

                    MatrixX3dRM V_skinned;
                    std::string eval_err;
                    if (!smplh_evaluate_lbs(m_smplh_model, interp_params, V_skinned, &eval_err))
                    {
                        console_add_line(state,
                            std::string("ERROR: LBS failed for smooth sequence step ") +
                            std::to_string(i) + ": " + eval_err);
                        seq_ok = false;
                        break;
                    }
                    Eigen::MatrixXd body_frame = V_skinned;
                    to_display_space_body(body_frame);
                    body_seq.push_back(std::move(body_frame));
                }

                if (seq_ok)
                {
                    console_add_line(state, "Smooth sequence ready; cloth starts from pushed-out betas T-pose.");
                }
                else
                {
                    body_seq.clear();
                }
            }
            else
            {
                // seq_steps == 0: toggle is on but length is 0% — start cloth from T-pose with static target body.
            }
        }
        if (xpbd_params.has_value())
        {
            const double keep_percent = std::clamp((double)state.xpbd_params.smpl_geometry_percent, 1.0, 100.0);
            if (keep_percent < 99.5)
            {
                state.controls.sim_stage = "Simplifying SMPL";
                const auto t0 = std::chrono::steady_clock::now();

                SmplSimplification simplification;
                std::string simpl_err;
                Eigen::MatrixXd simplified_body_V;
                bool simpl_ok = build_smpl_simplification(
                    body_tpose,
                    body_F,
                    m_smplh_model.weights,
                    keep_percent,
                    simplification,
                    &simpl_err)
                    && apply_smpl_simplification(body_V, simplification, simplified_body_V, &simpl_err);

                std::vector<Eigen::MatrixXd> simplified_body_sequence;
                if (simpl_ok && !xpbd_params->body_sequence.empty())
                {
                    simplified_body_sequence.reserve(xpbd_params->body_sequence.size());
                    for (const Eigen::MatrixXd& frame : xpbd_params->body_sequence)
                    {
                        Eigen::MatrixXd simplified_frame;
                        if (!apply_smpl_simplification(frame, simplification, simplified_frame, &simpl_err))
                        {
                            simpl_ok = false;
                            break;
                        }
                        simplified_body_sequence.push_back(std::move(simplified_frame));
                    }
                }

                if (simpl_ok)
                {
                    const int raw_vertices = (int)body_V.rows();
                    const int raw_faces = (int)body_F.rows();
                    body_V = std::move(simplified_body_V);
                    body_F = simplification.F;
                    if (!xpbd_params->body_sequence.empty())
                        xpbd_params->body_sequence = std::move(simplified_body_sequence);

                    const auto elapsed = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2)
                        << "XPBD SMPL simplification: kept "
                        << body_V.rows() << "/" << raw_vertices << " vertices, "
                        << body_F.rows() << "/" << raw_faces << " faces"
                        << " (requested " << keep_percent << "%, " << elapsed << " ms)";
                    console_add_line(state, oss.str());
                }
                else
                {
                    console_add_line(state,
                        std::string("WARNING: XPBD SMPL simplification skipped: ")
                            + (simpl_err.empty() ? std::string("(unknown error)") : simpl_err));
                }
            }
        }

        m_cloth_sim_progress_steps_total = steps;
        state.controls.sim_steps_total = steps;
        state.controls.sim_steps_done = 0;

        m_cloth_sim_frames.clear();
        state.controls.frame_count = 0;
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

        m_cloth_sim_job_running = true;
        state.controls.simulating = true;
        state.controls.sim_stage = (solver == SimulationSolver::ProjectiveDynamics) ? "Simulating PD" : "Simulating XPBD";
        console_add_line(state,
            (solver == SimulationSolver::ProjectiveDynamics)
                ? "Starting cloth simulation (Projective Dynamics)…"
                : "Starting cloth simulation (XPBD)…");

        m_cloth_sim_future = std::async(std::launch::async,
            [cloth_V0 = std::move(cloth_V0),
                cloth_F = std::move(cloth_F),
                body_V = std::move(body_V),
                body_F = std::move(body_F),
                pd_params,
                xpbd_params,
                cancel_flag,
                progress_steps_done,
                preview_state,
                preview_enabled]() mutable -> ClothSimJobResult
            {
                ClothSimJobResult jr;
                std::string err;
                ClothPdResult pd_res;
                ClothXpbdResult xpbd_res;

                struct PreviewCb
                {
                    std::shared_ptr<ClothSimPreviewState> state;
                    std::shared_ptr<std::atomic_bool> enabled;
                    Eigen::MatrixXi body_F;
                    std::chrono::steady_clock::time_point last;
                    std::chrono::milliseconds min_interval{ 100 }; // low-FPS preview
                };

                auto on_frame_pd = [](const Eigen::MatrixXd& frame_V, int /*steps_simulated*/, void* user)
                {
                    auto* pcb = static_cast<PreviewCb*>(user);
                    if (!pcb || !pcb->state || !pcb->enabled)
                        return;
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
                        pcb->state->seq.fetch_add(1, std::memory_order_release);
                    }
                };

                auto on_frame_xpbd = [](const Eigen::MatrixXd& frame_V, const Eigen::MatrixXd* body_V, int /*steps_simulated*/, void* user)
                {
                    auto* pcb = static_cast<PreviewCb*>(user);
                    if (!pcb || !pcb->state || !pcb->enabled)
                        return;
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
                preview_cb.body_F = body_F;

                bool ok = false;
                if (pd_params.has_value())
                {
                    ok = run_cloth_projective_dynamics(
                        cloth_V0, cloth_F,
                        body_V, body_F,
                        *pd_params,
                        &pd_res,
                        &err,
                        cancel_flag.get(),
                        progress_steps_done.get(),
                        on_frame_pd,
                        &preview_cb);
                    if (ok)
                    {
                        jr.frames = std::move(pd_res.frames);
                        jr.steps_simulated = pd_res.stats.steps_simulated;
                        jr.last_step_contacts = pd_res.stats.last_step_contacts;

                        const ClothPdStats& s = pd_res.stats;
                        std::ostringstream oss;
                        oss << std::fixed << std::setprecision(2);
                        oss << "PD timing breakdown (ms)\n";
                        oss << "  total: " << s.ms_total << "\n";
                        oss << "  bvh build: " << s.ms_bvh_build << " (" << (s.ms_total > 0 ? 100.0 * s.ms_bvh_build / s.ms_total : 0.0) << "%)\n";
                        oss << "  predict: " << s.ms_predict << " (" << (s.ms_total > 0 ? 100.0 * s.ms_predict / s.ms_total : 0.0) << "%)\n";
                        oss << "  wind: " << s.ms_wind << " (" << (s.ms_total > 0 ? 100.0 * s.ms_wind / s.ms_total : 0.0) << "%)\n";
                        oss << "  collision contacts total: " << s.ms_contacts_bvh << " (" << (s.ms_total > 0 ? 100.0 * s.ms_contacts_bvh / s.ms_total : 0.0) << "%)\n";
                        oss << "    narrow phase (BVH query): " << s.ms_contacts_query << " (" << (s.ms_total > 0 ? 100.0 * s.ms_contacts_query / s.ms_total : 0.0) << "%)\n";
                        oss << "    broad phase (filters): " << s.ms_contacts_other << " (" << (s.ms_total > 0 ? 100.0 * s.ms_contacts_other / s.ms_total : 0.0) << "%)\n";
                        oss << "  assemble: " << s.ms_assemble << " (" << (s.ms_total > 0 ? 100.0 * s.ms_assemble / s.ms_total : 0.0) << "%)\n";
                        oss << "  factor: " << s.ms_factor << " (" << (s.ms_total > 0 ? 100.0 * s.ms_factor / s.ms_total : 0.0) << "%)\n";
                        oss << "  local stretch: " << s.ms_local_stretch << " (" << (s.ms_total > 0 ? 100.0 * s.ms_local_stretch / s.ms_total : 0.0) << "%)\n";
                        oss << "  local bend: " << s.ms_local_bend << " (" << (s.ms_total > 0 ? 100.0 * s.ms_local_bend / s.ms_total : 0.0) << "%)\n";
                        oss << "  local collision: " << s.ms_local_collision << " (" << (s.ms_total > 0 ? 100.0 * s.ms_local_collision / s.ms_total : 0.0) << "%)\n";
                        oss << "  global solve: " << s.ms_global_solve << " (" << (s.ms_total > 0 ? 100.0 * s.ms_global_solve / s.ms_total : 0.0) << "%)\n";
                        oss << "  post pushout: " << s.ms_post_pushout << " (" << (s.ms_total > 0 ? 100.0 * s.ms_post_pushout / s.ms_total : 0.0) << "%)\n";
                        oss << "  friction: " << s.ms_friction << " (" << (s.ms_total > 0 ? 100.0 * s.ms_friction / s.ms_total : 0.0) << "%)";
                        jr.timing_report = oss.str();
                    }
                }
                else if (xpbd_params.has_value())
                {
                    ok = run_cloth_xpbd(
                        cloth_V0, cloth_F,
                        body_V, body_F,
                        *xpbd_params,
                        &xpbd_res,
                        &err,
                        cancel_flag.get(),
                        progress_steps_done.get(),
                        on_frame_xpbd,
                        &preview_cb);
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

                        oss << "XPBD timing breakdown (ms)\n";
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
                        oss << "    narrow phase (BVH query): " << ms_collision_narrowphase << " (" << pct(ms_collision_narrowphase) << "%)\n";
                        oss << "  collision solve: " << s.ms_collision_solve << " (" << pct(s.ms_collision_solve) << "%)\n";
                        oss << "  self collision contacts: " << s.ms_self_collision_bvh << " (" << pct(s.ms_self_collision_bvh) << "%)\n";
                        oss << "  self collision solve: " << s.ms_self_collision_solve << " (" << pct(s.ms_self_collision_solve) << "%)\n";
                        oss << "  friction: " << s.ms_friction << " (" << pct(s.ms_friction) << "%)";
                        jr.timing_report = oss.str();
                    }
                }

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
            state.controls.sim_steps_done = std::max(0, m_cloth_sim_progress_steps_done->load());
            state.controls.sim_steps_total = std::max(0, m_cloth_sim_progress_steps_total);
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
        }

        m_cloth_sim_preview_state.reset();
        m_cloth_sim_preview_enabled.reset();
        m_cloth_sim_preview_last_seq_applied = 0;

        if (!jr.ok)
        {
            state.controls.sim_stage.clear();
            m_cloth_sim_last_error = jr.error;
            console_add_line(state, std::string("ERROR: Cloth simulation failed: ") + (jr.error.empty() ? "(unknown error)" : jr.error));
            return;
        }

        if (m_cloth_sim_restart_pending)
        {
            // Discard result and restart immediately to avoid a transient "stopped" state.
            m_cloth_sim_frames.clear();
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
            state.controls.frame_count = 0;
            state.controls.sim_stage.clear();
            return;
        }

        m_cloth_sim_frames = std::move(jr.frames);
        state.controls.frame_count = (int)m_cloth_sim_frames.size();

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
        if (!state.show_garment_mesh || m_garment_mesh_name.empty())
            return;
        if (m_cloth_sim_job_running)
        {
            if (!state.controls.preview || !m_cloth_sim_preview_state)
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
            state.controls.frame_count = 0;
            return;
        }

        garment.set_vertices(m_cloth_sim_frames[(size_t)fi]);
        garment.compute_normals();
    }
}
