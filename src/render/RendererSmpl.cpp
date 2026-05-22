#include "render/Renderer.h"
#include "core/Paths.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>

namespace poseweaver
{
    bool Renderer::ensure_smplh_model_loaded(AppState& state, SmplType gender)
    {
        if (m_smplh_loaded && m_smplh_loaded_gender == gender)
            return true;

        const std::filesystem::path model_path = assets_dir() / "smplh" / ((gender == SmplType::Male) ? "male" : "female") / "model.npz";
        std::string err;
        if (!m_smplh_model.load_from_npz(model_path, &err))
        {
            std::fprintf(stderr, "[PoseWeaver] Failed to load SMPLH model: %s\n", err.c_str());
            console_add_line(state, std::string("ERROR: Failed to load SMPLH model: ") + err);
            m_smplh_loaded = false;
            return false;
        }

        console_add_line(state, std::string("Loaded SMPLH model: ") + ((gender == SmplType::Male) ? "male" : "female"));
        console_add_line(state,
            std::string("SMPLH mesh: ")
                + std::to_string(m_smplh_model.vertex_count()) + " vertices, "
                + std::to_string(m_smplh_model.face_count()) + " faces");
        m_pose_version++;

        m_smplh_loaded = true;
        m_smplh_loaded_gender = gender;

        m_smpl_uv_layout_loaded = false;
        m_smpl_uv_layout_source.clear();
        m_smpl_uv_coords.resize(0, 0);
        m_smpl_F_uv.resize(0, 0);

        m_smpl_uv_vertex_colors.resize(0, 0);
        m_smpl_uv_per_vertex.resize(0, 0);
        m_smpl_uv_vertex_colors_valid = false;
        m_last_smpl_color_by_uv = false;

        m_garment_to_smpl.clear();
        m_garment_mapping_mesh_name.clear();
        m_garment_uv_segments.clear();
        m_garment_uv_triangles.clear();
        m_garment_uv_overlay_valid = false;
        m_garment_uv_vertex_colors.resize(0, 0);
        m_garment_uv_vertex_colors_valid = false;
        m_last_garment_color_by_smpl_uv = false;

        m_smplh_vertices.resize(0, 0);
        m_smplh_deform_dirty = true;
        m_smplh_loaded_pose_name.clear();
        m_smplh_loaded_pose_axis_angle.resize(0);
        m_smplh_last_pose_reload_request_id = 0;
        m_smplh_loaded_pose_mirror_x = false;
        m_smplh_mesh_mirror_x = false;

        m_smplh_params.betas = Eigen::VectorXd::Zero(m_smplh_model.beta_count());
        m_smplh_params.pose_axis_angle = Eigen::VectorXd::Zero(3 * m_smplh_model.joint_count());

        return true;
    }

    void Renderer::ensure_smplh_mesh_loaded(AppState& state)
    {
        const SmplType gender = state.selection.gender;
        const std::string desired = (gender == SmplType::Male) ? "SMPLH:Male" : "SMPLH:Female";
        auto& data = m_viewer.data(m_smplh_mesh_index);

        const bool want_visible = state.show_smpl_character;
        if (m_smplh_sim_preview_mesh_active && m_cloth_sim_job_running)
        {
            data.set_visible(state.controls.preview ? true : want_visible);
            return;
        }
        if (m_smplh_sim_preview_mesh_active && !m_cloth_sim_job_running)
        {
            m_smplh_sim_preview_mesh_active = false;
            m_smplh_mesh_name.clear();
        }

        if (!m_smplh_last_show_character && want_visible)
            m_smplh_deform_dirty = true;
        m_smplh_last_show_character = want_visible;

        if (!ensure_smplh_model_loaded(state, gender))
        {
            data.clear();
            data.set_visible(false);
            m_smplh_mesh_name.clear();
            return;
        }

        {
            const int B = m_smplh_model.beta_count();
            if (m_smplh_params.betas.size() != B)
            {
                m_smplh_params.betas = Eigen::VectorXd::Zero(B);
                m_smplh_deform_dirty = true;
            }

            const int n = std::min<int>((int)state.smpl_shape.betas.size(), B);
            for (int i = 0; i < n; ++i)
            {
                const double v = (double)state.smpl_shape.betas[(size_t)i];
                if (m_smplh_params.betas[i] != v)
                {
                    m_smplh_params.betas[i] = v;
                    m_smplh_deform_dirty = true;
                }
            }
            for (int i = n; i < B; ++i)
            {
                if (m_smplh_params.betas[i] != 0.0)
                {
                    m_smplh_params.betas[i] = 0.0;
                    m_smplh_deform_dirty = true;
                }
            }
        }

        {
            std::string pose_name;
            if (state.selection.pose_index >= 0 && state.selection.pose_index < (int)state.selection.poses.size())
                pose_name = state.selection.poses[(size_t)state.selection.pose_index];

            const bool pose_reload_requested =
                state.pose_asset_reload_request_id != m_smplh_last_pose_reload_request_id;
            if (pose_name != m_smplh_loaded_pose_name || pose_reload_requested)
            {
                m_smplh_loaded_pose_name = pose_name;
                m_smplh_last_pose_reload_request_id = state.pose_asset_reload_request_id;
                m_smplh_loaded_pose_axis_angle.resize(0);
                m_smplh_loaded_pose_mirror_x = false;
                m_pose_version++;

                m_smplh_deform_dirty = true;

                if (!pose_name.empty())
                {
                    console_add_line(state, std::string("Loading pose: ") + pose_name + " (root rotated -90° about +X)");
                    const std::filesystem::path pose_path = assets_dir() / "poses" / pose_name;
                    std::string pose_err;
                    Eigen::VectorXd pose_axis_angle;
                    bool mirror_x = false;
                    if (!smplh_load_pose_npz(pose_path, pose_axis_angle, nullptr, &pose_err, &mirror_x))
                    {
                        std::fprintf(stderr, "[PoseWeaver] Failed to load pose '%s': %s\n", pose_name.c_str(), pose_err.c_str());
                        console_add_line(state, std::string("ERROR: Failed to load pose '") + pose_name + "': " + pose_err);
                    }
                    else
                    {
                        m_smplh_loaded_pose_axis_angle = std::move(pose_axis_angle);
                        m_smplh_loaded_pose_mirror_x = mirror_x;
                        console_add_line(state, std::string("Pose loaded: ") + pose_name);
                    }
                }
                else
                {
                    console_add_line(state, "Pose set: Rest/T-pose");
                }
            }

            const int expected = 3 * m_smplh_model.joint_count();
            if (m_smplh_params.pose_axis_angle.size() != expected)
            {
                m_smplh_params.pose_axis_angle = Eigen::VectorXd::Zero(expected);
                m_smplh_deform_dirty = true;
            }

            if ((int)m_smplh_loaded_pose_axis_angle.size() == expected)
            {
                if (!m_smplh_params.pose_axis_angle.isApprox(m_smplh_loaded_pose_axis_angle, 0.0))
                {
                    m_smplh_params.pose_axis_angle = m_smplh_loaded_pose_axis_angle;
                    m_smplh_deform_dirty = true;
                }
            }
            else
            {
                if (!m_smplh_params.pose_axis_angle.isZero())
                {
                    m_smplh_params.pose_axis_angle.setZero();
                    m_smplh_deform_dirty = true;
                }
            }
        }

        bool recomputed_vertices = false;
        const bool align_to_raw_rig =
            m_smplh_loaded_pose_mirror_x && state.scene_view_raw_rig_root_valid;
        const bool scale_to_raw_rig =
            align_to_raw_rig && state.scene_view_raw_rig_height_valid;
        const Eigen::Vector3d raw_rig_root(
            (double)state.scene_view_raw_rig_root[0],
            (double)state.scene_view_raw_rig_root[1],
            (double)state.scene_view_raw_rig_root[2]);
        const double raw_rig_height = (double)state.scene_view_raw_rig_height;
        if (align_to_raw_rig != m_smplh_last_raw_rig_root_valid
            || (align_to_raw_rig && (raw_rig_root - m_smplh_last_raw_rig_root).squaredNorm() > 1e-12))
        {
            m_smplh_deform_dirty = true;
            m_smplh_last_raw_rig_root_valid = align_to_raw_rig;
            m_smplh_last_raw_rig_root = raw_rig_root;
        }
        if (scale_to_raw_rig != m_smplh_last_raw_rig_height_valid
            || (scale_to_raw_rig && std::abs(raw_rig_height - m_smplh_last_raw_rig_height) > 1e-9))
        {
            m_smplh_deform_dirty = true;
            m_smplh_last_raw_rig_height_valid = scale_to_raw_rig;
            m_smplh_last_raw_rig_height = raw_rig_height;
        }

        if (m_smplh_deform_dirty || m_smplh_vertices.rows() == 0)
        {
            MatrixX3dRM V_skinned;
            std::string eval_err;
            if (!smplh_evaluate_lbs(m_smplh_model, m_smplh_params, V_skinned, &eval_err))
            {
                std::fprintf(stderr, "[PoseWeaver] SMPLH LBS eval failed: %s\n", eval_err.c_str());
                data.clear();
                data.set_visible(false);
                m_smplh_mesh_name.clear();
                return;
            }
            if (m_smplh_loaded_pose_mirror_x)
                V_skinned.col(0) *= -1.0;
            if (align_to_raw_rig
                && m_smplh_model.J_regressor.rows() > 0
                && m_smplh_model.J_regressor.cols() == V_skinned.rows())
            {
                const Eigen::MatrixXd joints = m_smplh_model.J_regressor * V_skinned;
                if (joints.rows() > 0)
                {
                    const Eigen::Vector3d current_root = joints.row(0).transpose();
                    double scale = 1.0;
                    if (scale_to_raw_rig)
                    {
                        const double smpl_height = joints.col(1).maxCoeff() - joints.col(1).minCoeff();
                        if (smpl_height > 1e-6 && std::isfinite(smpl_height)
                            && raw_rig_height > 1e-6 && std::isfinite(raw_rig_height))
                            scale = std::clamp(raw_rig_height / smpl_height, 0.05, 20.0);
                    }

                    V_skinned.rowwise() -= current_root.transpose();
                    V_skinned *= scale;
                    V_skinned.rowwise() += raw_rig_root.transpose();
                }
            }
            m_smplh_vertices = V_skinned;
            recomputed_vertices = true;
            m_smplh_deform_dirty = false;
        }

        m_smplh_vertices_recomputed_this_frame = recomputed_vertices;
        if (recomputed_vertices)
            m_pose_version++;

        const Eigen::MatrixXd& V = m_smplh_vertices;
        Eigen::MatrixXi F = m_smplh_model.F;
        if (m_smplh_loaded_pose_mirror_x && F.cols() >= 3)
            F.col(1).swap(F.col(2));

        if (m_smplh_mesh_name != desired || m_smplh_mesh_mirror_x != m_smplh_loaded_pose_mirror_x)
        {
            data.clear();
            data.set_mesh(V, F);
            data.show_faces = 1;
            data.show_lines = 0;
            data.show_overlay = 0;
            data.set_colors(Eigen::RowVector3d(0.85, 0.86, 0.88));
            data.set_visible(want_visible);

            m_viewer.core().align_camera_center(V, F);
            m_smplh_mesh_name = desired;
            m_smplh_mesh_mirror_x = m_smplh_loaded_pose_mirror_x;
        }
        else
        {
            if (recomputed_vertices)
            {
                data.set_vertices(V);
                data.compute_normals();
            }
            data.set_visible(want_visible);
        }

        if (state.smpl_color_by_uv)
        {
            ensure_smpl_uv_vertex_colors();
            if (m_smpl_uv_vertex_colors_valid && m_smpl_uv_vertex_colors.rows() == V.rows())
                data.set_colors(m_smpl_uv_vertex_colors);
        }
        else if (m_last_smpl_color_by_uv)
        {
            data.set_colors(Eigen::RowVector3d(0.85, 0.86, 0.88));
        }
        m_last_smpl_color_by_uv = state.smpl_color_by_uv;
    }
}
