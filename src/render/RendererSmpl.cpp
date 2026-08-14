#include "render/Renderer.h"
#include "core/Paths.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

namespace poseweaver
{
    namespace
    {
        const char* smplh_gender_dir(SmplType gender)
        {
            if (gender == SmplType::Female)
                return "female";
            if (gender == SmplType::Neutral)
                return "neutral";
            return "male";
        }

        const char* smplh_gender_label(SmplType gender)
        {
            if (gender == SmplType::Female)
                return "Female";
            if (gender == SmplType::Neutral)
                return "Neutral";
            return "Male";
        }
    }

    bool Renderer::fit_smpl_to_sketch_rig(AppState& state, const Eigen::MatrixXd& vertices)
    {
        state.scene_view_smpl_fit_valid = false;
        state.scene_view_smpl_fit_scale = { 1.0f, 1.0f, 1.0f };
        state.scene_view_smpl_fit_translation = { 0.0f, 0.0f, 0.0f };
        state.scene_view_smpl_fit_rms = 0.0f;

        if (!m_smplh_loaded_pose_mirror_x
            || vertices.cols() != 3
            || m_sketch2rig_rig_scene_joints.cols() != 3
            || m_smplh_model.J_regressor.cols() != vertices.rows())
            return false;

        const Eigen::MatrixXd smpl_joints = m_smplh_model.J_regressor * vertices;
        const int joint_count = std::min<int>(
            22,
            std::min<int>(smpl_joints.rows(), m_sketch2rig_rig_scene_joints.rows()));
        if (joint_count < 2)
            return false;

        Eigen::VectorXd weights = Eigen::VectorXd::Zero(joint_count);
        for (int child = 0; child < joint_count && child < (int)m_sketch2rig_rig_parents.size(); ++child)
        {
            const int parent = m_sketch2rig_rig_parents[(size_t)child];
            if (parent >= 0 && parent < joint_count)
            {
                weights[child] += 1.0;
                weights[parent] += 1.0;
            }
        }

        const double weight_sum = weights.sum();
        if (!(weight_sum > 0.0) || !std::isfinite(weight_sum))
            return false;

        Eigen::Vector3d source_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d target_mean = Eigen::Vector3d::Zero();
        for (int i = 0; i < joint_count; ++i)
        {
            const Eigen::Vector3d source = smpl_joints.row(i).transpose();
            const Eigen::Vector3d target = m_sketch2rig_rig_scene_joints.row(i).transpose();
            if (!source.allFinite() || !target.allFinite())
                return false;
            source_mean.noalias() += weights[i] * source;
            target_mean.noalias() += weights[i] * target;
        }
        source_mean /= weight_sum;
        target_mean /= weight_sum;

        double xy_numerator = 0.0;
        double xy_denominator = 0.0;
        double z_numerator = 0.0;
        double z_denominator = 0.0;
        for (int i = 0; i < joint_count; ++i)
        {
            const Eigen::Vector3d source = smpl_joints.row(i).transpose() - source_mean;
            const Eigen::Vector3d target = m_sketch2rig_rig_scene_joints.row(i).transpose() - target_mean;
            const double weight = weights[i];
            xy_numerator += weight * (source.x() * target.x() + source.y() * target.y());
            xy_denominator += weight * (source.x() * source.x() + source.y() * source.y());
            z_numerator += weight * source.z() * target.z();
            z_denominator += weight * source.z() * source.z();
        }

        if (!(xy_denominator > 1e-12) || !std::isfinite(xy_denominator))
            return false;

        const double scale_xy = std::clamp(xy_numerator / xy_denominator, 0.05, 20.0);
        const double scale_z = (z_denominator > 1e-12 && std::isfinite(z_denominator))
            ? std::clamp(z_numerator / z_denominator, 0.05, 20.0)
            : scale_xy;
        if (!std::isfinite(scale_xy) || !std::isfinite(scale_z))
            return false;

        const Eigen::Vector3d scale(scale_xy, scale_xy, scale_z);
        const Eigen::Vector3d translation = target_mean - scale.cwiseProduct(source_mean);
        if (!translation.allFinite())
            return false;

        double squared_error = 0.0;
        for (int i = 0; i < joint_count; ++i)
        {
            const Eigen::Vector3d fitted =
                scale.cwiseProduct(smpl_joints.row(i).transpose()) + translation;
            const Eigen::Vector3d delta =
                fitted - m_sketch2rig_rig_scene_joints.row(i).transpose();
            squared_error += weights[i] * delta.squaredNorm();
        }
        const double rms = std::sqrt(squared_error / weight_sum);

        state.scene_view_smpl_fit_scale = {
            (float)scale_xy,
            (float)scale_xy,
            (float)scale_z
        };
        state.scene_view_smpl_fit_translation = {
            (float)translation.x(),
            (float)translation.y(),
            (float)translation.z()
        };
        state.scene_view_smpl_fit_rms = std::isfinite(rms) ? (float)rms : 0.0f;
        state.scene_view_smpl_fit_valid = true;
        return true;
    }

    void Renderer::apply_smpl_scene_transform(const AppState& state, Eigen::MatrixXd& vertices) const
    {
        if (!state.scene_view_smpl_fit_valid || vertices.cols() != 3)
            return;

        const Eigen::RowVector3d scale(
            (double)state.scene_view_smpl_fit_scale[0],
            (double)state.scene_view_smpl_fit_scale[1],
            (double)state.scene_view_smpl_fit_scale[2]);
        const Eigen::RowVector3d translation(
            (double)state.scene_view_smpl_fit_translation[0],
            (double)state.scene_view_smpl_fit_translation[1],
            (double)state.scene_view_smpl_fit_translation[2]);
        if (!scale.allFinite() || !translation.allFinite())
            return;

        vertices.array().rowwise() *= scale.array();
        vertices.rowwise() += translation;
    }

    bool Renderer::ensure_smplh_model_loaded(AppState& state, SmplType gender)
    {
        if (m_smplh_loaded && m_smplh_loaded_gender == gender)
            return true;

        const std::filesystem::path model_path = assets_dir() / "smplh" / smplh_gender_dir(gender) / "model.npz";
        std::string err;
        if (!m_smplh_model.load_from_npz(model_path, &err))
        {
            std::fprintf(stderr, "[PoseWeaver] Failed to load SMPLH model: %s\n", err.c_str());
            console_add_line(state, std::string("ERROR: Failed to load SMPLH model: ") + err);
            m_smplh_loaded = false;
            return false;
        }

        console_add_line(state, std::string("Loaded SMPLH model: ") + smplh_gender_dir(gender));
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
        m_smplh_loaded_direct_vertices.resize(0, 0);
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
        const std::string desired = std::string("SMPLH:") + smplh_gender_label(gender);
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
                m_smplh_loaded_direct_vertices.resize(0, 0);
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
                        MatrixX3dRM direct_vertices;
                        std::string direct_vertices_err;
                        if (smplh_load_pose_npz_vertices(pose_path, direct_vertices, &direct_vertices_err))
                        {
                            if (direct_vertices.rows() == m_smplh_model.vertex_count()
                                && direct_vertices.cols() == 3)
                            {
                                m_smplh_loaded_direct_vertices = std::move(direct_vertices);
                                console_add_line(state, "Pose direct vertices loaded: " + pose_name);
                            }
                            else
                            {
                                console_add_line(state, "Pose direct vertices ignored: vertex count does not match model");
                            }
                        }
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
        if (m_smplh_deform_dirty || m_smplh_vertices.rows() == 0)
        {
            Eigen::MatrixXd display_vertices;
            if (m_smplh_loaded_direct_vertices.rows() == m_smplh_model.vertex_count()
                && m_smplh_loaded_direct_vertices.cols() == 3)
            {
                display_vertices = m_smplh_loaded_direct_vertices;
            }
            else
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
                display_vertices = V_skinned;
            }
            if (m_smplh_loaded_pose_mirror_x)
                display_vertices.col(0) *= -1.0;
            (void)fit_smpl_to_sketch_rig(state, display_vertices);
            apply_smpl_scene_transform(state, display_vertices);
            m_smplh_vertices = std::move(display_vertices);
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
