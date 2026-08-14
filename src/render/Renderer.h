#pragma once

#include "app/AppState.h"
#include "simulation/ClothToSmplMapping.h"
#include "smpl/Smplh.h"

#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <cstdint>
#include <future>
#include <memory>
#include <atomic>
#include <mutex>
#include <array>
#include <vector>

struct GLFWwindow;

namespace poseweaver
{
    class Renderer
    {
    public:
        struct SceneCameraInfo
        {
            Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
            Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
            float camera_zoom = 1.0f;
            float camera_base_zoom = 1.0f;
            bool orthographic = false;
            bool valid = false;
        };

        struct UvDebugTextureInfo
        {
            unsigned int texture_id = 0;
            int width = 0;
            int height = 0;
            bool valid = false;
        };

        struct UvSegment
        {
            float u0 = 0.0f;
            float v0 = 0.0f;
            float u1 = 0.0f;
            float v1 = 0.0f;
        };

        struct SmplBodyPreviewInfo
        {
            unsigned int texture_id = 0;
            int width = 0;
            int height = 0;
            bool valid = false;
        };

        struct UvTriangle
        {
            float u0 = 0.0f;
            float v0 = 0.0f;
            float u1 = 0.0f;
            float v1 = 0.0f;
            float u2 = 0.0f;
            float v2 = 0.0f;
        };

        void init(GLFWwindow* window);
        void render(AppState& state, int fb_width, int fb_height);
        void reset_computed_state(AppState& state);

        SceneCameraInfo scene_camera_info() const;

        UvDebugTextureInfo uv_debug_texture_info() const;
        UvDebugTextureInfo sketch2uv_prediction_texture_info() const;
        UvDebugTextureInfo sketch2iso_prediction_texture_info() const;
        bool sketch2uv_prediction_uv_at(int x, int y, float& out_u, float& out_v) const;
        bool sketch2iso_prediction_match_at(
            int x,
            int y,
            float distance_threshold,
            int& out_vertex,
            float& out_distance) const;
        const std::vector<UvSegment>& smpl_uv_wireframe_segments() const;
        SmplBodyPreviewInfo render_smpl_body_preview(AppState& state, int width, int height);
        bool smpl_body_point_for_uv(float u, float v, Eigen::Vector3f& out_point) const;
        bool smpl_body_point_for_vertex(int vertex, Eigen::Vector3f& out_point) const;
        bool smpl_body_preview_project(const Eigen::Vector3f& point, float& out_x01, float& out_y01) const;
        bool fit_cloth_render_to_smpl_visibility(
            AppState& state,
            const std::vector<unsigned char>& cloth_rgba,
            int width,
            int height,
            const std::string& iso_npz_path,
            const std::string& depth_prediction_path,
            const std::string& rig_json_path,
            float prediction_from_cloth_scale,
            float prediction_from_cloth_offset_x,
            float prediction_from_cloth_offset_y,
            std::vector<unsigned char>& out_rgba,
            std::vector<unsigned char>& out_warped_rgba,
            std::vector<unsigned char>& out_pixelated_occluded_rgba,
            std::vector<unsigned char>& out_unwarped_upscaled_rgba,
            std::vector<unsigned char>& out_debug_rgba,
            std::vector<unsigned char>& out_warped_debug_rgba,
            std::vector<unsigned char>& out_unfiltered_occlusion_rgba,
            std::vector<unsigned char>& out_occlusion_rgba,
            std::vector<unsigned char>& out_final_unfiltered_occlusion_rgba,
            std::vector<unsigned char>& out_final_occlusion_rgba,
            std::vector<unsigned char>& out_scene_occlusion_rgba,
            std::vector<unsigned char>& out_aligned_predicted_depth_occlusion_rgba,
            std::vector<unsigned char>& out_iso_occlusion_rgba,
            std::vector<unsigned char>& out_scene_debug_rgba,
            std::vector<unsigned char>& out_deformation_rgba,
            std::vector<unsigned char>& out_close_to_body_rgba,
            std::vector<std::vector<unsigned char>>& out_deformation_steps_rgba,
            int& out_tested_count,
            int& out_removed_count,
            std::string* out_error);

        bool garment_to_smpl_mapping_valid() const { return m_garment_to_smpl.valid(); }
        int garment_constraint_count() const;
        const std::vector<UvSegment>& garment_uv_wireframe_segments() const { return m_garment_uv_segments; }
        const std::vector<UvTriangle>& garment_uv_triangles() const { return m_garment_uv_triangles; }

        bool smpl_uv_layout_loaded() const { return m_smpl_uv_layout_loaded; }
        const std::string& smpl_uv_layout_source() const { return m_smpl_uv_layout_source; }

        void shutdown();

        void on_framebuffer_resize(int fb_width, int fb_height);

        void on_cursor_pos(double x, double y);
        void on_mouse_button(int button, int action, int mods);
        void on_scroll(double xoffset, double yoffset);
        void on_key(int key, int scancode, int action, int mods);
        void on_char(unsigned int codepoint, int mods);

    private:
        enum class MeshShaderVariant
        {
            Default = 0,
            TwoLight = 1,
            ClothToon = 2,
        };

        void ensure_initialized();

        void ensure_grid_built();
        void ensure_object_mesh_loaded(const AppState& state);
        void ensure_garment_mesh_loaded(AppState& state);
        void ensure_smplh_mesh_loaded(AppState& state);
        void ensure_sketch2rig_rig_loaded(AppState& state);
        bool fit_smpl_to_sketch_rig(AppState& state, const Eigen::MatrixXd& vertices);
        void apply_smpl_scene_transform(const AppState& state, Eigen::MatrixXd& vertices) const;

        bool ensure_smplh_model_loaded(AppState& state, SmplType gender);
        void ensure_garment_to_smpl_mapping(AppState& state);
        void ensure_garment_uv_projection();
        void handle_garment_posing(AppState& state, bool force);
        void pose_garment_to_smpl(AppState& state);

        void call_commands_from_state(AppState& state);
        void process_garment_pushout(AppState& state);
        void process_garment_ease(AppState& state);
        void process_garment_prefit(AppState& state);
        void process_garment_relax(AppState& state);
        void process_cloth_constraints_save(AppState& state);
        void process_cloth_constraints_clear(AppState& state);
        void process_cloth_constraint_click(AppState& state);
        void load_garment_constraints(AppState& state, int vertex_count);
        void update_garment_constraint_visualization(AppState& state);
        void generate_uv_debug_image(AppState& state);
        void ensure_sketch2uv_prediction_texture(AppState& state);
        void ensure_sketch2iso_prediction_texture(AppState& state);
        bool ensure_iso_vertex_descriptors(AppState& state);
        bool find_nearest_iso_vertex(
            const float* descriptor,
            int& out_vertex,
            float& out_distance_squared) const;
        void search_iso_kd_node(
            int node_index,
            const float* descriptor,
            int& best_vertex,
            float& best_distance_squared) const;
        void release_smpl_body_preview();

        void process_cloth_simulate(AppState& state);
        void process_cloth_clear_simulation(AppState& state);
        void process_cloth_cancel_simulation(AppState& state);
        void update_cloth_simulation_job(AppState& state);
        void apply_cloth_simulation_frame(AppState& state);
        void capture_scene_view_snapshot_if_requested(AppState& state, const ViewportRect& view);
        void capture_generated_cloth_render_if_requested(AppState& state);
        void process_canva_displacement_map(AppState& state);
        void process_canva_iso_displacement_map(AppState& state);
        void update_collision_proxy_gizmo(AppState& state);

        void regenerate_uv_wireframe_segments();
        bool try_load_smpl_uv_layout_once();
		void ensure_smpl_uv_vertex_colors();

        GLFWwindow* m_window = nullptr;
        bool m_initialized = false;

        igl::opengl::glfw::Viewer m_viewer;
        int m_grid_mesh_index = 0;
        int m_object_mesh_index = 1;
        int m_garment_mesh_index = 2;
        int m_smplh_mesh_index = 3;
        int m_wind_arrow_mesh_index = 4;
        int m_sketch2rig_rig_mesh_index = 5;
        int m_cloth_render_mesh_index = 6;
        int m_collision_proxy_mesh_index = 7;

        // Indexed by mesh index in m_viewer.data_list.
        std::array<MeshShaderVariant, 9> m_mesh_shader_variant{};

        std::string m_object_mesh_name;
        std::string m_garment_mesh_name;
        std::string m_smplh_mesh_name;
        std::string m_sketch2rig_rig_source_path;
        std::string m_sketch2rig_rig_last_error;
        int m_last_fb_w = 0;
        int m_last_fb_h = 0;
        Eigen::MatrixXd m_garment_rest_vertices;
        bool m_smplh_vertices_recomputed_this_frame = false;

        bool m_smplh_loaded = false;
        SmplType m_smplh_loaded_gender = SmplType::Male;
        SmplhModel m_smplh_model;
        SmplhParams m_smplh_params;
        Eigen::MatrixXd m_smplh_vertices;

        Eigen::MatrixXd m_sketch2rig_rig_source_joints;
        Eigen::MatrixXd m_sketch2rig_rig_scene_joints;
        std::vector<int> m_sketch2rig_rig_parents;
        int m_sketch2rig_rig_image_size = 512;
        bool m_sketch2rig_rig_source_valid = false;

        bool m_smplh_deform_dirty = true;
        bool m_smplh_last_show_character = true;
        bool m_garment_last_show_mesh = false;

        bool m_garment_vertex_merge_settings_valid = false;
        bool m_garment_vertex_merge_enabled_last = true;
        double m_garment_vertex_merge_distance_last = 0.0025;

        std::uint64_t m_last_garment_pushout_request_id = 0;

        std::uint64_t m_last_garment_ease_request_id = 0;

        std::uint64_t m_last_garment_prefit_request_id = 0;

        std::uint64_t m_last_garment_relax_request_id = 0;
        std::uint64_t m_last_cloth_constraints_save_request_id = 0;
        std::uint64_t m_last_cloth_constraints_clear_request_id = 0;
        std::uint64_t m_last_cloth_constraint_click_request_id = 0;
        std::vector<unsigned char> m_garment_constraints;
        std::string m_garment_constraints_mesh_name;

        unsigned int m_uv_debug_texture_id = 0;
        int m_uv_debug_texture_w = 0;
        int m_uv_debug_texture_h = 0;
        std::uint64_t m_last_uv_image_request_id = 0;

        unsigned int m_sketch2uv_prediction_texture_id = 0;
        int m_sketch2uv_prediction_texture_w = 0;
        int m_sketch2uv_prediction_texture_h = 0;
        std::string m_sketch2uv_prediction_json_path;
        std::uint64_t m_sketch2uv_prediction_request_id = 0;
        std::vector<unsigned char> m_sketch2uv_prediction_pixels;
        std::vector<float> m_sketch2uv_prediction_uv;
        std::vector<float> m_sketch2uv_prediction_mask;

        unsigned int m_sketch2iso_prediction_texture_id = 0;
        int m_sketch2iso_prediction_texture_w = 0;
        int m_sketch2iso_prediction_texture_h = 0;
        int m_sketch2iso_prediction_channels = 0;
        std::string m_sketch2iso_prediction_npz_path;
        std::string m_sketch2iso_prediction_sketch_path;
        std::uint64_t m_sketch2iso_prediction_request_id = 0;
        std::vector<float> m_sketch2iso_prediction_values;
        std::vector<float> m_sketch2iso_prediction_mask;
        std::vector<float> m_iso_vertex_descriptors;
        int m_iso_vertex_descriptor_count = 0;
        int m_iso_vertex_descriptor_dimensions = 0;
        bool m_iso_vertex_descriptors_load_attempted = false;
        struct IsoKdNode
        {
            int vertex = -1;
            int left = -1;
            int right = -1;
            int axis = 0;
        };
        std::vector<IsoKdNode> m_iso_kd_nodes;
        int m_iso_kd_root = -1;

        unsigned int m_smpl_body_preview_fbo = 0;
        unsigned int m_smpl_body_preview_texture_id = 0;
        unsigned int m_smpl_body_preview_depth = 0;
        int m_smpl_body_preview_w = 0;
        int m_smpl_body_preview_h = 0;
        Eigen::Matrix4f m_smpl_body_preview_view = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f m_smpl_body_preview_proj = Eigen::Matrix4f::Identity();
        bool m_smpl_body_preview_camera_valid = false;

        std::uint64_t m_last_cloth_simulate_request_id = 0;
        std::uint64_t m_last_cloth_clear_simulation_request_id = 0;
        std::uint64_t m_last_cloth_cancel_simulation_request_id = 0;
        std::uint64_t m_last_scene_view_snap_to_target_request_id = 0;
        std::uint64_t m_last_scene_view_snapshot_request_id = 0;
        std::uint64_t m_last_cloth_render_request_id = 0;
        std::uint64_t m_last_canva_displacement_request_id = 0;
        std::uint64_t m_last_canva_iso_displacement_request_id = 0;

        std::vector<Eigen::MatrixXd> m_cloth_sim_frames;
        std::vector<Eigen::MatrixXd> m_smpl_sim_frames;
        std::vector<Eigen::MatrixXd> m_cloth_sim_append_prefix_frames;
        std::vector<Eigen::MatrixXd> m_smpl_sim_append_prefix_frames;
        std::vector<Eigen::MatrixXd> m_collision_proxy_append_prefix_frames;
        bool m_cloth_sim_append_result = false;
        std::uint64_t m_cloth_sim_pose_version = 0;
        std::uint64_t m_cloth_sim_garment_version = 0;
        bool m_cloth_sim_require_pose_version_match = true;
        bool m_cloth_sim_play_on_finish = true;

        struct ClothSimJobResult
        {
            bool ok = false;
            std::vector<Eigen::MatrixXd> frames; // each is (#V x 3)
            std::vector<Eigen::MatrixXd> smpl_frames;
            std::vector<Eigen::MatrixXd> collision_proxy_frames;
            Eigen::MatrixXi collision_proxy_F;
            int steps_simulated = 0;
            int last_step_contacts = 0;
            std::string timing_report;
            double sequence_seconds = 0.0;
            double proxy_seconds = 0.0;
            double solver_seconds = 0.0;
            double pipeline_seconds = 0.0;
            std::vector<std::string> sequence_timing_details;
            std::vector<std::string> proxy_timing_details;
            std::vector<std::string> solver_timing_details;
            std::string error;
        };

        std::future<ClothSimJobResult> m_cloth_sim_future;
        bool m_cloth_sim_job_running = false;
        std::shared_ptr<std::atomic_bool> m_cloth_sim_cancel_flag;
        std::shared_ptr<std::atomic_int> m_cloth_sim_progress_steps_done;
        std::shared_ptr<std::atomic_int> m_cloth_sim_progress_stage;
        int m_cloth_sim_progress_steps_total = 0;
        std::string m_cloth_sim_last_error;

        bool m_cloth_sim_restart_pending = false;
        bool m_cloth_sim_restart_play_on_finish = true;

        Eigen::MatrixXd m_cloth_sim_reset_frame0;
        bool m_cloth_sim_reset_frame0_valid = false;
        std::uint64_t m_cloth_sim_reset_pose_version = 0;
        std::uint64_t m_cloth_sim_reset_garment_version = 0;

        struct ClothSimPreviewState
        {
            std::mutex mutex;
            Eigen::MatrixXd V; // (#V x 3)
            bool valid = false;

            Eigen::MatrixXd body_V; // (#Vbody x 3)
            Eigen::MatrixXi body_F; // (#Fbody x 3), needed when XPBD uses simplified SMPL topology
            bool body_valid = false;
            bool body_topology_valid = false;
            std::atomic_uint64_t seq{ 0 };
        };

        std::shared_ptr<ClothSimPreviewState> m_cloth_sim_preview_state;
        std::shared_ptr<std::atomic_bool> m_cloth_sim_preview_enabled;
        std::uint64_t m_cloth_sim_preview_last_seq_applied = 0;
        std::vector<Eigen::MatrixXd> m_collision_proxy_sim_frames;
        Eigen::MatrixXi m_collision_proxy_sim_F;

        std::uint64_t m_pose_version = 0;
        std::uint64_t m_garment_version = 0;
        std::uint64_t m_last_posed_pose_version = 0;
        std::uint64_t m_last_posed_garment_version = 0;

        std::vector<UvSegment> m_smpl_uv_segments;

        std::vector<UvSegment> m_garment_uv_segments;
        std::vector<UvTriangle> m_garment_uv_triangles;
        bool m_garment_uv_overlay_valid = false;

        Eigen::MatrixXd m_smpl_uv_vertex_colors; // (#V x 3)
        Eigen::MatrixXd m_smpl_uv_per_vertex;    // (#V x 2), averaged if UV layout has seams
        bool m_smpl_uv_vertex_colors_valid = false;
        bool m_last_smpl_color_by_uv = false;

        Eigen::MatrixXd m_garment_uv_vertex_colors; // (#Vcloth x 3)
        bool m_garment_uv_vertex_colors_valid = false;
        bool m_last_garment_color_by_smpl_uv = false;

        ClothToSmplMapping m_garment_to_smpl;
        std::string m_garment_mapping_mesh_name;
        SmplType m_garment_mapping_gender = SmplType::Male;

        // Avoids retrying a failing cloth→SMPL mapping every frame.
        std::string m_garment_last_mapping_attempt_mesh_name;
        SmplType m_garment_last_mapping_attempt_gender = SmplType::Male;
        bool m_garment_last_mapping_attempt_failed = false;
        std::string m_garment_last_mapping_attempt_error;

        bool m_smpl_uv_layout_loaded = false;
        std::string m_smpl_uv_layout_source;
        Eigen::MatrixXd m_smpl_uv_coords;   // (#UV x 2)
        Eigen::MatrixXi m_smpl_F_uv;        // (#F x 3)

        std::string m_smplh_loaded_pose_name;
        Eigen::VectorXd m_smplh_loaded_pose_axis_angle;
        MatrixX3dRM m_smplh_loaded_direct_vertices;
        std::uint64_t m_smplh_last_pose_reload_request_id = 0;
        bool m_smplh_loaded_pose_mirror_x = false;
        bool m_smplh_mesh_mirror_x = false;
        bool m_smplh_sim_preview_mesh_active = false;
    };
}
