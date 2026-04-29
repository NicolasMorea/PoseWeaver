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

        SceneCameraInfo scene_camera_info() const;

        UvDebugTextureInfo uv_debug_texture_info() const;
        const std::vector<UvSegment>& smpl_uv_wireframe_segments() const;

        bool garment_to_smpl_mapping_valid() const { return m_garment_to_smpl.valid(); }
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
        void ensure_initialized();

        void ensure_grid_built();
        void ensure_object_mesh_loaded(const AppState& state);
        void ensure_garment_mesh_loaded(AppState& state);
        void ensure_smplh_mesh_loaded(AppState& state);

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
        void generate_uv_debug_image(AppState& state);

        void process_cloth_simulate(AppState& state);
        void process_cloth_clear_simulation(AppState& state);
        void process_cloth_cancel_simulation(AppState& state);
        void update_cloth_simulation_job(AppState& state);
        void apply_cloth_simulation_frame(AppState& state);

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

        // Indexed by mesh index in m_viewer.data_list.
        std::array<bool, 8> m_two_light_shader_installed{};

        std::string m_object_mesh_name;
        std::string m_garment_mesh_name;
        std::string m_smplh_mesh_name;
        int m_last_fb_w = 0;
        int m_last_fb_h = 0;
        Eigen::MatrixXd m_garment_rest_vertices;
        bool m_smplh_vertices_recomputed_this_frame = false;

        bool m_smplh_loaded = false;
        SmplType m_smplh_loaded_gender = SmplType::Male;
        SmplhModel m_smplh_model;
        SmplhParams m_smplh_params;
        Eigen::MatrixXd m_smplh_vertices;

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

        unsigned int m_uv_debug_texture_id = 0;
        int m_uv_debug_texture_w = 0;
        int m_uv_debug_texture_h = 0;
        std::uint64_t m_last_uv_image_request_id = 0;

        std::uint64_t m_last_cloth_simulate_request_id = 0;
        std::uint64_t m_last_cloth_clear_simulation_request_id = 0;
        std::uint64_t m_last_cloth_cancel_simulation_request_id = 0;

        std::vector<Eigen::MatrixXd> m_cloth_sim_frames;
        std::uint64_t m_cloth_sim_pose_version = 0;
        std::uint64_t m_cloth_sim_garment_version = 0;

        struct ClothSimJobResult
        {
            bool ok = false;
            std::vector<Eigen::MatrixXd> frames; // each is (#V x 3)
            int steps_simulated = 0;
            int last_step_contacts = 0;
            std::string timing_report;
            std::string error;
        };

        std::future<ClothSimJobResult> m_cloth_sim_future;
        bool m_cloth_sim_job_running = false;
        std::shared_ptr<std::atomic_bool> m_cloth_sim_cancel_flag;
        std::shared_ptr<std::atomic_int> m_cloth_sim_progress_steps_done;
        int m_cloth_sim_progress_steps_total = 0;
        std::string m_cloth_sim_last_error;

        bool m_cloth_sim_restart_pending = false;

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
            bool body_valid = false;
            std::atomic_uint64_t seq{ 0 };
        };

        std::shared_ptr<ClothSimPreviewState> m_cloth_sim_preview_state;
        std::shared_ptr<std::atomic_bool> m_cloth_sim_preview_enabled;
        std::uint64_t m_cloth_sim_preview_last_seq_applied = 0;

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
    };
}
