#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace poseweaver
{
    class Sketch2RigService;

    enum class SimulationSolver
    {
        ProjectiveDynamics = 0,
        Xpbd = 1,
    };

    struct ViewportRect
    {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
        bool valid = false;
    };

    enum class SmplType
    {
        Male = 0,
        Female = 1,
        Neutral = 2,
    };

    struct SimulationControls
    {
        SimulationSolver solver = SimulationSolver::Xpbd;

        bool simulating = false;
        bool playing = false;
        bool preview = true;
        std::string sim_stage;
        int current_frame = 0;
        int max_frames = 300; // cap for recorded frames
        int frame_count = 0;  // actual recorded frames (0 when none)
        float fps = 30.0f;

        int sim_steps_done = 0;
        int sim_steps_total = 0;

        bool smooth_sequence = true;
        float smooth_sequence_length = 0.5f;
    };

    struct SimulationParams
    {
        float dt = 1.0f / 60.0f;
        float duration_seconds = 2.0f;
        int solver_iterations = 10;
        float gravity = -9.81f;

        bool stretch_enabled = true;
        float stretch_stiffness = 1.0f;
        float base_stretch_mult = 1.0f;

        bool bend_enabled = true;
        float bend_stiffness = 0.5f; // dihedral-angle bending (PD)

        bool iso_bend_enabled = false;
        float iso_bend_stiffness = 0.5f; // isometric bending (PD)

        bool collision_enabled = true;
        float collision_stiffness = 1.0f;
        float collision_thickness = 0.003f;

        bool enable_friction = false;
        float friction_mu = 0.30f;
        float damping = 0.01f;

        float wind_force = 0.0f;
        float wind_yaw_deg = 0.0f;
        float wind_pitch_deg = 0.0f;

        float wind_random_space = 0.0f;
        float wind_random_time = 0.0f;
    };

    // Compliance values are >= 0, where 0 is perfectly rigid.
    struct SimulationXpbdParams
    {
        float dt = 1.0f / 60.0f;
        float duration_seconds = 2.0f;
        int solver_iterations = 20;
        int substeps = 2;
        float gravity = -9.81f;

        bool stretch_enabled = true;
        float stretch_compliance = 0.0f;
        float base_stretch_mult = 1.0f;

        bool bend_enabled = true;
        float bend_compliance = 1e-4f;

        bool collision_enabled = true;
        float collision_compliance = 0.0f;
        float collision_thickness = 0.005f;

        bool self_collision_enabled = false;
        float self_collision_compliance = 0.0f;
        float self_collision_thickness = 0.005f;

        bool enable_friction = false;
        float friction_mu = 0.30f;
        float damping = 0.005f;

        float wind_force = 0.0f;
        float wind_yaw_deg = 0.0f;
        float wind_pitch_deg = 0.0f;

        float wind_random_space = 0.0f;
        float wind_random_time = 0.0f;

        float smpl_geometry_percent = 100.0f;
    };

    struct SelectionState
    {
        SmplType gender = SmplType::Male;

        std::vector<std::string> poses;
        int pose_index = -1;

        std::vector<std::string> garments;
        int garment_index = -1;

        std::vector<std::string> meshes;
        int mesh_index = -1;
    };

    struct DebugConsoleState
    {
        bool auto_scroll = true;
        std::vector<std::string> lines;
    };

    struct UiLayoutState
    {
        float left_panel_width = 360.0f;
        float console_height = 240.0f;
    };

    inline void console_add_line(DebugConsoleState& console, std::string line)
    {
        constexpr size_t kMaxLines = 2000;
        console.lines.emplace_back(std::move(line));
        if (console.lines.size() > kMaxLines)
        {
            const size_t extra = console.lines.size() - kMaxLines;
            console.lines.erase(console.lines.begin(), console.lines.begin() + (std::ptrdiff_t)extra);
        }
    }

    struct SmplShapeState
    {
        std::array<float, 16> betas{};
    };

    enum class Sketch2RigStatus
    {
        NotStarted = 0,
        Loading,
        Ready,
        Busy,
        Error,
        Stopped,
    };

    struct Sketch2RigUiState
    {
        Sketch2RigService* service = nullptr;
        Sketch2RigStatus status = Sketch2RigStatus::NotStarted;
        std::string message = "Model not started";
        std::string last_sketch_path;
        std::string last_output_path;
        std::uint64_t next_request_id = 1;
        std::uint64_t active_request_id = 0;
        std::uint64_t completed_request_id = 0;
    };

    struct AppState
    {
        SimulationControls controls;
        SimulationParams params;
        SimulationXpbdParams xpbd_params;
        SelectionState selection;
        ViewportRect cloth_view;

        std::array<float, 3> light_position{ 0.8f, 1.2f, 1.2f };
        bool light_position_initialized = false;
        std::array<float, 3> wind_arrow_pivot{ 0.0f, 0.6f, 0.6f };
        bool wind_arrow_pivot_initialized = false;

        bool scene_view_gizmo_over = false;
        bool scene_view_gizmo_using = false;

        // -1 = none, 0 = wind rotation, 1 = light translation
        int scene_view_active_gizmo = -1;

        std::array<float, 2> scene_view_light_marker_screen{ 0.0f, 0.0f };
        bool scene_view_light_marker_valid = false;
        std::array<float, 2> scene_view_wind_marker_screen{ 0.0f, 0.0f };
        bool scene_view_wind_marker_valid = false;

        std::uint64_t scene_view_snap_to_target_request_id = 0;
        bool scene_view_target_valid = false;
        std::array<float, 3> scene_view_target_eye{ 0.0f, 0.0f, 0.0f };
        std::array<float, 3> scene_view_target_direction{ 0.0f, 0.0f, -1.0f };
        std::array<float, 3> scene_view_target_up{ 0.0f, 1.0f, 0.0f };
        bool scene_view_raw_rig_root_valid = false;
        std::array<float, 3> scene_view_raw_rig_root{ 0.0f, 0.0f, 0.0f };
        bool scene_view_raw_rig_height_valid = false;
        float scene_view_raw_rig_height = 0.0f;
        std::uint64_t scene_view_snapshot_request_id = 0;
        std::uint64_t scene_view_snapshot_completed_id = 0;
        std::string scene_view_snapshot_pose_name;
        std::string scene_view_snapshot_path;

        std::uint64_t garment_pushout_request_id = 0;
        std::uint64_t garment_ease_request_id = 0;
        std::uint64_t garment_prefit_request_id = 0;
        std::uint64_t garment_relax_request_id = 0;
        std::uint64_t uv_image_request_id = 0;

        std::uint64_t cloth_simulate_request_id = 0;
        std::uint64_t cloth_clear_simulation_request_id = 0;
        std::uint64_t cloth_cancel_simulation_request_id = 0;
        std::uint64_t cloth_render_request_id = 0;
        std::uint64_t cloth_render_completed_id = 0;
        std::string cloth_render_path;

        float garment_min_cloth_skin_distance = 0.001f;
        float garment_ease_amount = 0.20f;
        int garment_prefit_steps = 10;

        bool garment_vertex_merge_enabled = true;
        float garment_vertex_merge_distance = 0.0025f;

        float uv_view_zoom = 1.0f;
        float uv_view_pan_x = 0.0f;
        float uv_view_pan_y = 0.0f;

        float canva_brush_radius = 8.0f;
        std::array<float, 4> canva_brush_color{ 0.0f, 0.0f, 0.0f, 1.0f };

        bool canva_tab_active = false;
        std::uint64_t canva_undo_request_id = 0;
        bool cursor_restore_request = false;
        bool soft_cursor_enabled = false;

        std::uint64_t asset_refresh_request_id = 0;
        std::uint64_t pose_asset_reload_request_id = 0;
        std::string pending_pose_selection;

        Sketch2RigUiState sketch2rig;
        Sketch2RigUiState sketch2betas;

        SmplShapeState smpl_shape;
        UiLayoutState layout;

        DebugConsoleState console;

        bool show_cloth_view = true;
        bool show_simulation = true;
        bool show_smpl = true;
        bool show_smpl_character = true;
        bool show_sketch2rig_rig = true;
        bool smpl_color_by_uv = false;
        bool show_garments = true;
        bool show_object_mesh = false;
        bool show_garment_mesh = true;
        bool garment_color_by_smpl_uv = false;
        bool show_console = true;
    };

    inline void console_add_line(AppState& state, std::string line)
    {
        console_add_line(state.console, std::move(line));
    }
}
