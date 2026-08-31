#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "simulation/SimulationTypes.h"

namespace poseweaver
{
    enum class ClothInkStyle : int
    {
        Off = 0,
        Detailed = 1,
        Clean = 2,
        SoftGeometry = 3,
    };

    class ModelInferenceService;

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

    enum class CollisionGeometryType
    {
        RawSmpl = 0,
        SimplifiedSmpl = 1,
        SimpleProxy = 2,
    };

    enum class SimulationDevice
    {
        Cpu = 0,
        Gpu = 1,
    };

    struct SimulationControls
    {
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
        int sim_pipeline_stage = 0; // 0 sequence, 1 proxy, 2 simulation
        int sim_sequence_steps_done = 0;
        int sim_sequence_steps_total = 0;
        int sim_proxy_steps_done = 0;
        int sim_proxy_steps_total = 0;
        int sim_solver_steps_done = 0;
        int sim_solver_steps_total = 0;
        double sim_sequence_seconds = 0.0;
        double sim_proxy_seconds = 0.0;
        double sim_solver_seconds = 0.0;
        double sim_pipeline_seconds = 0.0;
        std::vector<std::string> sim_sequence_timing_details;
        std::vector<std::string> sim_proxy_timing_details;
        std::vector<std::string> sim_solver_timing_details;

        bool smooth_sequence = true;
        float smooth_sequence_length = 0.5f;
        bool save_smpl_sequence = false;
        bool show_collision_proxy_sequence = false;
    };

    // Compliance values are >= 0, where 0 is perfectly rigid.
    struct SimulationXpbdParams
    {
        SimulationDevice device = SimulationDevice::Cpu;
        float dt = 1.0f / 60.0f;
        float duration_seconds = 2.0f;
        int simulation_steps = 120;
        int solver_iterations = 20;
        int substeps = 2;
        float gravity = -9.81f;

        bool stretch_enabled = true;
        float stretch_compliance = 0.0f;
        float base_stretch_mult = 1.0f;

        bool bend_enabled = true;
        BendingMethod bending_method = BendingMethod::OppositeVertexDistance;
        float bend_compliance = 1e-4f;
        float dihedral_compliance = 1e-6f;

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

        float wind_random_magnitude = 0.0f;
        float wind_random_time_scale = 1.0f;
        float wind_random_spatial_scale = 1.0f;

        CollisionGeometryType collision_geometry = CollisionGeometryType::RawSmpl;
        bool simplify_smpl = false; // Legacy setting, mirrored from collision_geometry for old imgui.ini files.
        float smpl_geometry_percent = 100.0f;
    };

    inline float simulation_dt_seconds(const SimulationXpbdParams& params)
    {
        const int steps = std::max(1, params.simulation_steps);
        const float duration = std::max(0.0f, params.duration_seconds);
        return duration / (float)steps;
    }

    inline float solver_dt_seconds(const SimulationXpbdParams& params)
    {
        return std::max(1e-6f, simulation_dt_seconds(params));
    }

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

    enum class ModelInferenceStatus
    {
        NotStarted = 0,
        Loading,
        Ready,
        Busy,
        Error,
        Stopped,
    };

    struct ModelInferenceUiState
    {
        ModelInferenceService* service = nullptr;
        ModelInferenceStatus status = ModelInferenceStatus::NotStarted;
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
        SimulationXpbdParams xpbd_params;
        bool gpu_simulation_available = false;
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
        float scene_view_target_fov_degrees = 0.0f;
        bool scene_view_sequence_rig_overlay_active = false;
        bool scene_view_sequence_rig_overlay_visible = false;
        std::string scene_view_sequence_rig_overlay_path;
        bool scene_view_raw_rig_root_valid = false;
        std::array<float, 3> scene_view_raw_rig_root{ 0.0f, 0.0f, 0.0f };
        bool scene_view_raw_rig_height_valid = false;
        float scene_view_raw_rig_height = 0.0f;
        bool scene_view_smpl_fit_valid = false;
        std::array<float, 3> scene_view_smpl_fit_scale{ 1.0f, 1.0f, 1.0f };
        std::array<float, 3> scene_view_smpl_fit_translation{ 0.0f, 0.0f, 0.0f };
        float scene_view_smpl_fit_rms = 0.0f;
        std::uint64_t scene_view_snapshot_request_id = 0;
        std::uint64_t scene_view_snapshot_completed_id = 0;
        std::string scene_view_snapshot_pose_name;
        std::string scene_view_snapshot_path;
        bool scene_view_snapshot_use_rigopt_camera = false;
        float scene_view_snapshot_rigopt_camera_scale = 1.0f;
        std::array<float, 2> scene_view_snapshot_rigopt_camera_translation{ 0.0f, 0.0f };
        int scene_view_snapshot_rigopt_camera_source_size = 512;

        std::uint64_t garment_pushout_request_id = 0;
        std::uint64_t garment_ease_request_id = 0;
        std::uint64_t garment_prefit_request_id = 0;
        std::uint64_t garment_relax_request_id = 0;
        std::uint64_t uv_image_request_id = 0;
        std::uint64_t cloth_constraints_save_request_id = 0;
        std::uint64_t cloth_constraints_clear_request_id = 0;
        std::uint64_t cloth_constraint_click_request_id = 0;
        int cloth_constraint_selection_x0 = 0;
        int cloth_constraint_selection_y0 = 0;
        int cloth_constraint_selection_x1 = 0;
        int cloth_constraint_selection_y1 = 0;
        int cloth_constraint_click_mods = 0;
        bool cloth_constraint_drag_active = false;
        std::array<float, 2> cloth_constraint_drag_start_screen{ 0.0f, 0.0f };
        std::array<float, 2> cloth_constraint_drag_end_screen{ 0.0f, 0.0f };

        std::uint64_t cloth_simulate_request_id = 0;
        std::uint64_t cloth_clear_simulation_request_id = 0;
        std::uint64_t cloth_cancel_simulation_request_id = 0;
        bool cloth_simulate_play_on_finish = true;
        bool cloth_simulate_continue_from_last_frame = false;
        float cloth_simulate_duration_override_seconds = -1.0f;
        int cloth_simulate_max_frames_override = 0;
        bool cloth_simulate_pose_transition_enabled = false;
        std::string cloth_simulate_transition_from_pose_path;
        int cloth_simulate_pose_transition_frames = 0;
        bool cloth_simulate_sequence_enabled = false;
        std::vector<std::string> cloth_simulate_sequence_pose_paths;
        std::vector<std::string> cloth_simulate_sequence_rig_paths;
        int cloth_simulate_sequence_frame_rate = 12;
        int cloth_simulate_sequence_transition_frames = 0;
        bool cloth_simulate_sequence_initial_smooth = false;
        std::uint64_t cloth_render_request_id = 0;
        std::uint64_t cloth_render_completed_id = 0;
        std::string cloth_render_path;
        std::string cloth_render_stroke_reference_path;
        std::string cloth_render_output_dir;
        std::string cloth_render_camera_rig_path;
        bool cloth_render_fine = false;
        bool cloth_render_use_current_camera = false;
        float cloth_render_target_detail_size = 0.01f;
        int cloth_render_source_vertex_count = 0;
        int cloth_render_vertex_count = 0;
        int cloth_render_face_count = 0;

        std::uint64_t canva_displacement_request_id = 0;
        std::uint64_t canva_displacement_completed_id = 0;
        std::string canva_displacement_uv_json_path;
        std::string canva_displacement_model_input_path;
        std::string canva_displacement_path;
        std::string canva_manual_uv_png_path;
        std::string canva_manual_iso_npz_path;
        std::uint64_t canva_iso_displacement_request_id = 0;
        std::uint64_t canva_iso_displacement_completed_id = 0;
        std::string canva_iso_displacement_npz_path;
        std::string canva_iso_displacement_model_input_path;
        std::string canva_iso_displacement_path;

        float garment_min_cloth_skin_distance = 0.001f;
        float garment_ease_amount = 0.20f;
        int garment_prefit_steps = 10;
        bool cloth_constraints_edit_enabled = false;
        int cloth_constraint_edit_mode = 0; // 0 toggle, 1 add, 2 erase
        float cloth_constraint_pick_radius = 18.0f;
        int cloth_constraint_count = 0;
        std::string cloth_constraint_status;

        bool garment_vertex_merge_enabled = true;
        float garment_vertex_merge_distance = 0.0025f;

        float uv_view_zoom = 1.0f;
        float uv_view_pan_x = 0.0f;
        float uv_view_pan_y = 0.0f;

        float canva_brush_radius = 8.0f;
        std::array<float, 4> canva_brush_color{ 0.0f, 0.0f, 0.0f, 1.0f };
        int canva_sequence_pose_transition_frames = 2;
        float cloth_fitting_depth_margin = 0.0015f;
        float cloth_fitting_depth_margin_pred = 0.05f;
        float cloth_fitting_body_distance_threshold = 0.01f;
        float cloth_fitting_warp_radius = 48.0f;
        float cloth_fitting_warp_force = 1.0f;
        int cloth_fitting_warp_steps = 1;
        int cloth_fitting_debug_smoothing_radius = 3;
        int cloth_fitting_debug_smoothing_passes = 3;
        int cloth_fitting_debug_min_patch_area = 64;
        int cloth_fitting_debug_error_fill_radius = 0;
        int cloth_fitting_occlusion_cleanup_radius = 6;
        int cloth_fitting_occlusion_cleanup_passes = 2;
        float cloth_fitting_occlusion_chaos_threshold = 0.20f;
        int cloth_fitting_occlusion_min_patch_area = 24;
        float cloth_smoothing_filter_strength = 0.90f;
        float cloth_smoothing_inner_line_width = 4.0f;
        int cloth_smoothing_contour_radius = 10;
        int cloth_smoothing_line_gap_radius = 3;
        std::string cloth_fitting_output_dir;
        bool sketch2iso_continuity_filter_enabled = true;
        float sketch2iso_continuity_filter_strength = 0.50f;
        float sketch2iso_continuity_edge_threshold = 0.04f;
        int sketch2iso_continuity_iterations = 12;
        float iso_distance_threshold = 0.15f;

        bool canva_tab_active = false;
        std::uint64_t canva_undo_request_id = 0;
        int canva_preferred_bone_prediction = 0; // 0 custom, 1 NLF

        std::uint64_t asset_refresh_request_id = 0;
        std::uint64_t pose_asset_reload_request_id = 0;
        std::string pending_pose_selection;

        ModelInferenceUiState sketch2rig;
        ModelInferenceUiState sketch2betas;
        ModelInferenceUiState sketch2uv;
        ModelInferenceUiState sketch2iso;
        ModelInferenceUiState sketch2depth;
        ModelInferenceUiState nlf;
        ModelInferenceUiState vitpose;

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
        bool show_collision_colliders = false;
        bool garment_color_by_smpl_uv = false;
        ClothInkStyle cloth_ink_style = ClothInkStyle::Off;
        std::array<float, 3> cloth_ink_color{ 1.0f, 1.0f, 1.0f };
        bool show_console = true;
    };

    inline void console_add_line(AppState& state, std::string line)
    {
        console_add_line(state.console, std::move(line));
    }
}
