#pragma once

#include "app/AppState.h"
#include "services/ModelInferenceService.h"
#include "ui/canva/SketchPose.h"

#include <glad/glad.h>
#include <imgui.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

namespace poseweaver { class Renderer; }

namespace poseweaver::ui_canva
{
    enum class CanvaTool
    {
        Brush,
        Transform
    };

    enum class CanvaLayerKind
    {
        Generic,
        Smpl,
        VitPose,
        Sketch2UV,
        Sketch2ISO,
        Sketch2Depth,
        ManualUV,
        GeneratedCloth,
        Displacement,
        IsoDisplacement,
        ClothFitting,
        ClothFittingWarped,
        ClothFittingDebug,
        ClothFittingWarpedDebug,
        ClothFittingOcclusionDebug,
        ClothFittingSceneOcclusionDebug,
        ClothFittingAlignedPredictedDepthOcclusionDebug,
        ClothFittingIsoOcclusionDebug,
        ClothFittingSceneDebug,
        ClothFittingDeformation
    };

    struct CanvaLayer
    {
        int id = 0;
        std::string name;
        CanvaLayerKind kind = CanvaLayerKind::Generic;
        bool enabled = true;

        // Layer transform in canvas pixels (top-left origin).
        // Canvas position = offset + (layer_pixel * scale).
        ImVec2 offset = ImVec2(0.0f, 0.0f);
        float scale = 1.0f;

        // Layer content is stored as a full canvas-sized RGBA8 texture.
        GLuint texture_id = 0;
        std::vector<unsigned char> pixels; // RGBA8
        bool dirty = true;

        // Cached bounds of non-transparent content in *layer pixel coordinates*.
        // max_* are exclusive.
        bool bounds_dirty = true;
        int bounds_min_x = 0;
        int bounds_min_y = 0;
        int bounds_max_x_excl = 0;
        int bounds_max_y_excl = 0;
    };

    struct CanvaUndo
    {
        int layer_id = 0;
        std::vector<unsigned char> pixels; // RGBA8 snapshot
        ImVec2 offset = ImVec2(0.0f, 0.0f);
        float scale = 1.0f;
    };

    struct RigOptimizationJobResult
    {
        bool ok = false;
        int exit_code = -1;
        std::filesystem::path output_json_path;
        std::string error;
    };

    static constexpr int kGenerateClothStepCount = 8;
    static constexpr int kGenerateClothBaseStepCount = 6;

    enum class GenerateClothStepId
    {
        Inference = 0,
        Pose = 1,
        Sequence = 2,
        Proxy = 3,
        Simulation = 4,
        Render = 5,
        FittingInference = 6,
        Fitting = 7
    };

    struct GenerateClothStepTiming
    {
        bool started = false;
        bool finished = false;
        bool skipped = false;
        double seconds = 0.0;
        std::chrono::steady_clock::time_point start_time{};
        std::vector<std::string> details;
    };

    struct CanvaState
    {
        int width = 1024;
        int height = 1024;

        float zoom = 1.0f;
        ImVec2 pan = ImVec2(0.0f, 0.0f);

        // Brush settings are persisted via AppState; these mirror AppState when the tab is open.
        float brush_radius = 8.0f; // in canvas pixels
        ImVec4 brush_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);

        CanvaTool tool = CanvaTool::Brush;

        // Brush stroke state
        bool drawing = false;
        ImVec2 last_canvas_pos = ImVec2(-1.0f, -1.0f);

        // Shift-drag brush sizing state
        bool sizing = false;
        ImVec2 size_start_mouse = ImVec2(0.0f, 0.0f);
        float size_start_radius = 0.0f;

        // Transform tool state
        bool transforming = false;
        bool scaling = false;
        int transform_layer = -1;
        ImVec2 transform_start_mouse = ImVec2(0.0f, 0.0f);
        ImVec2 transform_start_offset = ImVec2(0.0f, 0.0f);
        float transform_start_scale = 1.0f;
        ImVec2 transform_anchor_px = ImVec2(0.0f, 0.0f);
        ImVec2 transform_drag_px = ImVec2(0.0f, 0.0f);
        ImVec2 transform_anchor_canvas = ImVec2(0.0f, 0.0f);

        // Routed from GLFW key callback via AppState.
        std::uint64_t last_undo_request_id = 0;
        std::uint64_t last_sketch2rig_import_request_id = 0;
        std::uint64_t last_sketch2betas_import_request_id = 0;
        std::uint64_t last_sketch2uv_import_request_id = 0;
        std::uint64_t last_sketch2iso_import_request_id = 0;
        std::uint64_t last_sketch2depth_import_request_id = 0;
        std::uint64_t last_vitpose_import_request_id = 0;
        std::uint64_t ignore_sketch2rig_request_id = 0;
        std::uint64_t ignore_sketch2betas_request_id = 0;
        std::uint64_t ignore_sketch2uv_request_id = 0;
        std::uint64_t ignore_sketch2iso_request_id = 0;
        std::uint64_t ignore_sketch2depth_request_id = 0;
        std::uint64_t ignore_vitpose_request_id = 0;
        std::uint64_t last_scene_view_snapshot_completed_id = 0;
        std::uint64_t pending_scene_view_snapshot_request_id = 0;

        bool extract_pose_active = false;
        bool extract_pose_rig_done = false;
        bool extract_pose_betas_done = false;
        bool extract_pose_hide_rig_layer = false;
        bool extract_pose_generate_cloth_after = false;
        std::uint64_t extract_pose_rig_request_id = 0;
        std::uint64_t extract_pose_betas_request_id = 0;
        std::filesystem::path extract_pose_model_input_path;
        std::filesystem::path extract_pose_rig_path;
        std::filesystem::path extract_pose_betas_path;

        bool generate_cloth_active = false;
        bool generate_cloth_waiting_pose = false;
        bool generate_cloth_waiting_simulation = false;
        bool generate_cloth_waiting_render = false;
        std::uint64_t generate_cloth_render_request_id = 0;
        std::uint64_t last_cloth_render_completed_id = 0;
        std::filesystem::path generate_cloth_model_input_path;
        std::string generate_cloth_pose_name;
        std::string generate_cloth_status = "Idle";
        int generate_cloth_failed_step = -1; // 0 inference, 1 pose, 2 sequence, 3 proxy, 4 simulation, 5 render
        bool generate_cloth_fit_after_render = false;
        bool generate_cloth_include_fitting_steps = false;
        bool generate_cloth_timing_started = false;
        bool generate_cloth_timing_complete = false;
        double generate_cloth_total_seconds = 0.0;
        std::chrono::steady_clock::time_point generate_cloth_pipeline_start{};
        std::array<GenerateClothStepTiming, kGenerateClothStepCount> generate_cloth_step_timings;
        int generate_cloth_last_sim_phase = -1;
        std::chrono::steady_clock::time_point generate_cloth_extract_rig_start{};
        std::chrono::steady_clock::time_point generate_cloth_extract_betas_start{};
        std::chrono::steady_clock::time_point cloth_fitting_iso_start{};
        std::chrono::steady_clock::time_point cloth_fitting_depth_start{};

        bool render_cloth_active = false;
        bool render_cloth_fine = false;
        std::uint64_t render_cloth_request_id = 0;
        std::filesystem::path render_cloth_model_input_path;
        std::string render_cloth_status = "Idle";

        std::filesystem::path pending_scene_view_snapshot_model_input_path;

        bool generate_displacement_active = false;
        bool generate_displacement_waiting_uv = false;
        bool generate_displacement_waiting_render = false;
        std::uint64_t generate_displacement_uv_request_id = 0;
        std::uint64_t generate_displacement_render_request_id = 0;
        std::uint64_t last_displacement_completed_id = 0;
        std::filesystem::path generate_displacement_model_input_path;
        std::filesystem::path generate_displacement_uv_json_path;
        std::string generate_displacement_status = "Idle";

        bool generate_iso_displacement_active = false;
        bool generate_iso_displacement_waiting_iso = false;
        bool generate_iso_displacement_waiting_render = false;
        std::uint64_t generate_iso_displacement_iso_request_id = 0;
        std::uint64_t generate_iso_displacement_render_request_id = 0;
        std::uint64_t last_iso_displacement_completed_id = 0;
        std::filesystem::path generate_iso_displacement_model_input_path;
        std::filesystem::path generate_iso_displacement_npz_path;
        std::string generate_iso_displacement_status = "Idle";

        bool cloth_fitting_active = false;
        bool cloth_fitting_waiting_iso = false;
        bool cloth_fitting_waiting_depth = false;
        std::uint64_t cloth_fitting_iso_request_id = 0;
        std::uint64_t cloth_fitting_depth_request_id = 0;
        std::filesystem::path cloth_fitting_model_input_path;
        std::filesystem::path cloth_fitting_iso_path;
        std::filesystem::path cloth_fitting_depth_path;
        int cloth_fitting_source_layer_id = 0;
        std::string cloth_fitting_status = "Idle";

        bool rig_optimization_active = false;
        bool rig_optimization_waiting_uv = false;
        bool rig_optimization_job_running = false;
        std::uint64_t rig_optimization_uv_request_id = 0;
        std::filesystem::path rig_optimization_model_input_path;
        std::filesystem::path rig_optimization_uv_json_path;
        std::filesystem::path rig_optimization_output_json_path;
        std::string rig_optimization_status = "Idle";
        std::string rig_optimization_last_output;
        std::string rig_optimization_last_error;
        float rig_optimization_last_initial_loss = 0.0f;
        float rig_optimization_last_final_loss = 0.0f;
        int rig_optimization_iterations = 120;
        int rig_optimization_render_size = 160;
        float rig_optimization_lr = 0.035f;
        float rig_optimization_pose_reg = 0.018f;
        float rig_optimization_beta_reg = 0.030f;
        std::future<RigOptimizationJobResult> rig_optimization_future;
        bool rig_optimization_discard_result = false;

        int active_layer = 0;
        int next_layer_number = 2;
        int next_layer_id = 1;

        std::vector<CanvaLayer> layers;
        std::vector<CanvaUndo> undo;
    };

    struct Sketch2RigJoint2d
    {
        int index = -1;
        int parent = -1;
        float x = 0.0f;
        float y = 0.0f;
        bool valid = false;
    };

    struct VitPoseKeypoint2d
    {
        float x = 0.0f;
        float y = 0.0f;
        float score = 0.0f;
        bool valid = false;
    };

    struct CanvaPreprocessTransform
    {
        float scale = 1.0f;
        ImVec2 src_center = ImVec2(0.0f, 0.0f);
        ImVec2 dst_center = ImVec2(0.0f, 0.0f);
        bool valid = false;
    };

    enum class GenerateClothStepStatus
    {
        NotStarted,
        Working,
        Done,
        Failed
    };

    struct GenerateClothStepView
    {
        const char* label = "";
        GenerateClothStepStatus status = GenerateClothStepStatus::NotStarted;
        float progress = -1.0f;
        std::string detail;
        std::vector<std::string> details;
        double seconds = -1.0;
        double percent = -1.0;
    };

    struct CanvaBounds
    {
        int min_x = 0;
        int min_y = 0;
        int max_x_excl = 0;
        int max_y_excl = 0;
    };

    bool canva_add_canvas_filled_png_layer(
        CanvaState& canva,
        const std::string& file_path,
        const std::string& layer_name,
        CanvaLayerKind layer_kind,
        bool enabled,
        std::string* out_error);
    bool canva_add_generated_cloth_layer(
        CanvaState& canva,
        const std::string& file_path,
        const std::filesystem::path& model_input_path,
        bool enabled,
        std::string* out_error);
    bool canva_add_iso_debug_layer(
        CanvaState& canva,
        const std::filesystem::path& npz_path,
        const std::filesystem::path& model_input_path,
        const std::string& layer_name,
        bool enabled,
        std::string* out_error);
    bool canva_add_png_layer(CanvaState& canva, const char* file_path, std::string* out_error);
    bool canva_add_preprocess_aligned_png_layer(
        CanvaState& canva,
        const std::string& file_path,
        const std::filesystem::path& model_input_path,
        const std::string& layer_name,
        CanvaLayerKind layer_kind,
        bool enabled,
        std::string* out_error);
    bool canva_add_rendered_cloth_layer(
        CanvaState& canva,
        const std::string& file_path,
        const std::filesystem::path& model_input_path,
        bool fine,
        std::string* out_error);
    bool canva_add_sketch2depth_layer(
        CanvaState& canva,
        const std::filesystem::path& json_path,
        const std::filesystem::path& model_input_path,
        std::string* out_error);
    bool canva_add_sketch2rig_layer(
        CanvaState& canva,
        const std::filesystem::path& json_path,
        const std::filesystem::path& model_input_path,
        bool enabled,
        std::string* out_error);
    bool canva_add_sketch2uv_layer(
        CanvaState& canva,
        const std::filesystem::path& json_path,
        const std::filesystem::path& model_input_path,
        bool enabled,
        const std::string& layer_name,
        std::string* out_error);
    bool canva_add_smpl_render_layer(
        CanvaState& canva,
        const std::string& file_path,
        const std::filesystem::path& model_input_path,
        std::string* out_error);
    bool canva_add_vitpose_layer(
        CanvaState& canva,
        const std::filesystem::path& json_path,
        const std::filesystem::path& model_input_path,
        std::string* out_error);
    bool canva_apply_rig_optimization_result(
        AppState& state,
        CanvaState& canva,
        const std::filesystem::path& json_path,
        std::string* out_error);
    bool canva_apply_sketch2betas_result(
        AppState& state,
        const std::string& json_path,
        std::string* out_error);
    bool canva_apply_undo(CanvaState& canva);
    std::string canva_basename(std::string path);
    void canva_cancel_interactions(CanvaState& canva);
    bool canva_current_predicted_pose_path(
        const AppState& state,
        std::filesystem::path* out_pose_path,
        std::string* out_error);
    void canva_delete_layer_texture(CanvaLayer& layer);
    void canva_draw_segment(CanvaLayer& layer, int w, int h, ImVec2 a, ImVec2 b, float radius, const ImVec4& col);
    int canva_find_layer_index_by_id(const std::vector<CanvaLayer>& layers, int layer_id);
    int canva_find_layer_index_by_id(const CanvaState& canva, int layer_id);
    void canva_finish_cloth_fitting_if_ready(
        AppState& state,
        CanvaState& canva,
        Renderer& renderer);
    void canva_generate_cloth_append_detail(
        CanvaState& canva,
        GenerateClothStepId step,
        const std::string& detail);
    void canva_generate_cloth_append_timed_detail(
        CanvaState& canva,
        GenerateClothStepId step,
        const char* label,
        std::chrono::steady_clock::time_point start_time);
    void canva_generate_cloth_finish_pipeline(CanvaState& canva);
    void canva_generate_cloth_finish_simulation_steps(const AppState& state, CanvaState& canva);
    void canva_generate_cloth_finish_step(CanvaState& canva, GenerateClothStepId step, bool skipped = false);
    void canva_generate_cloth_reset_timings(CanvaState& canva, bool include_fitting_steps);
    void canva_generate_cloth_start_step(CanvaState& canva, GenerateClothStepId step);
    void canva_generate_cloth_update_simulation_phase(const AppState& state, CanvaState& canva);
    CanvaBounds canva_layer_content_bounds(CanvaLayer& layer, int w, int h);
    bool canva_layer_is_generated_cloth(const CanvaLayer& layer);
    bool canva_load_preprocess_transform(
        const std::filesystem::path& model_input_path,
        CanvaPreprocessTransform* out_transform,
        std::string* out_error);
    CanvaLayer canva_make_layer(int w, int h, int id, const std::string& name);
    bool canva_manual_iso_source_path(const AppState& state, std::filesystem::path* out_path);
    bool canva_manual_uv_source_path(const AppState& state, std::filesystem::path* out_path);
    bool canva_parse_number_array_field(
        const std::string& text,
        const char* key,
        std::vector<float>* out_values,
        std::string* out_error);
    bool canva_parse_number_field(const std::string& text, const char* key, double* out_value);
    std::string canva_parse_string_field(const std::string& text, const char* key);
    int canva_pick_layer_at(const CanvaState& canva, const ImVec2& canvas_pos);
    void canva_prune_undo_for_layer(std::vector<CanvaUndo>& undo, int layer_id);
    void canva_push_undo(CanvaState& canva, int layer_id, const std::vector<unsigned char>& pixels, const ImVec2& offset, float scale);
    bool canva_queue_cloth_fitting_predictions(
        AppState& state,
        CanvaState& canva,
        const CanvaLayer& source_layer,
        std::string* out_error);
    bool canva_queue_extract_pose(
        AppState& state,
        CanvaState& canva,
        bool generate_cloth_after,
        bool hide_rig_layer,
        std::string* out_error);
    bool canva_queue_generate_cloth_current_pose(
        AppState& state,
        CanvaState& canva,
        std::string* out_error);
    bool canva_queue_generate_displacement(
        AppState& state,
        CanvaState& canva,
        std::string* out_error);
    bool canva_queue_generate_iso_displacement(
        AppState& state,
        CanvaState& canva,
        std::string* out_error);
    bool canva_queue_render_cloth(
        AppState& state,
        CanvaState& canva,
        bool fine,
        std::string* out_error);
    bool canva_queue_rig_optimization(
        AppState& state,
        CanvaState& canva,
        std::string* out_error);
    void canva_reset_all(AppState& state, CanvaState& canva, Renderer& renderer);
    bool canva_save_model_input_png(
        const CanvaState& canva,
        std::filesystem::path* out_path,
        std::string* out_error);
    bool canva_save_visible_layers_png(const CanvaState& canva, std::filesystem::path* out_path, std::string* out_error);
    void canva_stamp_circle(CanvaLayer& layer, int w, int h, float cx, float cy, float radius, const ImVec4& col);
    bool canva_start_rig_optimization(
        AppState& state,
        CanvaState& canva,
        const std::filesystem::path& uv_json_path,
        const std::filesystem::path& model_input_path,
        std::string* out_error);
    bool canva_try_finish_extract_pose(AppState& state, CanvaState& canva);
    void canva_upload_layer(CanvaLayer& layer, int w, int h);
    void draw_generate_cloth_steps_in_view(
        const AppState& state,
        const CanvaState& canva,
        ImDrawList* dl,
        const ImVec2& anchor);
    int generate_cloth_step_index(GenerateClothStepId step);
    ImVec4 model_inference_status_color(ModelInferenceStatus status);
    const char* model_inference_status_text(ModelInferenceStatus status);
    void update_canva_frame(AppState& state, CanvaState& canva, Renderer& renderer);
    void draw_canva_canvas_view(AppState& state, CanvaState& canva, Renderer& renderer, float left_w, bool show_generate_cloth_steps);
    void draw_canva_controls(AppState& state, CanvaState& canva, Renderer& renderer);
}
