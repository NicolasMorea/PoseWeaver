#include "ui/canva/CanvaPanel.h"

#include "ui/canva/CanvaPreprocess.h"
#include "ui/canva/SketchPose.h"
#include "services/ModelInferenceService.h"
#include "core/Paths.h"
#include "render/Renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <glad/glad.h>
#include <imgui.h>
#include <GLFW/glfw3.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include <tinyfiledialogs.h>
#include <cnpy.h>

#ifndef POSEWEAVER_RIG_OPTIMIZER_WORKER
#define POSEWEAVER_RIG_OPTIMIZER_WORKER "src/rigopt/optimize_rig.py"
#endif

namespace poseweaver::ui_canva
{
    namespace
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
            ClothFittingPredictedDepthOcclusionDebug,
            ClothFittingAlignedPredictedDepthOcclusionDebug,
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

        static CanvaLayer canva_make_layer(int w, int h, int id, const std::string& name);
        static void canva_cancel_interactions(CanvaState& canva);
        static void canva_stamp_circle(CanvaLayer& layer, int w, int h, float cx, float cy, float radius, const ImVec4& col);
        static void canva_draw_segment(CanvaLayer& layer, int w, int h, ImVec2 a, ImVec2 b, float radius, const ImVec4& col);

        // Implementation sections are split by responsibility to keep this panel navigable.
#include "ui/canva/CanvaPanelPipelineUi.inc"
#include "ui/canva/CanvaPanelLayerIO.inc"
#include "ui/canva/CanvaPanelModelOps.inc"
#include "ui/canva/CanvaPanelCanvasEditing.inc"

    }

    void draw_canva_tab(AppState& state, Renderer& renderer)
    {
        static CanvaState canva;

        if (canva.layers.empty())
        {
            // Load persisted brush settings from AppState on first open.
            canva.brush_radius = state.canva_brush_radius;
            canva.brush_color = ImVec4(
                state.canva_brush_color[0],
                state.canva_brush_color[1],
                state.canva_brush_color[2],
                state.canva_brush_color[3]);

            canva.next_layer_id = 1;
            canva.layers.emplace_back(canva_make_layer(canva.width, canva.height, canva.next_layer_id++, "Layer 1"));
            canva.active_layer = 0;
            canva.zoom = 1.0f;
            canva.pan = ImVec2(0.0f, 0.0f);
            canva.tool = CanvaTool::Brush;

            // Avoid consuming stale requests if any.
            canva.last_undo_request_id = state.canva_undo_request_id;
        }
        canva.active_layer = std::clamp(canva.active_layer, 0, std::max(0, (int)canva.layers.size() - 1));

        // Per-frame service polling and deferred request handling.
#include "ui/canva/CanvaPanelFrameUpdates.inc"

        // Immediate-mode canvas view and side-panel controls.
#include "ui/canva/CanvaPanelDrawUi.inc"
    }
}
