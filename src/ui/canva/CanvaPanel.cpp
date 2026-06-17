#include "ui/canva/CanvaPanel.h"

#include "ui/canva/CanvaPreprocess.h"
#include "ui/canva/SketchPose.h"
#include "services/ModelInferenceService.h"
#include "core/Paths.h"
#include "render/Renderer.h"

#include <algorithm>
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
            Sketch2UV,
            Sketch2ISO,
            ManualUV,
            GeneratedCloth,
            Displacement,
            IsoDisplacement,
            ClothFitting,
            ClothFittingDebug
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
            std::uint64_t ignore_sketch2rig_request_id = 0;
            std::uint64_t ignore_sketch2betas_request_id = 0;
            std::uint64_t ignore_sketch2uv_request_id = 0;
            std::uint64_t ignore_sketch2iso_request_id = 0;
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
            int generate_cloth_failed_step = -1; // 0 inference, 1 pose, 2 simulation, 3 render

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
            bool cloth_fitting_waiting_uv = false;
            std::uint64_t cloth_fitting_uv_request_id = 0;
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

        struct Sketch2RigJoint2d
        {
            int index = -1;
            int parent = -1;
            float x = 0.0f;
            float y = 0.0f;
            bool valid = false;
        };

        struct CanvaPreprocessTransform
        {
            float scale = 1.0f;
            ImVec2 src_center = ImVec2(0.0f, 0.0f);
            ImVec2 dst_center = ImVec2(0.0f, 0.0f);
            bool valid = false;
        };

        static const char* model_inference_status_text(ModelInferenceStatus status)
        {
            switch (status)
            {
            case ModelInferenceStatus::NotStarted: return "Not started";
            case ModelInferenceStatus::Loading: return "Loading";
            case ModelInferenceStatus::Ready: return "Loaded";
            case ModelInferenceStatus::Busy: return "Processing";
            case ModelInferenceStatus::Error: return "Error";
            case ModelInferenceStatus::Stopped: return "Stopped";
            }
            return "Unknown";
        }

        static ImVec4 model_inference_status_color(ModelInferenceStatus status)
        {
            switch (status)
            {
            case ModelInferenceStatus::Ready: return ImVec4(0.20f, 0.85f, 0.38f, 1.0f);
            case ModelInferenceStatus::Busy: return ImVec4(0.95f, 0.72f, 0.20f, 1.0f);
            case ModelInferenceStatus::Loading: return ImVec4(0.30f, 0.62f, 1.0f, 1.0f);
            case ModelInferenceStatus::Error: return ImVec4(1.0f, 0.32f, 0.28f, 1.0f);
            case ModelInferenceStatus::Stopped: return ImVec4(0.70f, 0.70f, 0.70f, 1.0f);
            case ModelInferenceStatus::NotStarted: return ImVec4(0.60f, 0.60f, 0.60f, 1.0f);
            }
            return ImVec4(0.60f, 0.60f, 0.60f, 1.0f);
        }

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
        };

        static ImVec4 generate_cloth_step_color(GenerateClothStepStatus status)
        {
            switch (status)
            {
            case GenerateClothStepStatus::Done: return ImVec4(0.20f, 0.85f, 0.38f, 1.0f);
            case GenerateClothStepStatus::Working: return ImVec4(0.95f, 0.72f, 0.20f, 1.0f);
            case GenerateClothStepStatus::Failed: return ImVec4(1.0f, 0.32f, 0.28f, 1.0f);
            case GenerateClothStepStatus::NotStarted: return ImVec4(0.50f, 0.52f, 0.55f, 1.0f);
            }
            return ImVec4(0.50f, 0.52f, 0.55f, 1.0f);
        }

        static const char* generate_cloth_step_status_text(GenerateClothStepStatus status)
        {
            switch (status)
            {
            case GenerateClothStepStatus::Done: return "Done";
            case GenerateClothStepStatus::Working: return "Working";
            case GenerateClothStepStatus::Failed: return "Failed";
            case GenerateClothStepStatus::NotStarted: return "Not started";
            }
            return "Not started";
        }

        static void build_generate_cloth_steps(const AppState& state, const CanvaState& canva, GenerateClothStepView (&steps)[4])
        {
            const bool failed = canva.generate_cloth_status == "Failed";
            const bool done = canva.generate_cloth_status == "Done";
            const bool infer_working = canva.generate_cloth_active && canva.extract_pose_active;
            const bool pose_working = canva.generate_cloth_waiting_pose;
            const bool sim_working = canva.generate_cloth_waiting_simulation;
            const bool render_working = canva.generate_cloth_waiting_render;
            const int failed_step = failed
                ? ((canva.generate_cloth_failed_step >= 0 && canva.generate_cloth_failed_step < 4)
                    ? canva.generate_cloth_failed_step
                    : 0)
                : -1;

            const bool infer_done = done || pose_working || sim_working || render_working;
            const bool pose_done = done || sim_working || render_working;
            const bool sim_done = done || render_working;
            const bool render_done = done;

            steps[0].label = "Inference";
            steps[0].status = infer_done ? GenerateClothStepStatus::Done
                : infer_working ? GenerateClothStepStatus::Working
                                : (failed ? GenerateClothStepStatus::Failed : GenerateClothStepStatus::NotStarted);

            steps[1].label = "Pose";
            steps[1].status = pose_done ? GenerateClothStepStatus::Done
                : pose_working ? GenerateClothStepStatus::Working
                               : GenerateClothStepStatus::NotStarted;

            steps[2].label = "Simulation";
            steps[2].status = sim_done ? GenerateClothStepStatus::Done
                : sim_working ? GenerateClothStepStatus::Working
                              : GenerateClothStepStatus::NotStarted;
            steps[2].progress = 0.0f;
            if (state.controls.sim_steps_total > 0 && (sim_working || sim_done))
            {
                steps[2].progress = (float)state.controls.sim_steps_done / (float)std::max(1, state.controls.sim_steps_total);
                std::ostringstream oss;
                oss << std::max(0, state.controls.sim_steps_done)
                    << "/" << std::max(0, state.controls.sim_steps_total) << " steps";
                steps[2].detail = oss.str();
            }
            else if (sim_working && !state.controls.sim_stage.empty())
            {
                steps[2].detail = state.controls.sim_stage;
            }

            steps[3].label = "Render";
            steps[3].status = render_done ? GenerateClothStepStatus::Done
                : render_working ? GenerateClothStepStatus::Working
                                 : GenerateClothStepStatus::NotStarted;

            if (failed)
            {
                for (int i = 0; i < 4; ++i)
                {
                    if (i < failed_step)
                        steps[i].status = GenerateClothStepStatus::Done;
                    else if (i == failed_step)
                        steps[i].status = GenerateClothStepStatus::Failed;
                    else
                        steps[i].status = GenerateClothStepStatus::NotStarted;
                }
            }
        }

        static void draw_generate_cloth_steps_in_view(
            const AppState& state,
            const CanvaState& canva,
            ImDrawList* dl,
            const ImVec2& anchor)
        {
            GenerateClothStepView steps[4];
            build_generate_cloth_steps(state, canva, steps);

            const ImFont* font = ImGui::GetFont();
            const float font_size = ImGui::GetFontSize();
            const float title_size = font_size + 2.0f;
            const float w = 220.0f;
            const float row_h = 56.0f;
            const float header_h = 46.0f;
            const float pad = 12.0f;
            const float h = header_h + row_h * 4.0f + pad;
            const ImVec2 p0 = anchor;
            const ImVec2 p1(p0.x + w, p0.y + h);

            dl->AddRectFilled(p0, p1, IM_COL32(18, 20, 24, 218), 7.0f);
            dl->AddRect(p0, p1, IM_COL32(255, 255, 255, 42), 7.0f, 0, 1.0f);
            dl->AddText(font, title_size, ImVec2(p0.x + pad, p0.y + 10.0f), IM_COL32(238, 240, 244, 255), "Generate Cloth");
            dl->AddText(font, font_size, ImVec2(p0.x + pad, p0.y + 28.0f), IM_COL32(176, 181, 190, 255), canva.generate_cloth_status.c_str());

            const float rail_x = p0.x + 22.0f;
            const float rows_y = p0.y + header_h;
            dl->AddLine(
                ImVec2(rail_x, rows_y + 17.0f),
                ImVec2(rail_x, rows_y + row_h * 4.0f - 18.0f),
                IM_COL32(255, 255, 255, 38),
                2.0f);

            for (int i = 0; i < 4; ++i)
            {
                const GenerateClothStepView& step = steps[i];
                const ImVec4 col = generate_cloth_step_color(step.status);
                const ImU32 accent = ImGui::ColorConvertFloat4ToU32(col);
                const ImU32 soft = ImGui::ColorConvertFloat4ToU32(ImVec4(col.x, col.y, col.z, 0.16f));
                const float y = rows_y + (float)i * row_h;
                const ImVec2 row0(p0.x + 8.0f, y + 4.0f);
                const ImVec2 row1(p1.x - 8.0f, y + row_h - 4.0f);

                dl->AddRectFilled(row0, row1, soft, 6.0f);
                dl->AddCircleFilled(ImVec2(rail_x, y + row_h * 0.5f), 5.5f, accent, 18);
                dl->AddText(font, font_size, ImVec2(p0.x + 38.0f, y + 11.0f), accent, step.label);
                dl->AddText(
                    font,
                    font_size - 1.0f,
                    ImVec2(p0.x + 38.0f, y + 29.0f),
                    IM_COL32(190, 195, 204, 255),
                    generate_cloth_step_status_text(step.status));

                if (step.progress >= 0.0f)
                {
                    const float bar_x0 = p0.x + 118.0f;
                    const float bar_x1 = p1.x - 16.0f;
                    const float bar_y = y + 34.0f;
                    const float t = std::clamp(step.progress, 0.0f, 1.0f);
                    dl->AddRectFilled(ImVec2(bar_x0, bar_y), ImVec2(bar_x1, bar_y + 5.0f), IM_COL32(62, 66, 74, 255), 2.5f);
                    dl->AddRectFilled(ImVec2(bar_x0, bar_y), ImVec2(bar_x0 + (bar_x1 - bar_x0) * t, bar_y + 5.0f), accent, 2.5f);
                }

                if (!step.detail.empty())
                {
                    dl->AddText(
                        font,
                        font_size - 1.0f,
                        ImVec2(p0.x + 118.0f, y + 12.0f),
                        IM_COL32(190, 195, 204, 255),
                        step.detail.c_str());
                }
            }
        }

        static bool canva_parse_number_array_field(
            const std::string& text,
            const char* key,
            std::vector<float>* out_values,
            std::string* out_error)
        {
            if (!out_values)
                return false;
            out_values->clear();

            const std::string needle = std::string("\"") + key + "\"";
            const size_t key_pos = text.find(needle);
            if (key_pos == std::string::npos)
            {
                if (out_error)
                    *out_error = std::string("Missing JSON field: ") + key;
                return false;
            }

            size_t i = text.find('[', key_pos + needle.size());
            if (i == std::string::npos)
            {
                if (out_error)
                    *out_error = std::string("JSON field is not an array: ") + key;
                return false;
            }
            ++i;

            while (i < text.size())
            {
                while (i < text.size() && (std::isspace((unsigned char)text[i]) || text[i] == ','))
                    ++i;
                if (i >= text.size())
                    break;
                if (text[i] == ']')
                    return !out_values->empty();

                const char* start = text.c_str() + i;
                char* end = nullptr;
                errno = 0;
                const float value = std::strtof(start, &end);
                if (end == start || errno == ERANGE)
                {
                    if (out_error)
                        *out_error = std::string("Failed to parse numeric JSON array field: ") + key;
                    return false;
                }
                out_values->push_back(value);
                i = (size_t)(end - text.c_str());
            }

            if (out_error)
                *out_error = std::string("Unterminated JSON array field: ") + key;
            return false;
        }

        static std::string canva_parse_string_field(const std::string& text, const char* key)
        {
            const std::string needle = std::string("\"") + key + "\"";
            const size_t key_pos = text.find(needle);
            if (key_pos == std::string::npos)
                return {};

            size_t start = text.find(':', key_pos + needle.size());
            if (start == std::string::npos)
                return {};
            ++start;
            while (start < text.size() && std::isspace((unsigned char)text[start]))
                ++start;
            if (start >= text.size() || text[start] != '"')
                return {};
            ++start;

            std::string out;
            bool escaped = false;
            for (size_t i = start; i < text.size(); ++i)
            {
                const char c = text[i];
                if (escaped)
                {
                    switch (c)
                    {
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    default: out += c; break;
                    }
                    escaped = false;
                    continue;
                }
                if (c == '\\')
                {
                    escaped = true;
                    continue;
                }
                if (c == '"')
                    break;
                out += c;
            }
            return out;
        }

        static bool canva_parse_number_field(const std::string& text, const char* key, double* out_value)
        {
            if (!out_value)
                return false;
            const std::string needle = std::string("\"") + key + "\"";
            size_t pos = text.find(needle);
            if (pos == std::string::npos)
                return false;
            pos = text.find(':', pos + needle.size());
            if (pos == std::string::npos)
                return false;
            ++pos;
            while (pos < text.size() && std::isspace((unsigned char)text[pos]))
                ++pos;

            const char* start = text.c_str() + pos;
            char* end = nullptr;
            errno = 0;
            const double value = std::strtod(start, &end);
            if (end == start || errno == ERANGE)
                return false;
            *out_value = value;
            return true;
        }

        static bool canva_apply_sketch2betas_result(
            AppState& state,
            const std::string& json_path,
            std::string* out_error)
        {
            std::ifstream in(json_path);
            if (!in)
            {
                if (out_error)
                    *out_error = "Failed to open result JSON";
                return false;
            }

            std::ostringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();

            std::vector<float> betas;
            if (!canva_parse_number_array_field(text, "betas", &betas, out_error))
                return false;

            const int n = std::min<int>((int)state.smpl_shape.betas.size(), (int)betas.size());
            for (int i = 0; i < n; ++i)
                state.smpl_shape.betas[(size_t)i] = betas[(size_t)i];
            for (int i = n; i < (int)state.smpl_shape.betas.size(); ++i)
                state.smpl_shape.betas[(size_t)i] = 0.0f;

            const std::string gender = canva_parse_string_field(text, "gender");
            if (gender == "male")
                state.selection.gender = SmplType::Male;
            else if (gender == "female")
                state.selection.gender = SmplType::Female;

            return true;
        }

        static void canva_apply_sketch_pose_result(AppState& state, const SketchPoseBuildResult& result)
        {
            if (result.gender == "male")
                state.selection.gender = SmplType::Male;
            else if (result.gender == "female")
                state.selection.gender = SmplType::Female;
            else if (result.gender == "neutral")
                state.selection.gender = SmplType::Neutral;

            const int n = std::min<int>((int)state.smpl_shape.betas.size(), (int)result.betas.size());
            for (int i = 0; i < n; ++i)
                state.smpl_shape.betas[(size_t)i] = (float)result.betas[(size_t)i];
            for (int i = n; i < (int)state.smpl_shape.betas.size(); ++i)
                state.smpl_shape.betas[(size_t)i] = 0.0f;

            state.pending_pose_selection = result.npz_path.filename().string();
            state.asset_refresh_request_id++;
            state.pose_asset_reload_request_id++;
        }

        static bool canva_try_finish_extract_pose(AppState& state, CanvaState& canva)
        {
            if (!canva.extract_pose_active || !canva.extract_pose_rig_done || !canva.extract_pose_betas_done)
                return false;

            const std::filesystem::path out_npz = assets_dir() / "poses" / "sketch_pose.npz";
            const std::filesystem::path out_json = assets_dir() / "poses" / "sketch_pose.json";

            SketchPoseBuildResult result;
            std::string err;
            if (!save_sketch_pose_from_outputs(
                    canva.extract_pose_rig_path,
                    canva.extract_pose_betas_path,
                    out_json,
                    out_npz,
                    &result,
                    &err))
            {
                console_add_line(state, std::string("Canva: Extract Pose failed: ") + (err.empty() ? "(unknown error)" : err));
                canva.extract_pose_active = false;
                canva.extract_pose_hide_rig_layer = false;
                canva.extract_pose_generate_cloth_after = false;
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Failed";
                canva.generate_cloth_failed_step = 0;
                return false;
            }

            canva_apply_sketch_pose_result(state, result);
            state.show_smpl = true;
            state.show_smpl_character = true;
            state.show_sketch2rig_rig = true;
            state.scene_view_snapshot_pose_name = result.npz_path.filename().string();
            state.scene_view_snapshot_path.clear();
            state.scene_view_snapshot_use_rigopt_camera = false;
            canva.pending_scene_view_snapshot_request_id = ++state.scene_view_snapshot_request_id;
            canva.pending_scene_view_snapshot_model_input_path = canva.extract_pose_model_input_path;
            state.scene_view_snap_to_target_request_id++;
            canva.extract_pose_active = false;
            if (canva.extract_pose_generate_cloth_after)
            {
                canva.generate_cloth_active = true;
                canva.generate_cloth_waiting_pose = true;
                canva.generate_cloth_waiting_simulation = false;
                canva.generate_cloth_waiting_render = false;
                if (canva.generate_cloth_model_input_path.empty())
                    canva.generate_cloth_model_input_path = canva.extract_pose_model_input_path;
                canva.generate_cloth_pose_name = result.npz_path.filename().string();
                canva.extract_pose_generate_cloth_after = false;
                canva.generate_cloth_status = "Posing";
                canva.generate_cloth_failed_step = -1;
                console_add_line(state, "Canva: Generate Cloth pose extracted; waiting for pose reload");
            }
            canva.extract_pose_hide_rig_layer = false;
            console_add_line(state, std::string("Canva: Saved sketch pose JSON: ") + result.json_path.string());
            console_add_line(state, std::string("Canva: Saved sketch pose NPZ: ") + result.npz_path.string());
            return true;
        }

        static bool canva_load_preprocess_transform(
            const std::filesystem::path& model_input_path,
            CanvaPreprocessTransform* out_transform,
            std::string* out_error)
        {
            if (!out_transform)
                return false;
            *out_transform = CanvaPreprocessTransform{};

            if (model_input_path.empty())
            {
                if (out_error)
                    *out_error = "Missing Sketch2Rig model-input path";
                return false;
            }

            const std::filesystem::path meta_path = model_input_path.string() + ".preprocess.json";
            std::ifstream in(meta_path);
            if (!in)
            {
                if (out_error)
                    *out_error = "Missing preprocessing metadata: " + meta_path.string();
                return false;
            }

            std::ostringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();

            std::vector<float> src_center;
            std::vector<float> dst_center;
            if (!canva_parse_number_array_field(text, "src_center", &src_center, out_error)
                || !canva_parse_number_array_field(text, "dst_center", &dst_center, out_error))
                return false;
            if (src_center.size() < 2 || dst_center.size() < 2)
            {
                if (out_error)
                    *out_error = "Malformed preprocessing center metadata";
                return false;
            }

            size_t scale_pos = text.find("\"scale\"");
            if (scale_pos == std::string::npos)
            {
                if (out_error)
                    *out_error = "Missing preprocessing scale metadata";
                return false;
            }
            scale_pos = text.find(':', scale_pos);
            if (scale_pos == std::string::npos)
            {
                if (out_error)
                    *out_error = "Malformed preprocessing scale metadata";
                return false;
            }
            ++scale_pos;

            while (scale_pos < text.size() && std::isspace((unsigned char)text[scale_pos]))
                ++scale_pos;
            const char* begin = text.c_str() + scale_pos;
            char* end = nullptr;
            const float scale = std::strtof(begin, &end);
            if (end == begin || scale <= 1e-6f)
            {
                if (out_error)
                    *out_error = "Invalid preprocessing scale metadata";
                return false;
            }

            out_transform->scale = scale;
            out_transform->src_center = ImVec2(src_center[0], src_center[1]);
            out_transform->dst_center = ImVec2(dst_center[0], dst_center[1]);
            out_transform->valid = true;
            return true;
        }

        static std::string canva_basename(std::string path)
        {
            // Handle both Unix and Windows separators.
            for (char& c : path)
            {
                if (c == '\\')
                    c = '/';
            }
            const size_t p = path.find_last_of('/');
            if (p != std::string::npos)
                path = path.substr(p + 1);
            return path;
        }

        static std::vector<unsigned char> canva_resample_bilinear_rgba(
            const unsigned char* src,
            int src_w,
            int src_h,
            int dst_w,
            int dst_h)
        {
            std::vector<unsigned char> dst;
            if (!src || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0)
                return dst;

            dst.assign((size_t)dst_w * (size_t)dst_h * 4u, 0u);

            if (src_w == dst_w && src_h == dst_h)
            {
                std::copy(src, src + (size_t)src_w * (size_t)src_h * 4u, dst.data());
                return dst;
            }

            const float sx = (float)src_w / (float)dst_w;
            const float sy = (float)src_h / (float)dst_h;
            for (int y = 0; y < dst_h; ++y)
            {
                const float fy = ((float)y + 0.5f) * sy - 0.5f;
                int y0 = (int)std::floor(fy);
                int y1 = y0 + 1;
                const float ty = fy - (float)y0;
                y0 = std::clamp(y0, 0, src_h - 1);
                y1 = std::clamp(y1, 0, src_h - 1);

                for (int x = 0; x < dst_w; ++x)
                {
                    const float fx = ((float)x + 0.5f) * sx - 0.5f;
                    int x0 = (int)std::floor(fx);
                    int x1 = x0 + 1;
                    const float tx = fx - (float)x0;
                    x0 = std::clamp(x0, 0, src_w - 1);
                    x1 = std::clamp(x1, 0, src_w - 1);

                    const unsigned char* p00 = src + (((size_t)y0 * (size_t)src_w + (size_t)x0) * 4u);
                    const unsigned char* p10 = src + (((size_t)y0 * (size_t)src_w + (size_t)x1) * 4u);
                    const unsigned char* p01 = src + (((size_t)y1 * (size_t)src_w + (size_t)x0) * 4u);
                    const unsigned char* p11 = src + (((size_t)y1 * (size_t)src_w + (size_t)x1) * 4u);

                    for (int c = 0; c < 4; ++c)
                    {
                        const float a = (float)p00[c] + ((float)p10[c] - (float)p00[c]) * tx;
                        const float b = (float)p01[c] + ((float)p11[c] - (float)p01[c]) * tx;
                        const float v = a + (b - a) * ty;
                        dst[((size_t)y * (size_t)dst_w + (size_t)x) * 4u + (size_t)c] =
                            (unsigned char)std::clamp<int>((int)std::lround(v), 0, 255);
                    }
                }
            }

            return dst;
        }

        struct CanvaBounds
        {
            int min_x = 0;
            int min_y = 0;
            int max_x_excl = 0;
            int max_y_excl = 0;
        };

        static CanvaBounds canva_layer_content_bounds(CanvaLayer& layer, int w, int h)
        {
            if (!layer.bounds_dirty)
                return { layer.bounds_min_x, layer.bounds_min_y, layer.bounds_max_x_excl, layer.bounds_max_y_excl };

            // Default to full canvas if empty/invalid.
            int min_x = w;
            int min_y = h;
            int max_x = -1;
            int max_y = -1;

            if (!layer.pixels.empty() && w > 0 && h > 0)
            {
                for (int y = 0; y < h; ++y)
                {
                    const size_t row = (size_t)y * (size_t)w * 4u;
                    for (int x = 0; x < w; ++x)
                    {
                        const unsigned char a = layer.pixels[row + (size_t)x * 4u + 3u];
                        if (a == 0u)
                            continue;
                        min_x = std::min(min_x, x);
                        min_y = std::min(min_y, y);
                        max_x = std::max(max_x, x);
                        max_y = std::max(max_y, y);
                    }
                }
            }

            if (max_x < 0 || max_y < 0)
            {
                layer.bounds_min_x = 0;
                layer.bounds_min_y = 0;
                layer.bounds_max_x_excl = std::max(0, w);
                layer.bounds_max_y_excl = std::max(0, h);
            }
            else
            {
                layer.bounds_min_x = std::clamp(min_x, 0, std::max(0, w));
                layer.bounds_min_y = std::clamp(min_y, 0, std::max(0, h));
                layer.bounds_max_x_excl = std::clamp(max_x + 1, 0, std::max(0, w));
                layer.bounds_max_y_excl = std::clamp(max_y + 1, 0, std::max(0, h));
            }

            layer.bounds_dirty = false;
            return { layer.bounds_min_x, layer.bounds_min_y, layer.bounds_max_x_excl, layer.bounds_max_y_excl };
        }

        static bool canva_add_png_layer(CanvaState& canva, const char* file_path, std::string* out_error)
        {
            if (!file_path || file_path[0] == '\0')
            {
                if (out_error)
                    *out_error = "Empty file path";
                return false;
            }

            int iw = 0, ih = 0, comp = 0;
            stbi_uc* data = stbi_load(file_path, &iw, &ih, &comp, 4);
            if (!data)
            {
                if (out_error)
                    *out_error = std::string("Failed to load image: ") + file_path;
                return false;
            }

            const int w = canva.width;
            const int h = canva.height;
            if (w <= 0 || h <= 0)
            {
                stbi_image_free(data);
                if (out_error)
                    *out_error = "Invalid canva size";
                return false;
            }

            const std::string name = canva_basename(file_path);
            CanvaLayer layer = canva_make_layer(w, h, canva.next_layer_id++, name);

            // Ensure the whole image fits within the 1024x1024 canva by downsampling if needed.
            const float s = std::min(1.0f, std::min((float)w / (float)std::max(1, iw), (float)h / (float)std::max(1, ih)));
            const int dw = std::clamp((int)std::floor((float)iw * s + 0.5f), 1, w);
            const int dh = std::clamp((int)std::floor((float)ih * s + 0.5f), 1, h);
            const std::vector<unsigned char> resized = canva_resample_bilinear_rgba((const unsigned char*)data, iw, ih, dw, dh);

            const int off_x = std::max(0, (w - dw) / 2);
            const int off_y = std::max(0, (h - dh) / 2);
            for (int y = 0; y < dh; ++y)
            {
                for (int x = 0; x < dw; ++x)
                {
                    const size_t src = ((size_t)y * (size_t)dw + (size_t)x) * 4u;
                    const size_t dst = ((size_t)(y + off_y) * (size_t)w + (size_t)(x + off_x)) * 4u;
                    layer.pixels[dst + 0u] = resized[src + 0u];
                    layer.pixels[dst + 1u] = resized[src + 1u];
                    layer.pixels[dst + 2u] = resized[src + 2u];
                    layer.pixels[dst + 3u] = resized[src + 3u];
                }
            }

            stbi_image_free(data);

            layer.dirty = true;
            layer.bounds_dirty = true;
            canva.layers.insert(canva.layers.begin(), std::move(layer));
            canva.active_layer = 0;
            canva_cancel_interactions(canva);
            return true;
        }

        static bool canva_add_canvas_filled_png_layer(
            CanvaState& canva,
            const std::string& file_path,
            const std::string& layer_name,
            CanvaLayerKind layer_kind,
            bool enabled,
            std::string* out_error)
        {
            if (file_path.empty())
            {
                if (out_error)
                    *out_error = "Empty file path";
                return false;
            }

            int iw = 0, ih = 0, comp = 0;
            stbi_uc* data = stbi_load(file_path.c_str(), &iw, &ih, &comp, 4);
            if (!data)
            {
                if (out_error)
                    *out_error = "Failed to load image: " + file_path;
                return false;
            }

            const int w = canva.width;
            const int h = canva.height;
            if (w <= 0 || h <= 0)
            {
                stbi_image_free(data);
                if (out_error)
                    *out_error = "Invalid canva size";
                return false;
            }

            CanvaLayer layer = canva_make_layer(w, h, canva.next_layer_id++, layer_name);
            layer.kind = layer_kind;
            layer.pixels = canva_resample_bilinear_rgba((const unsigned char*)data, iw, ih, w, h);
            stbi_image_free(data);

            layer.enabled = enabled;
            layer.dirty = true;
            layer.bounds_dirty = true;
            canva.layers.insert(canva.layers.begin(), std::move(layer));
            canva.active_layer = 0;
            canva_cancel_interactions(canva);
            return true;
        }

        static bool canva_add_preprocess_aligned_png_layer(
            CanvaState& canva,
            const std::string& file_path,
            const std::filesystem::path& model_input_path,
            const std::string& layer_name,
            CanvaLayerKind layer_kind,
            bool enabled,
            std::string* out_error)
        {
            if (file_path.empty())
            {
                if (out_error)
                    *out_error = "Empty file path";
                return false;
            }

            CanvaPreprocessTransform preprocess_transform;
            std::string transform_error;
            if (!canva_load_preprocess_transform(model_input_path, &preprocess_transform, &transform_error))
            {
                if (out_error)
                    *out_error = transform_error.empty() ? "Missing preprocessing transform" : transform_error;
                return false;
            }

            int iw = 0, ih = 0, comp = 0;
            stbi_uc* data = stbi_load(file_path.c_str(), &iw, &ih, &comp, 4);
            if (!data)
            {
                if (out_error)
                    *out_error = "Failed to load image: " + file_path;
                return false;
            }

            const int w = canva.width;
            const int h = canva.height;
            if (w <= 0 || h <= 0)
            {
                stbi_image_free(data);
                if (out_error)
                    *out_error = "Invalid canva size";
                return false;
            }

            CanvaLayer layer = canva_make_layer(w, h, canva.next_layer_id++, layer_name);
            layer.kind = layer_kind;
            layer.pixels = canva_resample_bilinear_rgba((const unsigned char*)data, iw, ih, w, h);
            stbi_image_free(data);

            const float inv_scale = 1.0f / std::max(1e-6f, preprocess_transform.scale);
            layer.scale = inv_scale;
            layer.offset = ImVec2(
                preprocess_transform.src_center.x - preprocess_transform.dst_center.x * inv_scale,
                preprocess_transform.src_center.y - preprocess_transform.dst_center.y * inv_scale);
            layer.enabled = enabled;
            layer.dirty = true;
            layer.bounds_dirty = true;

            canva.layers.insert(canva.layers.begin(), std::move(layer));
            canva.active_layer = 0;
            canva_cancel_interactions(canva);
            return true;
        }

        static bool canva_add_smpl_render_layer(
            CanvaState& canva,
            const std::string& file_path,
            const std::filesystem::path& model_input_path,
            std::string* out_error)
        {
            if (!canva_add_preprocess_aligned_png_layer(
                    canva,
                    file_path,
                    model_input_path,
                    "SMPL",
                    CanvaLayerKind::Smpl,
                    true,
                    out_error))
                return false;
            return true;
        }

        static bool canva_add_generated_cloth_layer(
            CanvaState& canva,
            const std::string& file_path,
            const std::filesystem::path& model_input_path,
            std::string* out_error)
        {
            return canva_add_preprocess_aligned_png_layer(
                canva,
                file_path,
                model_input_path,
                "Generated cloth",
                CanvaLayerKind::GeneratedCloth,
                true,
                out_error);
        }

        static bool canva_add_rendered_cloth_layer(
            CanvaState& canva,
            const std::string& file_path,
            const std::filesystem::path& model_input_path,
            bool fine,
            std::string* out_error)
        {
            const std::string name = fine ? "Fine cloth render" : "Rendered cloth";
            if (model_input_path.empty())
            {
                return canva_add_canvas_filled_png_layer(
                    canva,
                    file_path,
                    name,
                    CanvaLayerKind::GeneratedCloth,
                    true,
                    out_error);
            }
            return canva_add_preprocess_aligned_png_layer(
                canva,
                file_path,
                model_input_path,
                name,
                CanvaLayerKind::GeneratedCloth,
                true,
                out_error);
        }

        static bool canva_add_sketch2uv_layer(
            CanvaState& canva,
            const std::filesystem::path& json_path,
            const std::filesystem::path& model_input_path,
            bool enabled,
            const std::string& layer_name,
            std::string* out_error)
        {
            std::ifstream in(json_path);
            if (!in)
            {
                if (out_error)
                    *out_error = "Failed to open " + json_path.string();
                return false;
            }

            std::ostringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();

            std::filesystem::path render_path = canva_parse_string_field(text, "render_path");
            if (render_path.empty())
                render_path = json_path.parent_path() / (json_path.stem().string() + "_render.png");
            if (render_path.is_relative())
                render_path = json_path.parent_path() / render_path;

            int iw = 0, ih = 0, comp = 0;
            stbi_uc* data = stbi_load(render_path.string().c_str(), &iw, &ih, &comp, 4);
            if (!data)
            {
                if (out_error)
                    *out_error = "Failed to load UV render PNG: " + render_path.string();
                return false;
            }

            const int w = canva.width;
            const int h = canva.height;
            if (w <= 0 || h <= 0)
            {
                stbi_image_free(data);
                if (out_error)
                    *out_error = "Invalid canva size";
                return false;
            }

            CanvaPreprocessTransform preprocess_transform;
            std::string transform_error;
            if (!canva_load_preprocess_transform(model_input_path, &preprocess_transform, &transform_error))
            {
                stbi_image_free(data);
                if (out_error)
                    *out_error = transform_error.empty() ? "Missing preprocessing transform" : transform_error;
                return false;
            }

            CanvaLayer layer = canva_make_layer(w, h, canva.next_layer_id++, layer_name);
            layer.kind = CanvaLayerKind::Sketch2UV;
            layer.pixels = canva_resample_bilinear_rgba((const unsigned char*)data, iw, ih, w, h);
            stbi_image_free(data);

            const float inv_scale = 1.0f / std::max(1e-6f, preprocess_transform.scale);
            layer.scale = inv_scale;
            layer.offset = ImVec2(
                preprocess_transform.src_center.x - preprocess_transform.dst_center.x * inv_scale,
                preprocess_transform.src_center.y - preprocess_transform.dst_center.y * inv_scale);
            layer.enabled = enabled;
            layer.dirty = true;
            layer.bounds_dirty = true;

            canva.layers.insert(canva.layers.begin(), std::move(layer));
            canva.active_layer = 0;
            if (!enabled && canva.layers.size() > 1)
                canva.active_layer = 1;
            canva_cancel_interactions(canva);
            return true;
        }

        static bool canva_add_iso_debug_layer(
            CanvaState& canva,
            const std::filesystem::path& npz_path,
            const std::filesystem::path& model_input_path,
            const std::string& layer_name,
            bool enabled,
            std::string* out_error)
        {
            cnpy::NpyArray isomap;
            try
            {
                isomap = cnpy::npz_load(npz_path.string(), "isomap");
            }
            catch (const std::exception& e)
            {
                if (out_error)
                    *out_error = "Failed to load isomap from " + npz_path.string() + ": " + e.what();
                return false;
            }

            if (isomap.shape.size() != 3 || (isomap.word_size != sizeof(float) && isomap.word_size != sizeof(double)))
            {
                if (out_error)
                    *out_error = "ISO NPZ must contain a float isomap array shaped [H,W,C] or [C,H,W]";
                return false;
            }

            const bool hwc = isomap.shape[2] >= 3 && isomap.shape[2] <= 256;
            const bool chw = !hwc && isomap.shape[0] >= 3 && isomap.shape[0] <= 256;
            if (!hwc && !chw)
            {
                if (out_error)
                    *out_error = "Could not identify the isomap channel dimension";
                return false;
            }

            const int ih = (int)(hwc ? isomap.shape[0] : isomap.shape[1]);
            const int iw = (int)(hwc ? isomap.shape[1] : isomap.shape[2]);
            const int channels = (int)(hwc ? isomap.shape[2] : isomap.shape[0]);
            if (iw <= 0 || ih <= 0 || channels < 3)
            {
                if (out_error)
                    *out_error = "ISO NPZ has an invalid isomap shape";
                return false;
            }

            cnpy::NpyArray mask;
            bool have_mask = false;
            try
            {
                mask = cnpy::npz_load(npz_path.string(), "mask");
                have_mask = mask.shape.size() == 2
                    && (int)mask.shape[0] == ih
                    && (int)mask.shape[1] == iw;
            }
            catch (const std::exception&)
            {
            }

            cnpy::NpyArray descriptor_min;
            cnpy::NpyArray descriptor_max;
            bool have_descriptor_range = false;
            try
            {
                descriptor_min = cnpy::npz_load(npz_path.string(), "descriptor_min");
                descriptor_max = cnpy::npz_load(npz_path.string(), "descriptor_max");
                have_descriptor_range =
                    descriptor_min.shape.size() == 1
                    && descriptor_max.shape.size() == 1
                    && descriptor_min.shape[0] >= 3
                    && descriptor_max.shape[0] >= 3
                    && (descriptor_min.word_size == sizeof(float) || descriptor_min.word_size == sizeof(double))
                    && (descriptor_max.word_size == sizeof(float) || descriptor_max.word_size == sizeof(double));
            }
            catch (const std::exception&)
            {
            }

            const auto float_value = [](const cnpy::NpyArray& array, size_t index) -> float
            {
                if (array.word_size == sizeof(float))
                    return array.data<float>()[index];
                return (float)array.data<double>()[index];
            };
            const auto mask_value = [](const cnpy::NpyArray& array, size_t index) -> float
            {
                if (array.word_size == sizeof(float))
                    return array.data<float>()[index];
                if (array.word_size == sizeof(double))
                    return (float)array.data<double>()[index];
                if (array.word_size == sizeof(std::uint8_t))
                    return array.data<std::uint8_t>()[index] != 0 ? 1.0f : 0.0f;
                return 1.0f;
            };

            std::vector<unsigned char> rgba((size_t)iw * (size_t)ih * 4u, 0u);
            for (int y = 0; y < ih; ++y)
            {
                for (int x = 0; x < iw; ++x)
                {
                    const size_t pixel = (size_t)y * (size_t)iw + (size_t)x;
                    unsigned char* dst = rgba.data() + pixel * 4u;
                    for (int c = 0; c < 3; ++c)
                    {
                        const size_t source = hwc
                            ? (pixel * (size_t)channels + (size_t)c)
                            : ((size_t)c * (size_t)ih * (size_t)iw + pixel);
                        float value = float_value(isomap, source);
                        if (have_descriptor_range)
                        {
                            const float lo = float_value(descriptor_min, (size_t)c);
                            const float hi = float_value(descriptor_max, (size_t)c);
                            value = (value - lo) / std::max(1e-8f, hi - lo);
                        }
                        dst[c] = (unsigned char)std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f);
                    }
                    const float alpha = have_mask ? mask_value(mask, pixel) : 1.0f;
                    dst[3] = (unsigned char)std::lround(std::clamp(alpha, 0.0f, 1.0f) * 255.0f);
                }
            }

            CanvaLayer layer = canva_make_layer(canva.width, canva.height, canva.next_layer_id++, layer_name);
            layer.kind = CanvaLayerKind::Sketch2ISO;
            layer.pixels = canva_resample_bilinear_rgba(rgba.data(), iw, ih, canva.width, canva.height);

            if (!model_input_path.empty())
            {
                CanvaPreprocessTransform preprocess_transform;
                std::string transform_error;
                if (!canva_load_preprocess_transform(model_input_path, &preprocess_transform, &transform_error))
                {
                    if (out_error)
                        *out_error = transform_error.empty() ? "Missing preprocessing transform" : transform_error;
                    return false;
                }
                const float inv_scale = 1.0f / std::max(1e-6f, preprocess_transform.scale);
                layer.scale = inv_scale;
                layer.offset = ImVec2(
                    preprocess_transform.src_center.x - preprocess_transform.dst_center.x * inv_scale,
                    preprocess_transform.src_center.y - preprocess_transform.dst_center.y * inv_scale);
            }

            layer.enabled = enabled;
            layer.dirty = true;
            layer.bounds_dirty = true;
            canva.layers.insert(canva.layers.begin(), std::move(layer));
            canva.active_layer = 0;
            if (!enabled && canva.layers.size() > 1)
                canva.active_layer = 1;
            canva_cancel_interactions(canva);
            return true;
        }

        static int canva_find_layer_index_by_id(const CanvaState& canva, int layer_id)
        {
            for (int i = 0; i < (int)canva.layers.size(); ++i)
            {
                if (canva.layers[(size_t)i].id == layer_id)
                    return i;
            }
            return -1;
        }

        static bool canva_manual_uv_source_path(const AppState& state, std::filesystem::path* out_path)
        {
            if (!out_path)
                return false;
            std::error_code ec;
            if (!state.canva_manual_uv_png_path.empty())
            {
                std::filesystem::path p = state.canva_manual_uv_png_path;
                if (std::filesystem::exists(p, ec))
                {
                    *out_path = p;
                    return true;
                }
            }
            return false;
        }

        static bool canva_manual_iso_source_path(const AppState& state, std::filesystem::path* out_path)
        {
            if (!out_path)
                return false;
            std::error_code ec;
            if (!state.canva_manual_iso_npz_path.empty())
            {
                const std::filesystem::path path = state.canva_manual_iso_npz_path;
                if (std::filesystem::exists(path, ec))
                {
                    *out_path = path;
                    return true;
                }
            }
            return false;
        }

        static bool canva_current_uv_source_path(const AppState& state, std::filesystem::path* out_path)
        {
            if (!out_path)
                return false;
            if (canva_manual_uv_source_path(state, out_path))
                return true;

            std::error_code ec;
            if (!state.sketch2uv.last_output_path.empty())
            {
                std::filesystem::path p = state.sketch2uv.last_output_path;
                if (std::filesystem::exists(p, ec))
                {
                    *out_path = p;
                    return true;
                }
            }
            if (!state.canva_displacement_uv_json_path.empty())
            {
                std::filesystem::path p = state.canva_displacement_uv_json_path;
                ec.clear();
                if (std::filesystem::exists(p, ec))
                {
                    *out_path = p;
                    return true;
                }
            }
            return false;
        }

        static void canva_uv_from_cloth_transform(
            const CanvaState& canva,
            const CanvaLayer& cloth_layer,
            const std::filesystem::path& uv_source_path,
            float* out_scale,
            float* out_offset_x,
            float* out_offset_y)
        {
            ImVec2 uv_offset(0.0f, 0.0f);
            float uv_scale = 1.0f;

            std::string ext = uv_source_path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                [](unsigned char c) { return (char)std::tolower(c); });

            if (ext == ".png")
            {
                for (const auto& layer : canva.layers)
                {
                    if (layer.kind != CanvaLayerKind::ManualUV)
                        continue;
                    uv_offset = layer.offset;
                    uv_scale = layer.scale;
                    break;
                }
            }
            else
            {
                std::ifstream in(uv_source_path);
                if (in)
                {
                    std::ostringstream ss;
                    ss << in.rdbuf();
                    std::filesystem::path model_input_path =
                        canva_parse_string_field(ss.str(), "source_sketch");
                    if (model_input_path.is_relative())
                        model_input_path = uv_source_path.parent_path() / model_input_path;

                    CanvaPreprocessTransform transform;
                    std::string ignored_error;
                    if (canva_load_preprocess_transform(model_input_path, &transform, &ignored_error))
                    {
                        uv_scale = 1.0f / std::max(1e-6f, transform.scale);
                        uv_offset = ImVec2(
                            transform.src_center.x - transform.dst_center.x * uv_scale,
                            transform.src_center.y - transform.dst_center.y * uv_scale);
                    }
                }
            }

            uv_scale = std::max(1e-6f, uv_scale);
            *out_scale = cloth_layer.scale / uv_scale;
            *out_offset_x = (cloth_layer.offset.x - uv_offset.x) / uv_scale;
            *out_offset_y = (cloth_layer.offset.y - uv_offset.y) / uv_scale;
        }

        static bool canva_layer_is_generated_cloth(const CanvaLayer& layer)
        {
            if (layer.kind == CanvaLayerKind::GeneratedCloth)
                return true;

            std::string name = layer.name;
            std::transform(name.begin(), name.end(), name.begin(),
                [](unsigned char c) { return (char)std::tolower(c); });
            return name.find("generated cloth") != std::string::npos;
        }

        static bool canva_add_cloth_fitting_layers(
            CanvaState& canva,
            const CanvaLayer& source_layer,
            std::vector<unsigned char> pixels,
            std::vector<unsigned char> debug_pixels,
            std::string* out_error)
        {
            if ((int)pixels.size() < canva.width * canva.height * 4)
            {
                if (out_error)
                    *out_error = "fitted pixel buffer is too small";
                return false;
            }
            if ((int)debug_pixels.size() < canva.width * canva.height * 4)
            {
                if (out_error)
                    *out_error = "debug pixel buffer is too small";
                return false;
            }

            const ImVec2 source_offset = source_layer.offset;
            const float source_scale = source_layer.scale;

            CanvaLayer layer = canva_make_layer(canva.width, canva.height, canva.next_layer_id++, "cloth Fitting");
            layer.kind = CanvaLayerKind::ClothFitting;
            layer.pixels = std::move(pixels);
            layer.offset = source_offset;
            layer.scale = source_scale;
            layer.enabled = true;
            layer.dirty = true;
            layer.bounds_dirty = true;

            canva.layers.insert(canva.layers.begin(), std::move(layer));

            CanvaLayer debug_layer = canva_make_layer(canva.width, canva.height, canva.next_layer_id++, "cloth Fitting Debug");
            debug_layer.kind = CanvaLayerKind::ClothFittingDebug;
            debug_layer.pixels = std::move(debug_pixels);
            debug_layer.offset = source_offset;
            debug_layer.scale = source_scale;
            debug_layer.enabled = true;
            debug_layer.dirty = true;
            debug_layer.bounds_dirty = true;

            canva.layers.insert(canva.layers.begin(), std::move(debug_layer));
            canva.active_layer = 0;
            canva_cancel_interactions(canva);
            return true;
        }

        static bool canva_run_cloth_fitting(
            AppState& state,
            CanvaState& canva,
            Renderer& renderer,
            const std::filesystem::path& uv_json_path,
            int source_layer_index,
            std::string* out_error)
        {
            if (source_layer_index < 0)
                source_layer_index = canva.active_layer;
            if (source_layer_index < 0 || source_layer_index >= (int)canva.layers.size())
            {
                if (out_error)
                    *out_error = "select a Generated cloth layer first";
                return false;
            }

            const CanvaLayer& source = canva.layers[(size_t)source_layer_index];
            if (!canva_layer_is_generated_cloth(source))
            {
                if (out_error)
                    *out_error = "selected layer is not a Generated cloth layer";
                return false;
            }
            if (source.pixels.size() < (size_t)canva.width * (size_t)canva.height * 4u)
            {
                if (out_error)
                    *out_error = "selected cloth layer has no pixels";
                return false;
            }
            if (uv_json_path.empty())
            {
                if (out_error)
                    *out_error = "missing UV source";
                return false;
            }

            std::vector<unsigned char> fitted;
            std::vector<unsigned char> debug;
            int tested_count = 0;
            int removed_count = 0;
            float uv_from_cloth_scale = 1.0f;
            float uv_from_cloth_offset_x = 0.0f;
            float uv_from_cloth_offset_y = 0.0f;
            canva_uv_from_cloth_transform(
                canva,
                source,
                uv_json_path,
                &uv_from_cloth_scale,
                &uv_from_cloth_offset_x,
                &uv_from_cloth_offset_y);
            std::string fit_err;
            if (!renderer.fit_cloth_render_to_smpl_visibility(
                    state,
                    source.pixels,
                    canva.width,
                    canva.height,
                    uv_json_path.string(),
                    uv_from_cloth_scale,
                    uv_from_cloth_offset_x,
                    uv_from_cloth_offset_y,
                    fitted,
                    debug,
                    tested_count,
                    removed_count,
                    &fit_err))
            {
                if (out_error)
                    *out_error = fit_err.empty() ? "renderer fitting failed" : fit_err;
                return false;
            }

            std::string add_err;
            if (!canva_add_cloth_fitting_layers(canva, source, std::move(fitted), std::move(debug), &add_err))
            {
                if (out_error)
                    *out_error = add_err.empty() ? "failed to create output layer" : add_err;
                return false;
            }

            console_add_line(state,
                "Canva: cloth Fitting added layer (removed="
                    + std::to_string(removed_count)
                    + ", tested=" + std::to_string(tested_count) + ")");
            return true;
        }

        static bool canva_parse_json_number(const std::string& text, size_t* pos, double* out_value)
        {
            if (!pos || !out_value)
                return false;

            size_t i = *pos;
            while (i < text.size() && std::isspace((unsigned char)text[i]))
                ++i;

            const char* begin = text.c_str() + i;
            char* end = nullptr;
            const double value = std::strtod(begin, &end);
            if (end == begin)
                return false;

            *out_value = value;
            *pos = (size_t)(end - text.c_str());
            return true;
        }

        static bool canva_parse_json_int_field(const std::string& object, const char* key, int* out_value)
        {
            const std::string needle = std::string("\"") + key + "\":";
            size_t pos = object.find(needle);
            if (pos == std::string::npos)
                return false;
            pos += needle.size();

            double value = 0.0;
            if (!canva_parse_json_number(object, &pos, &value))
                return false;

            *out_value = (int)std::lround(value);
            return true;
        }

        static bool canva_parse_json_position_field(const std::string& object, float* out_x, float* out_y)
        {
            const std::string needle = "\"position\":";
            size_t pos = object.find(needle);
            if (pos == std::string::npos)
                return false;
            pos += needle.size();

            while (pos < object.size() && std::isspace((unsigned char)object[pos]))
                ++pos;
            if (pos >= object.size() || object[pos] != '[')
                return false;
            ++pos;

            double x = 0.0;
            double y = 0.0;
            if (!canva_parse_json_number(object, &pos, &x))
                return false;
            while (pos < object.size() && (std::isspace((unsigned char)object[pos]) || object[pos] == ','))
                ++pos;
            if (!canva_parse_json_number(object, &pos, &y))
                return false;

            *out_x = (float)x;
            *out_y = (float)y;
            return true;
        }

        static bool canva_load_sketch2rig_joints(
            const std::filesystem::path& json_path,
            std::vector<Sketch2RigJoint2d>* out_joints,
            std::string* out_error)
        {
            if (!out_joints)
                return false;
            out_joints->clear();

            std::ifstream in(json_path);
            if (!in)
            {
                if (out_error)
                    *out_error = "Failed to open " + json_path.string();
                return false;
            }

            std::ostringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();

            const size_t joints_key = text.find("\"joints\"");
            if (joints_key == std::string::npos)
            {
                if (out_error)
                    *out_error = "Missing joints array in " + json_path.string();
                return false;
            }

            size_t array_pos = text.find('[', joints_key);
            if (array_pos == std::string::npos)
            {
                if (out_error)
                    *out_error = "Malformed joints array in " + json_path.string();
                return false;
            }

            std::vector<Sketch2RigJoint2d> parsed;
            size_t pos = array_pos + 1;
            while (pos < text.size())
            {
                const size_t object_begin = text.find('{', pos);
                const size_t array_end = text.find(']', pos);
                if (array_end != std::string::npos && (object_begin == std::string::npos || array_end < object_begin))
                    break;
                if (object_begin == std::string::npos)
                    break;

                int depth = 0;
                size_t object_end = std::string::npos;
                for (size_t i = object_begin; i < text.size(); ++i)
                {
                    if (text[i] == '{')
                        ++depth;
                    else if (text[i] == '}')
                    {
                        --depth;
                        if (depth == 0)
                        {
                            object_end = i;
                            break;
                        }
                    }
                }
                if (object_end == std::string::npos)
                    break;

                const std::string object = text.substr(object_begin, object_end - object_begin + 1);
                Sketch2RigJoint2d joint;
                if (canva_parse_json_int_field(object, "index", &joint.index)
                    && canva_parse_json_int_field(object, "parent", &joint.parent)
                    && canva_parse_json_position_field(object, &joint.x, &joint.y))
                {
                    joint.valid = true;
                    parsed.push_back(joint);
                }

                pos = object_end + 1;
            }

            if (parsed.empty())
            {
                if (out_error)
                    *out_error = "No valid joints in " + json_path.string();
                return false;
            }

            int max_index = -1;
            for (const auto& joint : parsed)
                max_index = std::max(max_index, joint.index);
            if (max_index < 0 || max_index > 4096)
            {
                if (out_error)
                    *out_error = "Invalid joint indices in " + json_path.string();
                return false;
            }

            out_joints->assign((size_t)max_index + 1u, Sketch2RigJoint2d{});
            for (const auto& joint : parsed)
            {
                if (joint.index >= 0 && joint.index < (int)out_joints->size())
                    (*out_joints)[(size_t)joint.index] = joint;
            }

            return true;
        }

        static void canva_project_sketch2rig_joints(std::vector<Sketch2RigJoint2d>& joints, int w, int h)
        {
            float min_x = std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_x = -std::numeric_limits<float>::max();
            float max_y = -std::numeric_limits<float>::max();

            for (const auto& joint : joints)
            {
                if (!joint.valid)
                    continue;
                min_x = std::min(min_x, joint.x);
                min_y = std::min(min_y, joint.y);
                max_x = std::max(max_x, joint.x);
                max_y = std::max(max_y, joint.y);
            }

            const bool looks_normalized =
                min_x >= -0.25f && min_y >= -0.25f && max_x <= 1.25f && max_y <= 1.25f;

            if (looks_normalized)
            {
                for (auto& joint : joints)
                {
                    if (!joint.valid)
                        continue;
                    joint.x = std::clamp(joint.x, 0.0f, 1.0f) * (float)(w - 1);
                    joint.y = std::clamp(joint.y, 0.0f, 1.0f) * (float)(h - 1);
                }
                return;
            }

            const float dx = std::max(1e-6f, max_x - min_x);
            const float dy = std::max(1e-6f, max_y - min_y);
            const float margin = 0.10f * (float)std::min(w, h);
            const float scale = std::min(((float)w - 2.0f * margin) / dx, ((float)h - 2.0f * margin) / dy);
            const float used_w = dx * scale;
            const float used_h = dy * scale;
            const float ox = ((float)w - used_w) * 0.5f;
            const float oy = ((float)h - used_h) * 0.5f;

            for (auto& joint : joints)
            {
                if (!joint.valid)
                    continue;
                joint.x = ox + (joint.x - min_x) * scale;
                joint.y = oy + (joint.y - min_y) * scale;
            }
        }

        static bool canva_add_sketch2rig_layer(
            CanvaState& canva,
            const std::filesystem::path& json_path,
            const std::filesystem::path& model_input_path,
            bool enabled,
            std::string* out_error)
        {
            std::vector<Sketch2RigJoint2d> joints;
            if (!canva_load_sketch2rig_joints(json_path, &joints, out_error))
                return false;

            canva_project_sketch2rig_joints(joints, canva.width, canva.height);

            CanvaPreprocessTransform preprocess_transform;
            std::string transform_error;
            const bool have_preprocess_transform =
                canva_load_preprocess_transform(model_input_path, &preprocess_transform, &transform_error);
            if (!have_preprocess_transform)
            {
                if (out_error)
                    *out_error = transform_error.empty() ? "Missing preprocessing transform" : transform_error;
                return false;
            }

            CanvaLayer layer = canva_make_layer(
                canva.width,
                canva.height,
                canva.next_layer_id++,
                "Sketch2Rig rig");
            layer.enabled = enabled;

            const float inv_scale = 1.0f / std::max(1e-6f, preprocess_transform.scale);
            layer.scale = inv_scale;
            layer.offset = ImVec2(
                preprocess_transform.src_center.x - preprocess_transform.dst_center.x * inv_scale,
                preprocess_transform.src_center.y - preprocess_transform.dst_center.y * inv_scale);

            const ImVec4 bone_col = ImVec4(0.05f, 0.30f, 1.00f, 0.92f);
            const ImVec4 joint_col = ImVec4(1.00f, 0.20f, 0.10f, 0.96f);
            const ImVec4 root_col = ImVec4(0.10f, 0.80f, 0.35f, 0.98f);

            for (const auto& joint : joints)
            {
                if (!joint.valid || joint.parent < 0 || joint.parent >= (int)joints.size())
                    continue;
                const auto& parent = joints[(size_t)joint.parent];
                if (!parent.valid)
                    continue;
                canva_draw_segment(layer, canva.width, canva.height,
                    ImVec2(parent.x, parent.y), ImVec2(joint.x, joint.y), 2.5f, bone_col);
            }

            for (const auto& joint : joints)
            {
                if (!joint.valid)
                    continue;
                canva_stamp_circle(layer, canva.width, canva.height,
                    joint.x, joint.y, joint.parent < 0 ? 6.5f : 5.0f, joint.parent < 0 ? root_col : joint_col);
            }

            canva.layers.insert(canva.layers.begin(), std::move(layer));
            canva.active_layer = 0;
            canva_cancel_interactions(canva);
            return true;
        }

        static void canva_sample_layer_bilinear_rgba(
            const CanvaLayer& layer,
            int w,
            int h,
            float u,
            float v,
            float& out_r,
            float& out_g,
            float& out_b,
            float& out_a)
        {
            out_r = out_g = out_b = out_a = 0.0f;
            if (layer.pixels.empty() || w <= 0 || h <= 0)
                return;

            const float max_u = (float)(w - 1);
            const float max_v = (float)(h - 1);
            const float cu = std::clamp(u, 0.0f, max_u);
            const float cv = std::clamp(v, 0.0f, max_v);

            const int x0 = (int)std::floor(cu);
            const int y0 = (int)std::floor(cv);
            const int x1 = std::min(x0 + 1, w - 1);
            const int y1 = std::min(y0 + 1, h - 1);

            const float tx = cu - (float)x0;
            const float ty = cv - (float)y0;

            auto px = [&](int x, int y, int c) -> float
            {
                const size_t idx = (((size_t)y * (size_t)w) + (size_t)x) * 4u + (size_t)c;
                return (float)layer.pixels[idx] / 255.0f;
            };

            const float c00_r = px(x0, y0, 0), c00_g = px(x0, y0, 1), c00_b = px(x0, y0, 2), c00_a = px(x0, y0, 3);
            const float c10_r = px(x1, y0, 0), c10_g = px(x1, y0, 1), c10_b = px(x1, y0, 2), c10_a = px(x1, y0, 3);
            const float c01_r = px(x0, y1, 0), c01_g = px(x0, y1, 1), c01_b = px(x0, y1, 2), c01_a = px(x0, y1, 3);
            const float c11_r = px(x1, y1, 0), c11_g = px(x1, y1, 1), c11_b = px(x1, y1, 2), c11_a = px(x1, y1, 3);

            const float r0 = c00_r + (c10_r - c00_r) * tx;
            const float g0 = c00_g + (c10_g - c00_g) * tx;
            const float b0 = c00_b + (c10_b - c00_b) * tx;
            const float a0 = c00_a + (c10_a - c00_a) * tx;

            const float r1 = c01_r + (c11_r - c01_r) * tx;
            const float g1 = c01_g + (c11_g - c01_g) * tx;
            const float b1 = c01_b + (c11_b - c01_b) * tx;
            const float a1 = c01_a + (c11_a - c01_a) * tx;

            out_r = r0 + (r1 - r0) * ty;
            out_g = g0 + (g1 - g0) * ty;
            out_b = b0 + (b1 - b0) * ty;
            out_a = a0 + (a1 - a0) * ty;
        }

        static bool canva_composite_visible_layers_rgba(
            const CanvaState& canva,
            std::vector<unsigned char>* out_rgba,
            std::string* out_error,
            bool sketch_layers_only = false)
        {
            const int w = canva.width;
            const int h = canva.height;
            if (w <= 0 || h <= 0)
            {
                if (out_error)
                    *out_error = "Invalid canva size";
                return false;
            }

            if (!out_rgba)
                return false;
            out_rgba->assign((size_t)w * (size_t)h * 4u, 255u); // opaque white background

            // Composite bottom->top to match rendering order.
            for (int li = (int)canva.layers.size() - 1; li >= 0; --li)
            {
                const CanvaLayer& layer = canva.layers[(size_t)li];
                if (!layer.enabled || layer.pixels.empty())
                    continue;
                if (sketch_layers_only && layer.kind != CanvaLayerKind::Generic)
                    continue;

                const float s = std::max(1e-6f, layer.scale);
                const float min_x = layer.offset.x;
                const float min_y = layer.offset.y;
                const float max_x = layer.offset.x + (float)w * s;
                const float max_y = layer.offset.y + (float)h * s;

                // Only pixels whose centers land inside the layer rect contribute.
                int x0 = (int)std::ceil(min_x - 0.5f);
                int y0 = (int)std::ceil(min_y - 0.5f);
                int x1_excl = (int)std::ceil(max_x - 0.5f);
                int y1_excl = (int)std::ceil(max_y - 0.5f);

                x0 = std::clamp(x0, 0, w);
                y0 = std::clamp(y0, 0, h);
                x1_excl = std::clamp(x1_excl, 0, w);
                y1_excl = std::clamp(y1_excl, 0, h);

                for (int y = y0; y < y1_excl; ++y)
                {
                    const float cy = (float)y + 0.5f;
                    const float v = (cy - layer.offset.y) / s - 0.5f;

                    for (int x = x0; x < x1_excl; ++x)
                    {
                        const float cx = (float)x + 0.5f;
                        const float u = (cx - layer.offset.x) / s - 0.5f;

                        float sr = 0.0f, sg = 0.0f, sb = 0.0f, sa = 0.0f;
                        canva_sample_layer_bilinear_rgba(layer, w, h, u, v, sr, sg, sb, sa);
                        sa = std::clamp(sa, 0.0f, 1.0f);
                        if (sa <= 0.0f)
                            continue;

                        const size_t idx = (((size_t)y * (size_t)w) + (size_t)x) * 4u;
                        const float dr = (float)(*out_rgba)[idx + 0u] / 255.0f;
                        const float dg = (float)(*out_rgba)[idx + 1u] / 255.0f;
                        const float db = (float)(*out_rgba)[idx + 2u] / 255.0f;

                        const float inv = 1.0f - sa;
                        const float or_ = sr * sa + dr * inv;
                        const float og_ = sg * sa + dg * inv;
                        const float ob_ = sb * sa + db * inv;

                        (*out_rgba)[idx + 0u] = (unsigned char)std::clamp<int>((int)(or_ * 255.0f + 0.5f), 0, 255);
                        (*out_rgba)[idx + 1u] = (unsigned char)std::clamp<int>((int)(og_ * 255.0f + 0.5f), 0, 255);
                        (*out_rgba)[idx + 2u] = (unsigned char)std::clamp<int>((int)(ob_ * 255.0f + 0.5f), 0, 255);
                        (*out_rgba)[idx + 3u] = 255u;
                    }
                }
            }
            return true;
        }

        static bool canva_next_render_path(
            const char* tag_suffix,
            const char* extension,
            std::filesystem::path* out_path,
            std::string* out_error)
        {
            std::error_code ec;
            const std::filesystem::path renders_dir = std::filesystem::path("renders");
            std::filesystem::create_directories(renders_dir, ec);
            if (ec)
            {
                if (out_error)
                    *out_error = "Failed to create renders directory: " + ec.message();
                return false;
            }

            const std::string tag = (tag_suffix && tag_suffix[0] != '\0')
                ? (std::string("_") + tag_suffix)
                : std::string();
            const std::filesystem::path out = renders_dir / ("canva" + tag + extension);

            if (out.empty())
            {
                if (out_error)
                    *out_error = "Failed to allocate render output path";
                return false;
            }
            if (out_path)
                *out_path = out;
            return true;
        }

        static bool canva_save_visible_layers_png(const CanvaState& canva, std::filesystem::path* out_path, std::string* out_error)
        {
            std::vector<unsigned char> out_rgba;
            if (!canva_composite_visible_layers_rgba(canva, &out_rgba, out_error))
                return false;

            std::filesystem::path out;
            if (!canva_next_render_path("", ".png", &out, out_error))
                return false;

            const int w = canva.width;
            const int h = canva.height;
            const int ok = stbi_write_png(out.string().c_str(), w, h, 4, out_rgba.data(), w * 4);
            if (ok == 0)
            {
                if (out_error)
                    *out_error = std::string("Failed to write PNG: ") + out.string();
                return false;
            }

            if (out_path)
                *out_path = out;
            return true;
        }

        static bool canva_save_model_input_png(
            const CanvaState& canva,
            std::filesystem::path* out_path,
            std::string* out_error)
        {
            const int w = canva.width;
            const int h = canva.height;

            std::vector<unsigned char> rgba;
            if (!canva_composite_visible_layers_rgba(canva, &rgba, out_error))
                return false;

            return save_normalized_model_input_png(w, h, rgba, out_path, out_error);
        }

        static bool canva_preprocess_metadata_exists(const std::filesystem::path& model_input_path)
        {
            if (model_input_path.empty())
                return false;
            std::error_code ec;
            return std::filesystem::exists(model_input_path.string() + ".preprocess.json", ec);
        }

        static std::filesystem::path canva_cloth_render_alignment_input(
            const AppState& state,
            const CanvaState& canva)
        {
            std::vector<unsigned char> sketch_rgba;
            std::string ignored_error;
            if (canva_composite_visible_layers_rgba(canva, &sketch_rgba, &ignored_error, true))
            {
                std::filesystem::path current_input;
                if (save_normalized_model_input_png(
                        canva.width,
                        canva.height,
                        sketch_rgba,
                        &current_input,
                        &ignored_error))
                    return current_input;
            }

            const std::filesystem::path candidates[] = {
                canva.extract_pose_model_input_path,
                std::filesystem::path(state.sketch2rig.last_sketch_path),
                std::filesystem::path(state.sketch2uv.last_sketch_path),
                canva.generate_cloth_model_input_path,
            };
            for (const auto& candidate : candidates)
            {
                if (canva_preprocess_metadata_exists(candidate))
                    return candidate;
            }
            return {};
        }

        static bool canva_queue_render_cloth(
            AppState& state,
            CanvaState& canva,
            bool fine,
            std::string* out_error)
        {
            if (state.selection.garment_index < 0
                || state.selection.garment_index >= (int)state.selection.garments.size())
            {
                if (out_error)
                    *out_error = "select a cloth mesh first";
                return false;
            }
            if (state.controls.simulating)
            {
                if (out_error)
                    *out_error = "wait for cloth simulation to finish";
                return false;
            }

            canva.render_cloth_active = true;
            canva.render_cloth_fine = fine;
            canva.render_cloth_model_input_path = canva_cloth_render_alignment_input(state, canva);
            canva.render_cloth_status = fine ? "Refining" : "Rendering";

            state.show_garment_mesh = true;
            state.cloth_render_fine = fine;
            state.cloth_render_use_current_camera = true;
            state.cloth_render_stroke_reference_path = canva.render_cloth_model_input_path.string();
            state.cloth_render_target_detail_size =
                std::clamp(state.cloth_render_target_detail_size, 0.0001f, 0.25f);
            state.cloth_render_path.clear();
            canva.render_cloth_request_id = ++state.cloth_render_request_id;

            console_add_line(state, fine
                ? "Canva: queued fine cloth render"
                : "Canva: queued cloth render");
            return true;
        }

        static bool canva_queue_extract_pose(
            AppState& state,
            CanvaState& canva,
            bool generate_cloth_after,
            bool hide_rig_layer,
            std::string* out_error)
        {
            std::filesystem::path sketch_path;
            std::string err;
            if (!canva_save_model_input_png(canva, &sketch_path, &err))
            {
                if (out_error)
                    *out_error = err.empty() ? "preprocessing failed" : err;
                return false;
            }

            const std::filesystem::path rig_output_path =
                sketch_path.parent_path() / "sketch_rig.json";
            const std::filesystem::path betas_output_path =
                sketch_path.parent_path() / "sketch_betas.json";
            const std::uint64_t rig_request_id = state.sketch2rig.next_request_id++;
            const std::uint64_t betas_request_id = state.sketch2betas.next_request_id++;

            std::string rig_request_err;
            std::string betas_request_err;
            const bool rig_ok = state.sketch2rig.service->request_inference(
                rig_request_id,
                sketch_path,
                rig_output_path,
                &rig_request_err);
            const bool betas_ok = state.sketch2betas.service->request_inference(
                betas_request_id,
                sketch_path,
                betas_output_path,
                &betas_request_err);

            if (!rig_ok || !betas_ok)
            {
                if (out_error)
                {
                    *out_error =
                        (!rig_ok ? (rig_request_err.empty() ? "rig request failed" : rig_request_err) : "")
                        + std::string(!rig_ok && !betas_ok ? "; " : "")
                        + (!betas_ok ? (betas_request_err.empty() ? "betas request failed" : betas_request_err) : "");
                }
                return false;
            }

            state.sketch2rig.last_sketch_path = sketch_path.string();
            state.sketch2betas.last_sketch_path = sketch_path.string();
            state.sketch2rig.status = ModelInferenceStatus::Busy;
            state.sketch2rig.active_request_id = rig_request_id;
            state.sketch2rig.message = "Processing sketch with Sketch2Rig...";
            state.sketch2betas.status = ModelInferenceStatus::Busy;
            state.sketch2betas.active_request_id = betas_request_id;
            state.sketch2betas.message = "Processing sketch with Sketch2Betas...";

            canva.extract_pose_active = true;
            canva.extract_pose_rig_done = false;
            canva.extract_pose_betas_done = false;
            canva.extract_pose_hide_rig_layer = hide_rig_layer;
            canva.extract_pose_generate_cloth_after = generate_cloth_after;
            canva.extract_pose_rig_request_id = rig_request_id;
            canva.extract_pose_betas_request_id = betas_request_id;
            canva.extract_pose_model_input_path = sketch_path;
            canva.extract_pose_rig_path = rig_output_path;
            canva.extract_pose_betas_path = betas_output_path;
            if (generate_cloth_after)
            {
                canva.generate_cloth_model_input_path = sketch_path;
                canva.generate_cloth_status = "Inference";
                canva.generate_cloth_failed_step = -1;
            }
            state.scene_view_snap_to_target_request_id++;

            console_add_line(state, std::string("Canva: Extract Pose queued from sketch: ") + sketch_path.string());
            return true;
        }

        static bool canva_queue_generate_displacement(
            AppState& state,
            CanvaState& canva,
            std::string* out_error)
        {
            if (!state.scene_view_target_valid
                || state.scene_view_snapshot_completed_id == 0
                || state.scene_view_snapshot_path.empty())
            {
                if (out_error)
                    *out_error = "predicted camera view is not available";
                return false;
            }

            std::filesystem::path manual_uv_path;
            if (canva_manual_uv_source_path(state, &manual_uv_path))
            {
                canva.generate_displacement_active = true;
                canva.generate_displacement_waiting_uv = false;
                canva.generate_displacement_waiting_render = true;
                canva.generate_displacement_uv_request_id = 0;
                canva.generate_displacement_render_request_id = ++state.canva_displacement_request_id;
                canva.generate_displacement_model_input_path.clear();
                canva.generate_displacement_uv_json_path = manual_uv_path;
                canva.generate_displacement_status = "Projecting displacement";

                state.canva_displacement_uv_json_path = manual_uv_path.string();
                state.canva_displacement_model_input_path.clear();
                state.canva_displacement_path.clear();

                console_add_line(state, std::string("Canva: Generate Displacement using loaded UV PNG: ")
                    + manual_uv_path.string());
                return true;
            }

            if (state.sketch2uv.service == nullptr || state.sketch2uv.status != ModelInferenceStatus::Ready)
            {
                if (out_error)
                    *out_error = "Sketch2UV is not ready";
                return false;
            }

            std::filesystem::path sketch_path;
            std::string err;
            if (!canva_save_model_input_png(canva, &sketch_path, &err))
            {
                if (out_error)
                    *out_error = err.empty() ? "preprocessing failed" : err;
                return false;
            }

            const std::filesystem::path output_path =
                sketch_path.parent_path() / "sketch_uv_displacement.json";
            const std::uint64_t request_id = state.sketch2uv.next_request_id++;

            std::string request_err;
            if (!state.sketch2uv.service->request_inference(request_id, sketch_path, output_path, &request_err))
            {
                if (out_error)
                    *out_error = request_err.empty() ? "failed to queue Sketch2UV" : request_err;
                return false;
            }

            state.sketch2uv.last_sketch_path = sketch_path.string();
            state.sketch2uv.status = ModelInferenceStatus::Busy;
            state.sketch2uv.active_request_id = request_id;
            state.sketch2uv.message = "Processing sketch UVs for displacement...";

            canva.generate_displacement_active = true;
            canva.generate_displacement_waiting_uv = true;
            canva.generate_displacement_waiting_render = false;
            canva.generate_displacement_uv_request_id = request_id;
            canva.generate_displacement_render_request_id = 0;
            canva.generate_displacement_model_input_path = sketch_path;
            canva.generate_displacement_uv_json_path = output_path;
            canva.generate_displacement_status = "Generating UV";

            console_add_line(state, std::string("Canva: Generate Displacement queued Sketch2UV from sketch: ")
                + sketch_path.string());
            return true;
        }

        static bool canva_queue_generate_iso_displacement(
            AppState& state,
            CanvaState& canva,
            std::string* out_error)
        {
            if (!state.scene_view_target_valid
                || state.scene_view_snapshot_completed_id == 0
                || state.scene_view_snapshot_path.empty())
            {
                if (out_error)
                    *out_error = "predicted camera view is not available";
                return false;
            }

            std::filesystem::path manual_iso_path;
            if (canva_manual_iso_source_path(state, &manual_iso_path))
            {
                canva.generate_iso_displacement_active = true;
                canva.generate_iso_displacement_waiting_iso = false;
                canva.generate_iso_displacement_waiting_render = true;
                canva.generate_iso_displacement_iso_request_id = 0;
                canva.generate_iso_displacement_render_request_id =
                    ++state.canva_iso_displacement_request_id;
                canva.generate_iso_displacement_model_input_path.clear();
                canva.generate_iso_displacement_npz_path = manual_iso_path;
                canva.generate_iso_displacement_status = "Projecting ISO displacement";

                state.canva_iso_displacement_npz_path = manual_iso_path.string();
                state.canva_iso_displacement_model_input_path.clear();
                state.canva_iso_displacement_path.clear();

                console_add_line(
                    state,
                    "Canva: Create ISO Displacement Map using loaded ISO NPZ: "
                        + manual_iso_path.string());
                return true;
            }

            if (state.sketch2iso.service == nullptr
                || state.sketch2iso.status != ModelInferenceStatus::Ready)
            {
                if (out_error)
                    *out_error = "Sketch2ISO is not ready";
                return false;
            }

            std::filesystem::path sketch_path;
            std::string error;
            if (!canva_save_model_input_png(canva, &sketch_path, &error))
            {
                if (out_error)
                    *out_error = error.empty() ? "preprocessing failed" : error;
                return false;
            }

            const std::filesystem::path output_path =
                sketch_path.parent_path() / "sketch_iso_displacement.npz";
            const std::uint64_t request_id = state.sketch2iso.next_request_id++;
            std::string request_error;
            if (!state.sketch2iso.service->request_inference(
                    request_id,
                    sketch_path,
                    output_path,
                    &request_error))
            {
                if (out_error)
                    *out_error = request_error.empty()
                        ? "failed to queue Sketch2ISO"
                        : request_error;
                return false;
            }

            state.sketch2iso.last_sketch_path = sketch_path.string();
            state.sketch2iso.status = ModelInferenceStatus::Busy;
            state.sketch2iso.active_request_id = request_id;
            state.sketch2iso.message = "Processing sketch ISO values for displacement...";

            canva.generate_iso_displacement_active = true;
            canva.generate_iso_displacement_waiting_iso = true;
            canva.generate_iso_displacement_waiting_render = false;
            canva.generate_iso_displacement_iso_request_id = request_id;
            canva.generate_iso_displacement_render_request_id = 0;
            canva.generate_iso_displacement_model_input_path = sketch_path;
            canva.generate_iso_displacement_npz_path = output_path;
            canva.generate_iso_displacement_status = "Generating ISO";

            console_add_line(
                state,
                "Canva: Create ISO Displacement Map queued Sketch2ISO from sketch: "
                    + sketch_path.string());
            return true;
        }

        static bool canva_queue_cloth_fitting_uv(
            AppState& state,
            CanvaState& canva,
            const CanvaLayer& source_layer,
            std::string* out_error)
        {
            if (state.sketch2uv.service == nullptr || state.sketch2uv.status != ModelInferenceStatus::Ready)
            {
                if (out_error)
                    *out_error = "Sketch2UV is not ready";
                return false;
            }

            std::filesystem::path sketch_path;
            std::string err;
            if (!canva_save_model_input_png(canva, &sketch_path, &err))
            {
                if (out_error)
                    *out_error = err.empty() ? "preprocessing failed" : err;
                return false;
            }

            const std::filesystem::path output_path =
                sketch_path.parent_path() / "sketch_uv_cloth_fitting.json";
            const std::uint64_t request_id = state.sketch2uv.next_request_id++;

            std::string request_err;
            if (!state.sketch2uv.service->request_inference(request_id, sketch_path, output_path, &request_err))
            {
                if (out_error)
                    *out_error = request_err.empty() ? "failed to queue Sketch2UV" : request_err;
                return false;
            }

            state.sketch2uv.last_sketch_path = sketch_path.string();
            state.sketch2uv.status = ModelInferenceStatus::Busy;
            state.sketch2uv.active_request_id = request_id;
            state.sketch2uv.message = "Processing sketch UVs for cloth fitting...";

            canva.cloth_fitting_active = true;
            canva.cloth_fitting_waiting_uv = true;
            canva.cloth_fitting_uv_request_id = request_id;
            canva.cloth_fitting_source_layer_id = source_layer.id;
            canva.cloth_fitting_status = "Generating UV";

            console_add_line(state, std::string("Canva: cloth Fitting queued Sketch2UV from sketch: ")
                + sketch_path.string());
            return true;
        }

        static bool canva_queue_generate_cloth_current_pose(
            AppState& state,
            CanvaState& canva,
            std::string* out_error)
        {
            if (state.selection.garment_index < 0
                || state.selection.garment_index >= (int)state.selection.garments.size())
            {
                if (out_error)
                    *out_error = "select a cloth mesh first";
                return false;
            }
            if (state.controls.simulating)
            {
                if (out_error)
                    *out_error = "cloth simulation is already running";
                return false;
            }

            std::filesystem::path sketch_path;
            std::string err;
            if (!canva_save_model_input_png(canva, &sketch_path, &err))
            {
                if (out_error)
                    *out_error = err.empty() ? "preprocessing failed" : err;
                return false;
            }

            canva.generate_cloth_active = true;
            canva.generate_cloth_waiting_pose = false;
            canva.generate_cloth_waiting_simulation = true;
            canva.generate_cloth_waiting_render = false;
            canva.generate_cloth_model_input_path = sketch_path;
            canva.generate_cloth_pose_name.clear();
            canva.generate_cloth_status = "Simulating";
            canva.generate_cloth_failed_step = -1;

            state.show_smpl = true;
            state.show_smpl_character = true;
            state.show_garment_mesh = true;
            state.controls.current_frame = 0;
            state.controls.playing = false;
            state.controls.simulating = true;
            state.cloth_simulate_request_id++;

            console_add_line(state, std::string("Canva: Generate Cloth queued current-pose XPBD simulation from sketch: ")
                + sketch_path.string());
            return true;
        }

        static std::string canva_shell_quote(const std::string& value)
        {
            std::string out = "'";
            for (char c : value)
            {
                if (c == '\'')
                    out += "'\\''";
                else
                    out += c;
            }
            out += "'";
            return out;
        }

        static std::string canva_python_command_prefix()
        {
            if (const char* env_python = std::getenv("POSEWEAVER_PYTHON"); env_python && env_python[0] != '\0')
                return canva_shell_quote(env_python);

            const char* env_conda_env = std::getenv("POSEWEAVER_CONDA_ENV");
            const char* env_conda_exe = std::getenv("CONDA_EXE");
            const std::string conda_env = (env_conda_env && env_conda_env[0] != '\0') ? env_conda_env : "dev";
            std::string conda_exe = (env_conda_exe && env_conda_exe[0] != '\0') ? env_conda_exe : "conda";
            if (!(env_conda_exe && env_conda_exe[0] != '\0'))
            {
                if (const char* home = std::getenv("HOME"); home && home[0] != '\0')
                {
                    const std::filesystem::path home_path(home);
                    const std::filesystem::path candidates[] = {
                        home_path / "miniconda3" / "bin" / "conda",
                        home_path / "miniconda3" / "condabin" / "conda",
                        home_path / "anaconda3" / "bin" / "conda",
                        home_path / "anaconda3" / "condabin" / "conda",
                        home_path / "miniforge3" / "bin" / "conda",
                        home_path / "mambaforge" / "bin" / "conda",
                    };
                    for (const auto& candidate : candidates)
                    {
                        if (std::filesystem::exists(candidate))
                        {
                            conda_exe = candidate.string();
                            break;
                        }
                    }
                }
            }
            return canva_shell_quote(conda_exe) + " run --no-capture-output -n "
                + canva_shell_quote(conda_env) + " python";
        }

        static bool canva_current_predicted_pose_path(
            const AppState& state,
            std::filesystem::path* out_pose_path,
            std::string* out_error)
        {
            if (state.selection.pose_index < 0 || state.selection.pose_index >= (int)state.selection.poses.size())
            {
                if (out_error)
                    *out_error = "no pose is selected";
                return false;
            }

            const std::string pose_name = state.selection.poses[(size_t)state.selection.pose_index];
            if (pose_name.empty())
            {
                if (out_error)
                    *out_error = "selected pose is empty";
                return false;
            }

            const std::filesystem::path pose_path = assets_dir() / "poses" / pose_name;
            if (!std::filesystem::exists(pose_path))
            {
                if (out_error)
                    *out_error = "selected pose file does not exist: " + pose_path.string();
                return false;
            }

            const bool predicted_context =
                !state.sketch2rig.last_output_path.empty()
                || pose_name == "sketch_pose.npz"
                || pose_name == "rig_optimized_pose.npz";
            if (!predicted_context)
            {
                if (out_error)
                    *out_error = "run Extract Pose first";
                return false;
            }

            if (out_pose_path)
                *out_pose_path = pose_path;
            return true;
        }

        static std::filesystem::path canva_smpl_model_path_for_state(const AppState& state)
        {
            const char* dir = "male";
            if (state.selection.gender == SmplType::Female)
                dir = "female";
            else if (state.selection.gender == SmplType::Neutral)
                dir = "neutral";
            return assets_dir() / "smplh" / dir / "model.npz";
        }

        static RigOptimizationJobResult canva_run_rig_optimization_process(const std::string& command, std::filesystem::path output_json)
        {
            RigOptimizationJobResult result;
            result.output_json_path = std::move(output_json);
            result.exit_code = std::system(command.c_str());
            result.ok = (result.exit_code == 0 && std::filesystem::exists(result.output_json_path));
            if (!result.ok)
            {
                std::ostringstream oss;
                oss << "optimizer process failed";
                if (result.exit_code != 0)
                    oss << " (exit code " << result.exit_code << ")";
                if (!std::filesystem::exists(result.output_json_path))
                    oss << "; missing output " << result.output_json_path.string();
                result.error = oss.str();
            }
            return result;
        }

        static bool canva_start_rig_optimization(
            AppState& state,
            CanvaState& canva,
            const std::filesystem::path& uv_json_path,
            const std::filesystem::path& model_input_path,
            std::string* out_error)
        {
            std::filesystem::path pose_path;
            if (!canva_current_predicted_pose_path(state, &pose_path, out_error))
                return false;

            const std::filesystem::path smpl_model_path = canva_smpl_model_path_for_state(state);
            if (!std::filesystem::exists(smpl_model_path))
            {
                if (out_error)
                    *out_error = "missing SMPL model: " + smpl_model_path.string();
                return false;
            }
            if (uv_json_path.empty() || !std::filesystem::exists(uv_json_path))
            {
                if (out_error)
                    *out_error = "missing Sketch2UV result";
                return false;
            }
            if (!state.scene_view_target_valid)
            {
                if (out_error)
                    *out_error = "predicted scene-view camera is not available";
                return false;
            }

            std::error_code ec;
            std::filesystem::create_directories(assets_dir() / "poses", ec);
            if (ec)
            {
                if (out_error)
                    *out_error = "failed to create pose output directory: " + ec.message();
                return false;
            }
            std::filesystem::create_directories("renders", ec);

            const std::filesystem::path output_json = assets_dir() / "poses" / "rig_optimized_pose.json";
            const std::filesystem::path output_npz = assets_dir() / "poses" / "rig_optimized_pose.npz";
            const std::filesystem::path script = std::filesystem::path(POSEWEAVER_RIG_OPTIMIZER_WORKER);

            auto append_vec3 = [](std::ostringstream& oss, const char* arg, const std::array<float, 3>& v)
            {
                oss << " " << arg
                    << " " << std::setprecision(9) << v[0]
                    << " " << std::setprecision(9) << v[1]
                    << " " << std::setprecision(9) << v[2];
            };

            std::ostringstream cmd;
            cmd << canva_python_command_prefix()
                << " " << canva_shell_quote(script.string())
                << " --pose-npz " << canva_shell_quote(pose_path.string())
                << " --uv-json " << canva_shell_quote(uv_json_path.string())
                << " --smpl-model " << canva_shell_quote(smpl_model_path.string())
                << " --output-json " << canva_shell_quote(output_json.string())
                << " --output-npz " << canva_shell_quote(output_npz.string())
                << " --debug-dir " << canva_shell_quote(std::filesystem::path("renders").string())
                << " --iterations " << std::clamp(canva.rig_optimization_iterations, 1, 1000)
                << " --render-size " << std::clamp(canva.rig_optimization_render_size, 64, 384)
                << " --lr " << std::clamp(canva.rig_optimization_lr, 0.0001f, 1.0f)
                << " --pose-reg " << std::clamp(canva.rig_optimization_pose_reg, 0.0f, 1.0f)
                << " --beta-reg " << std::clamp(canva.rig_optimization_beta_reg, 0.0f, 1.0f);
            append_vec3(cmd, "--camera-eye", state.scene_view_target_eye);
            append_vec3(cmd, "--camera-direction", state.scene_view_target_direction);
            append_vec3(cmd, "--camera-up", state.scene_view_target_up);
            if (state.scene_view_raw_rig_root_valid)
                append_vec3(cmd, "--raw-rig-root", state.scene_view_raw_rig_root);
            if (state.scene_view_raw_rig_height_valid)
                cmd << " --raw-rig-height " << std::setprecision(9) << state.scene_view_raw_rig_height;

            canva.rig_optimization_active = true;
            canva.rig_optimization_waiting_uv = false;
            canva.rig_optimization_job_running = true;
            canva.rig_optimization_status = "Optimizing";
            canva.rig_optimization_last_error.clear();
            canva.rig_optimization_model_input_path = model_input_path;
            canva.rig_optimization_uv_json_path = uv_json_path;
            canva.rig_optimization_output_json_path = output_json;
            canva.rig_optimization_future = std::async(
                std::launch::async,
                canva_run_rig_optimization_process,
                cmd.str(),
                output_json);

            console_add_line(state, std::string("Canva: Rig Optimization started using UV mask: ") + uv_json_path.string());
            return true;
        }

        static bool canva_queue_rig_optimization(
            AppState& state,
            CanvaState& canva,
            std::string* out_error)
        {
            std::filesystem::path pose_path;
            if (!canva_current_predicted_pose_path(state, &pose_path, out_error))
                return false;

            std::error_code ec;
            const bool have_uv =
                !state.sketch2uv.last_output_path.empty()
                && std::filesystem::exists(std::filesystem::path(state.sketch2uv.last_output_path), ec);
            if (have_uv)
            {
                std::filesystem::path model_input_path = state.sketch2uv.last_sketch_path;
                if (model_input_path.empty())
                    model_input_path = canva.extract_pose_model_input_path;
                return canva_start_rig_optimization(
                    state,
                    canva,
                    std::filesystem::path(state.sketch2uv.last_output_path),
                    model_input_path,
                    out_error);
            }

            if (state.sketch2uv.service == nullptr || state.sketch2uv.status != ModelInferenceStatus::Ready)
            {
                if (out_error)
                    *out_error = "Sketch2UV is not ready";
                return false;
            }

            std::filesystem::path sketch_path;
            std::string err;
            if (!canva_save_model_input_png(canva, &sketch_path, &err))
            {
                if (out_error)
                    *out_error = err.empty() ? "preprocessing failed" : err;
                return false;
            }

            const std::filesystem::path output_path = sketch_path.parent_path() / "sketch_uv_rig_optimization.json";
            const std::uint64_t request_id = state.sketch2uv.next_request_id++;
            std::string request_err;
            if (!state.sketch2uv.service->request_inference(request_id, sketch_path, output_path, &request_err))
            {
                if (out_error)
                    *out_error = request_err.empty() ? "failed to queue Sketch2UV" : request_err;
                return false;
            }

            state.sketch2uv.last_sketch_path = sketch_path.string();
            state.sketch2uv.status = ModelInferenceStatus::Busy;
            state.sketch2uv.active_request_id = request_id;
            state.sketch2uv.message = "Processing sketch UVs for Rig Optimization...";

            canva.rig_optimization_active = true;
            canva.rig_optimization_waiting_uv = true;
            canva.rig_optimization_job_running = false;
            canva.rig_optimization_uv_request_id = request_id;
            canva.rig_optimization_model_input_path = sketch_path;
            canva.rig_optimization_uv_json_path = output_path;
            canva.rig_optimization_status = "Predicting UV";
            canva.rig_optimization_last_error.clear();
            console_add_line(state, std::string("Canva: Rig Optimization queued Sketch2UV from sketch: ") + sketch_path.string());
            return true;
        }

        static bool canva_apply_rig_optimization_result(
            AppState& state,
            CanvaState& canva,
            const std::filesystem::path& json_path,
            std::string* out_error)
        {
            std::ifstream in(json_path);
            if (!in)
            {
                if (out_error)
                    *out_error = "Failed to open " + json_path.string();
                return false;
            }
            std::ostringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();

            const std::filesystem::path npz_path = canva_parse_string_field(text, "optimized_pose_npz");
            if (npz_path.empty())
            {
                if (out_error)
                    *out_error = "Missing optimized_pose_npz in optimizer result";
                return false;
            }

            std::vector<float> betas;
            if (canva_parse_number_array_field(text, "betas", &betas, nullptr))
            {
                const int n = std::min<int>((int)state.smpl_shape.betas.size(), (int)betas.size());
                for (int i = 0; i < n; ++i)
                    state.smpl_shape.betas[(size_t)i] = betas[(size_t)i];
                for (int i = n; i < (int)state.smpl_shape.betas.size(); ++i)
                    state.smpl_shape.betas[(size_t)i] = 0.0f;
            }

            const std::string gender = canva_parse_string_field(text, "gender");
            if (gender == "male")
                state.selection.gender = SmplType::Male;
            else if (gender == "female")
                state.selection.gender = SmplType::Female;
            else if (gender == "neutral")
                state.selection.gender = SmplType::Neutral;

            double initial_loss = 0.0;
            double final_loss = 0.0;
            (void)canva_parse_number_field(text, "initial", &initial_loss);
            (void)canva_parse_number_field(text, "final", &final_loss);
            canva.rig_optimization_last_initial_loss = (float)initial_loss;
            canva.rig_optimization_last_final_loss = (float)final_loss;

            state.pending_pose_selection = npz_path.filename().string();
            state.asset_refresh_request_id++;
            state.pose_asset_reload_request_id++;
            canva.rig_optimization_last_output = json_path.string();

            state.show_smpl = true;
            state.show_smpl_character = true;
            state.scene_view_snapshot_pose_name = npz_path.filename().string();
            state.scene_view_snapshot_path.clear();
            state.scene_view_snapshot_use_rigopt_camera = false;
            canva.pending_scene_view_snapshot_request_id = ++state.scene_view_snapshot_request_id;
            canva.pending_scene_view_snapshot_model_input_path = canva.rig_optimization_model_input_path;
            state.scene_view_snap_to_target_request_id++;

            const struct
            {
                const char* key;
                const char* label;
                bool enabled;
            } debug_layers[] = {
                { "target_contour_path", "Rig opt target contour", false },
                { "initial_silhouette_path", "Rig opt initial", false },
                { "optimized_silhouette_path", "Rig opt silhouette", false },
                { "contour_loss_overlay_path", "Rig opt loss", true },
            };

            for (const auto& item : debug_layers)
            {
                std::filesystem::path image_path = canva_parse_string_field(text, item.key);
                if (image_path.empty())
                    continue;
                std::string layer_err;
                if (!canva_add_preprocess_aligned_png_layer(
                        canva,
                        image_path.string(),
                        canva.rig_optimization_model_input_path,
                        item.label,
                        CanvaLayerKind::Generic,
                        item.enabled,
                        &layer_err))
                {
                    console_add_line(state, std::string("Canva: Failed to add Rig Optimization debug layer: ")
                        + (layer_err.empty() ? image_path.string() : layer_err));
                }
            }

            console_add_line(state, std::string("Canva: Rig Optimization saved pose NPZ: ") + npz_path.string());
            return true;
        }

        static void canva_ensure_layer_texture(CanvaLayer& layer, int w, int h)
        {
            if (layer.texture_id == 0)
                glGenTextures(1, &layer.texture_id);

            glBindTexture(GL_TEXTURE_2D, layer.texture_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                layer.pixels.empty() ? nullptr : layer.pixels.data());

            glBindTexture(GL_TEXTURE_2D, 0);
            layer.dirty = false;
        }

        static void canva_upload_layer(CanvaLayer& layer, int w, int h)
        {
            if (layer.texture_id == 0)
            {
                canva_ensure_layer_texture(layer, w, h);
                return;
            }

            glBindTexture(GL_TEXTURE_2D, layer.texture_id);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, layer.pixels.data());
            glBindTexture(GL_TEXTURE_2D, 0);
            layer.dirty = false;
        }

        static CanvaLayer canva_make_layer(int w, int h, int id, const std::string& name)
        {
            CanvaLayer layer;
            layer.id = id;
            layer.name = name;
            layer.enabled = true;
            layer.offset = ImVec2(0.0f, 0.0f);
            layer.scale = 1.0f;
            layer.pixels.assign((size_t)w * (size_t)h * 4u, 0u); // transparent
            layer.dirty = true;
            layer.bounds_dirty = true;
            layer.bounds_min_x = 0;
            layer.bounds_min_y = 0;
            layer.bounds_max_x_excl = w;
            layer.bounds_max_y_excl = h;
            canva_ensure_layer_texture(layer, w, h);
            return layer;
        }

        static void canva_delete_layer_texture(CanvaLayer& layer)
        {
            if (layer.texture_id != 0)
            {
                glDeleteTextures(1, &layer.texture_id);
                layer.texture_id = 0;
            }
        }

        static int canva_find_layer_index_by_id(const std::vector<CanvaLayer>& layers, int layer_id)
        {
            for (int i = 0; i < (int)layers.size(); ++i)
            {
                if (layers[(size_t)i].id == layer_id)
                    return i;
            }
            return -1;
        }

        static void canva_prune_undo_for_layer(std::vector<CanvaUndo>& undo, int layer_id)
        {
            undo.erase(
                std::remove_if(undo.begin(), undo.end(), [&](const CanvaUndo& u) { return u.layer_id == layer_id; }),
                undo.end());
        }

        static void canva_push_undo(CanvaState& canva, int layer_id, const std::vector<unsigned char>& pixels, const ImVec2& offset, float scale)
        {
            constexpr int kMaxUndoSteps = 30;
            CanvaUndo u;
            u.layer_id = layer_id;
            u.pixels = pixels;
            u.offset = offset;
            u.scale = scale;
            canva.undo.emplace_back(std::move(u));
            if ((int)canva.undo.size() > kMaxUndoSteps)
                canva.undo.erase(canva.undo.begin(), canva.undo.begin() + ((int)canva.undo.size() - kMaxUndoSteps));
        }

        static bool canva_apply_undo(CanvaState& canva)
        {
            if (canva.undo.empty())
                return false;

            CanvaUndo u = std::move(canva.undo.back());
            canva.undo.pop_back();

            const int idx = canva_find_layer_index_by_id(canva.layers, u.layer_id);
            if (idx < 0)
                return false;

            CanvaLayer& layer = canva.layers[(size_t)idx];
            layer.pixels = std::move(u.pixels);
            layer.offset = u.offset;
            layer.scale = u.scale;
            layer.dirty = true;
            layer.bounds_dirty = true;

            canva.active_layer = idx;
            canva.drawing = false;
            canva.sizing = false;
            canva.transforming = false;
            canva.transform_layer = -1;

            return true;
        }

        static void canva_blend_over(unsigned char* dst_rgba, const ImVec4& src)
        {
            const float sa = std::clamp(src.w, 0.0f, 1.0f);
            const float sr = std::clamp(src.x, 0.0f, 1.0f);
            const float sg = std::clamp(src.y, 0.0f, 1.0f);
            const float sb = std::clamp(src.z, 0.0f, 1.0f);

            const float dr = (float)dst_rgba[0] / 255.0f;
            const float dg = (float)dst_rgba[1] / 255.0f;
            const float db = (float)dst_rgba[2] / 255.0f;
            const float da = (float)dst_rgba[3] / 255.0f;

            const float out_a = sa + da * (1.0f - sa);
            if (out_a <= 1e-6f)
            {
                dst_rgba[0] = dst_rgba[1] = dst_rgba[2] = dst_rgba[3] = 0;
                return;
            }

            const float out_r = (sr * sa + dr * da * (1.0f - sa)) / out_a;
            const float out_g = (sg * sa + dg * da * (1.0f - sa)) / out_a;
            const float out_b = (sb * sa + db * da * (1.0f - sa)) / out_a;

            dst_rgba[0] = (unsigned char)std::clamp<int>((int)(out_r * 255.0f + 0.5f), 0, 255);
            dst_rgba[1] = (unsigned char)std::clamp<int>((int)(out_g * 255.0f + 0.5f), 0, 255);
            dst_rgba[2] = (unsigned char)std::clamp<int>((int)(out_b * 255.0f + 0.5f), 0, 255);
            dst_rgba[3] = (unsigned char)std::clamp<int>((int)(out_a * 255.0f + 0.5f), 0, 255);
        }

        static void canva_stamp_circle(CanvaLayer& layer, int w, int h, float cx, float cy, float radius, const ImVec4& col)
        {
            if (layer.pixels.empty() || w <= 0 || h <= 0)
                return;

            const int r = (int)std::ceil(std::max(0.5f, radius));
            const int x0 = std::max(0, (int)std::floor(cx) - r);
            const int x1 = std::min(w - 1, (int)std::floor(cx) + r);
            const int y0 = std::max(0, (int)std::floor(cy) - r);
            const int y1 = std::min(h - 1, (int)std::floor(cy) + r);

            const float rr = radius * radius;
            for (int y = y0; y <= y1; ++y)
            {
                for (int x = x0; x <= x1; ++x)
                {
                    const float dx = ((float)x + 0.5f) - cx;
                    const float dy = ((float)y + 0.5f) - cy;
                    if (dx * dx + dy * dy <= rr)
                    {
                        unsigned char* px = &layer.pixels[((size_t)y * (size_t)w + (size_t)x) * 4u];
                        canva_blend_over(px, col);
                    }
                }
            }
            layer.dirty = true;
            layer.bounds_dirty = true;
        }

        static void canva_draw_segment(CanvaLayer& layer, int w, int h, ImVec2 a, ImVec2 b, float radius, const ImVec4& col)
        {
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            const float dist = std::sqrt(dx * dx + dy * dy);

            const float step = std::max(0.5f, radius * 0.35f);
            const int n = (dist > 1e-4f) ? (int)std::ceil(dist / step) : 1;

            for (int i = 0; i <= n; ++i)
            {
                const float t = (n > 0) ? ((float)i / (float)n) : 0.0f;
                const float x = a.x + dx * t;
                const float y = a.y + dy * t;
                canva_stamp_circle(layer, w, h, x, y, radius, col);
            }
        }

        static bool canva_layer_has_opaque_pixel_at(const CanvaLayer& layer, int w, int h, const ImVec2& canvas_pos)
        {
            const float s = std::max(1e-6f, layer.scale);
            const ImVec2 p = ImVec2((canvas_pos.x - layer.offset.x) / s, (canvas_pos.y - layer.offset.y) / s);
            const int x = (int)std::floor(p.x);
            const int y = (int)std::floor(p.y);
            if (x < 0 || y < 0 || x >= w || y >= h)
                return false;
            if (layer.pixels.empty())
                return false;
            const size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 4u;
            return layer.pixels[idx + 3u] != 0u;
        }

        static int canva_pick_layer_at(const CanvaState& canva, const ImVec2& canvas_pos)
        {
            // layers are stored top->bottom, so iterate forward for topmost.
            for (int i = 0; i < (int)canva.layers.size(); ++i)
            {
                const CanvaLayer& layer = canva.layers[(size_t)i];
                if (!layer.enabled)
                    continue;
                if (canva_layer_has_opaque_pixel_at(layer, canva.width, canva.height, canvas_pos))
                    return i;
            }
            return -1;
        }

        static void canva_cancel_interactions(CanvaState& canva)
        {
            canva.drawing = false;
            canva.sizing = false;
            canva.transforming = false;
            canva.scaling = false;
            canva.transform_layer = -1;
        }

        static void canva_reset_all(AppState& state, CanvaState& canva, Renderer& renderer)
        {
            for (CanvaLayer& layer : canva.layers)
                canva_delete_layer_texture(layer);

            std::future<RigOptimizationJobResult> optimizer_future =
                std::move(canva.rig_optimization_future);
            const bool optimizer_pending = optimizer_future.valid();

            const std::uint64_t rig_active = state.sketch2rig.active_request_id;
            const std::uint64_t betas_active = state.sketch2betas.active_request_id;
            const std::uint64_t uv_active = state.sketch2uv.active_request_id;
            const std::uint64_t iso_active = state.sketch2iso.active_request_id;

            CanvaState fresh;
            fresh.brush_radius = state.canva_brush_radius;
            fresh.brush_color = ImVec4(
                state.canva_brush_color[0],
                state.canva_brush_color[1],
                state.canva_brush_color[2],
                state.canva_brush_color[3]);
            fresh.last_undo_request_id = state.canva_undo_request_id;
            fresh.last_sketch2rig_import_request_id =
                std::max(state.sketch2rig.completed_request_id, rig_active);
            fresh.last_sketch2betas_import_request_id =
                std::max(state.sketch2betas.completed_request_id, betas_active);
            fresh.last_sketch2uv_import_request_id =
                std::max(state.sketch2uv.completed_request_id, uv_active);
            fresh.last_sketch2iso_import_request_id =
                std::max(state.sketch2iso.completed_request_id, iso_active);
            fresh.ignore_sketch2rig_request_id = rig_active;
            fresh.ignore_sketch2betas_request_id = betas_active;
            fresh.ignore_sketch2uv_request_id = uv_active;
            fresh.ignore_sketch2iso_request_id = iso_active;
            fresh.last_scene_view_snapshot_completed_id = state.scene_view_snapshot_request_id;
            fresh.last_displacement_completed_id = state.canva_displacement_request_id;
            fresh.last_iso_displacement_completed_id = state.canva_iso_displacement_request_id;
            fresh.last_cloth_render_completed_id = state.cloth_render_request_id;
            fresh.rig_optimization_future = std::move(optimizer_future);
            fresh.rig_optimization_job_running = optimizer_pending;
            fresh.rig_optimization_discard_result = optimizer_pending;
            fresh.next_layer_id = 1;
            fresh.layers.emplace_back(
                canva_make_layer(fresh.width, fresh.height, fresh.next_layer_id++, "Layer 1"));
            fresh.active_layer = 0;
            canva = std::move(fresh);

            state.sketch2rig.last_sketch_path.clear();
            state.sketch2rig.last_output_path.clear();
            state.sketch2betas.last_sketch_path.clear();
            state.sketch2betas.last_output_path.clear();
            state.sketch2uv.last_sketch_path.clear();
            state.sketch2uv.last_output_path.clear();
            state.sketch2iso.last_sketch_path.clear();
            state.sketch2iso.last_output_path.clear();

            state.selection.pose_index = -1;
            state.pending_pose_selection.clear();
            state.pose_asset_reload_request_id++;
            state.smpl_shape.betas.fill(0.0f);

            state.scene_view_target_valid = false;
            state.scene_view_raw_rig_root_valid = false;
            state.scene_view_raw_rig_height_valid = false;
            state.scene_view_smpl_fit_valid = false;
            state.scene_view_smpl_fit_scale = { 1.0f, 1.0f, 1.0f };
            state.scene_view_smpl_fit_translation = { 0.0f, 0.0f, 0.0f };
            state.scene_view_smpl_fit_rms = 0.0f;
            state.scene_view_snapshot_pose_name.clear();
            state.scene_view_snapshot_path.clear();
            state.scene_view_snapshot_use_rigopt_camera = false;
            state.scene_view_snapshot_rigopt_camera_scale = 1.0f;
            state.scene_view_snapshot_rigopt_camera_translation = { 0.0f, 0.0f };

            state.cloth_render_path.clear();
            state.cloth_render_stroke_reference_path.clear();
            state.cloth_render_fine = false;
            state.cloth_render_use_current_camera = false;
            state.cloth_render_source_vertex_count = 0;
            state.cloth_render_vertex_count = 0;
            state.cloth_render_face_count = 0;

            state.canva_displacement_uv_json_path.clear();
            state.canva_displacement_model_input_path.clear();
            state.canva_displacement_path.clear();
            state.canva_manual_uv_png_path.clear();
            state.canva_manual_iso_npz_path.clear();
            state.canva_iso_displacement_npz_path.clear();
            state.canva_iso_displacement_model_input_path.clear();
            state.canva_iso_displacement_path.clear();

            state.uv_view_zoom = 1.0f;
            state.uv_view_pan_x = 0.0f;
            state.uv_view_pan_y = 0.0f;
            state.smpl_color_by_uv = false;
            state.garment_color_by_smpl_uv = false;

            renderer.reset_computed_state(state);
            console_add_line(state, "Canva: Reset all computed data");
        }
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

        const auto discard_reset_result = [](ModelInferenceUiState& ui, std::uint64_t& ignored_request_id)
        {
            if (ignored_request_id == 0 || ui.completed_request_id != ignored_request_id)
                return;
            ui.last_sketch_path.clear();
            ui.last_output_path.clear();
            ignored_request_id = 0;
        };
        discard_reset_result(state.sketch2rig, canva.ignore_sketch2rig_request_id);
        discard_reset_result(state.sketch2betas, canva.ignore_sketch2betas_request_id);
        discard_reset_result(state.sketch2uv, canva.ignore_sketch2uv_request_id);
        discard_reset_result(state.sketch2iso, canva.ignore_sketch2iso_request_id);

        // Consume undo requests routed from the GLFW key callback.
        while (canva.last_undo_request_id < state.canva_undo_request_id)
        {
            canva.last_undo_request_id++;
            console_add_line(state, "Canva: Ctrl+Z consumed (GLFW request)");
            const bool ok = canva_apply_undo(canva);
            console_add_line(state, ok ? "Canva: Undo applied" : "Canva: Undo stack empty");
        }

        if (state.scene_view_snapshot_completed_id > canva.last_scene_view_snapshot_completed_id)
        {
            canva.last_scene_view_snapshot_completed_id = state.scene_view_snapshot_completed_id;
            if (canva.pending_scene_view_snapshot_request_id == 0
                || state.scene_view_snapshot_completed_id >= canva.pending_scene_view_snapshot_request_id)
            {
                if (!state.scene_view_snapshot_path.empty())
                {
                    std::string err;
                    if (canva_add_smpl_render_layer(
                            canva,
                            state.scene_view_snapshot_path,
                            canva.pending_scene_view_snapshot_model_input_path,
                            &err))
                    {
                        console_add_line(state, "Canva: Added SMPL layer");
                        canva.pending_scene_view_snapshot_request_id = 0;
                        canva.pending_scene_view_snapshot_model_input_path.clear();
                    }
                    else
                    {
                        console_add_line(state, std::string("Canva: Failed to add SMPL layer: ")
                            + (err.empty() ? "(unknown error)" : err));
                        canva.pending_scene_view_snapshot_model_input_path.clear();
                    }
                }
            }
        }

        if (state.sketch2rig.completed_request_id > canva.last_sketch2rig_import_request_id)
        {
            canva.last_sketch2rig_import_request_id = state.sketch2rig.completed_request_id;
            if (canva.extract_pose_active
                && state.sketch2rig.completed_request_id == canva.extract_pose_rig_request_id)
            {
                canva.extract_pose_rig_done =
                    state.sketch2rig.status == ModelInferenceStatus::Ready
                    && !state.sketch2rig.last_output_path.empty();
                if (!canva.extract_pose_rig_done)
                {
                    console_add_line(state, "Canva: Extract Pose cancelled because rig extraction failed");
                    canva.extract_pose_active = false;
                    canva.extract_pose_hide_rig_layer = false;
                    canva.extract_pose_generate_cloth_after = false;
                    canva.generate_cloth_active = false;
                    canva.generate_cloth_status = "Failed";
                    canva.generate_cloth_failed_step = 0;
                }
            }
            if (state.sketch2rig.status == ModelInferenceStatus::Ready && !state.sketch2rig.last_output_path.empty())
            {
                std::string err;
                const bool hide_rig_layer =
                    canva.extract_pose_active
                    && canva.extract_pose_hide_rig_layer
                    && state.sketch2rig.completed_request_id == canva.extract_pose_rig_request_id;
                if (canva_add_sketch2rig_layer(
                        canva,
                        state.sketch2rig.last_output_path,
                        state.sketch2rig.last_sketch_path,
                        !hide_rig_layer,
                        &err))
                {
                    console_add_line(state, std::string(hide_rig_layer
                            ? "Canva: Added hidden Sketch2Rig rig layer from "
                            : "Canva: Added Sketch2Rig rig layer from ")
                        + state.sketch2rig.last_output_path);
                }
                else
                {
                    console_add_line(state, std::string("Canva: Failed to load Sketch2Rig rig: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
        }

        if (state.sketch2betas.completed_request_id > canva.last_sketch2betas_import_request_id)
        {
            canva.last_sketch2betas_import_request_id = state.sketch2betas.completed_request_id;
            if (canva.extract_pose_active
                && state.sketch2betas.completed_request_id == canva.extract_pose_betas_request_id)
            {
                canva.extract_pose_betas_done =
                    state.sketch2betas.status == ModelInferenceStatus::Ready
                    && !state.sketch2betas.last_output_path.empty();
                if (!canva.extract_pose_betas_done)
                {
                    console_add_line(state, "Canva: Extract Pose cancelled because betas extraction failed");
                    canva.extract_pose_active = false;
                    canva.extract_pose_hide_rig_layer = false;
                    canva.extract_pose_generate_cloth_after = false;
                    canva.generate_cloth_active = false;
                    canva.generate_cloth_status = "Failed";
                    canva.generate_cloth_failed_step = 0;
                }
            }
            if (state.sketch2betas.status == ModelInferenceStatus::Ready && !state.sketch2betas.last_output_path.empty())
            {
                std::string err;
                if (canva_apply_sketch2betas_result(state, state.sketch2betas.last_output_path, &err))
                {
                    console_add_line(state, std::string("Canva: Applied Sketch2Betas shape from ")
                        + state.sketch2betas.last_output_path);
                }
                else
                {
                    console_add_line(state, std::string("Canva: Failed to apply Sketch2Betas result: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
        }

        if (state.sketch2uv.completed_request_id > canva.last_sketch2uv_import_request_id)
        {
            canva.last_sketch2uv_import_request_id = state.sketch2uv.completed_request_id;
            const bool is_displacement_uv =
                canva.generate_displacement_active
                && canva.generate_displacement_waiting_uv
                && state.sketch2uv.completed_request_id == canva.generate_displacement_uv_request_id;
            const bool is_rig_optimization_uv =
                canva.rig_optimization_active
                && canva.rig_optimization_waiting_uv
                && state.sketch2uv.completed_request_id == canva.rig_optimization_uv_request_id;
            const bool is_cloth_fitting_uv =
                canva.cloth_fitting_active
                && canva.cloth_fitting_waiting_uv
                && state.sketch2uv.completed_request_id == canva.cloth_fitting_uv_request_id;
            const bool is_internal_uv = is_displacement_uv || is_rig_optimization_uv || is_cloth_fitting_uv;

            if (state.sketch2uv.status == ModelInferenceStatus::Ready && !state.sketch2uv.last_output_path.empty())
            {
                std::string err;
                const bool added_uv = canva_add_sketch2uv_layer(
                    canva,
                    state.sketch2uv.last_output_path,
                    state.sketch2uv.last_sketch_path,
                    !is_internal_uv,
                    is_internal_uv ? "Sketch2UV (hidden)" : "Sketch2UV",
                    &err);
                if (added_uv)
                {
                    console_add_line(state, std::string(is_internal_uv
                            ? "Canva: Added hidden Sketch2UV layer from "
                            : "Canva: Added Sketch2UV layer from ")
                        + state.sketch2uv.last_output_path);

                    if (is_displacement_uv)
                    {
                        canva.generate_displacement_waiting_uv = false;
                        canva.generate_displacement_waiting_render = true;
                        canva.generate_displacement_status = "Projecting displacement";
                        canva.generate_displacement_uv_json_path = state.sketch2uv.last_output_path;
                        state.canva_displacement_uv_json_path = state.sketch2uv.last_output_path;
                        state.canva_displacement_model_input_path = canva.generate_displacement_model_input_path.string();
                        state.canva_displacement_path.clear();
                        canva.generate_displacement_render_request_id = ++state.canva_displacement_request_id;
                        console_add_line(state, "Canva: Generate Displacement queued renderer projection");
                    }
                    if (is_rig_optimization_uv)
                    {
                        canva.rig_optimization_waiting_uv = false;
                        canva.rig_optimization_status = "Starting optimizer";
                        std::string opt_err;
                        if (!canva_start_rig_optimization(
                                state,
                                canva,
                                std::filesystem::path(state.sketch2uv.last_output_path),
                                canva.rig_optimization_model_input_path,
                                &opt_err))
                        {
                            canva.rig_optimization_active = false;
                            canva.rig_optimization_job_running = false;
                            canva.rig_optimization_status = "Failed";
                            canva.rig_optimization_last_error = opt_err.empty() ? "failed to start optimizer" : opt_err;
                            console_add_line(state, "Canva: Rig Optimization failed to start: " + canva.rig_optimization_last_error);
                        }
                    }
                    if (is_cloth_fitting_uv)
                    {
                        canva.cloth_fitting_waiting_uv = false;
                        canva.cloth_fitting_status = "Fitting";

                        const int source_idx = canva_find_layer_index_by_id(canva, canva.cloth_fitting_source_layer_id);
                        std::string fit_err;
                        if (canva_run_cloth_fitting(
                                state,
                                canva,
                                renderer,
                                std::filesystem::path(state.sketch2uv.last_output_path),
                                source_idx,
                                &fit_err))
                        {
                            canva.cloth_fitting_active = false;
                            canva.cloth_fitting_status = "Done";
                        }
                        else
                        {
                            canva.cloth_fitting_active = false;
                            canva.cloth_fitting_status = "Failed";
                            console_add_line(state, std::string("Canva: cloth Fitting failed: ")
                                + (fit_err.empty() ? "(unknown error)" : fit_err));
                        }
                    }
                }
                else
                {
                    console_add_line(state, std::string("Canva: Failed to load Sketch2UV result: ")
                        + (err.empty() ? "(unknown error)" : err));
                    if (is_displacement_uv)
                    {
                        canva.generate_displacement_active = false;
                        canva.generate_displacement_waiting_uv = false;
                        canva.generate_displacement_status = "Failed";
                    }
                    if (is_rig_optimization_uv)
                    {
                        canva.rig_optimization_active = false;
                        canva.rig_optimization_waiting_uv = false;
                        canva.rig_optimization_status = "Failed";
                        canva.rig_optimization_last_error = err.empty() ? "failed to load Sketch2UV result" : err;
                    }
                    if (is_cloth_fitting_uv)
                    {
                        canva.cloth_fitting_active = false;
                        canva.cloth_fitting_waiting_uv = false;
                        canva.cloth_fitting_status = "Failed";
                    }
                }
            }
            else if (is_displacement_uv)
            {
                canva.generate_displacement_active = false;
                canva.generate_displacement_waiting_uv = false;
                canva.generate_displacement_status = "Failed";
                console_add_line(state, "Canva: Generate Displacement cancelled because Sketch2UV failed");
            }
            else if (is_rig_optimization_uv)
            {
                canva.rig_optimization_active = false;
                canva.rig_optimization_waiting_uv = false;
                canva.rig_optimization_status = "Failed";
                canva.rig_optimization_last_error = "Sketch2UV failed";
                console_add_line(state, "Canva: Rig Optimization cancelled because Sketch2UV failed");
            }
            else if (is_cloth_fitting_uv)
            {
                canva.cloth_fitting_active = false;
                canva.cloth_fitting_waiting_uv = false;
                canva.cloth_fitting_status = "Failed";
                console_add_line(state, "Canva: cloth Fitting cancelled because Sketch2UV failed");
            }
        }

        if (state.sketch2iso.completed_request_id > canva.last_sketch2iso_import_request_id)
        {
            canva.last_sketch2iso_import_request_id = state.sketch2iso.completed_request_id;
            const bool is_iso_displacement =
                canva.generate_iso_displacement_active
                && canva.generate_iso_displacement_waiting_iso
                && state.sketch2iso.completed_request_id
                    == canva.generate_iso_displacement_iso_request_id;
            if (state.sketch2iso.status == ModelInferenceStatus::Ready
                && !state.sketch2iso.last_output_path.empty())
            {
                std::string err;
                if (canva_add_iso_debug_layer(
                        canva,
                        state.sketch2iso.last_output_path,
                        state.sketch2iso.last_sketch_path,
                        is_iso_displacement ? "Sketch2ISO (hidden)" : "Sketch2ISO Debug",
                        !is_iso_displacement,
                        &err))
                {
                    console_add_line(
                        state,
                        std::string(is_iso_displacement
                                ? "Canva: Added hidden Sketch2ISO layer from "
                                : "Canva: Added Sketch2ISO debug layer from ")
                            + state.sketch2iso.last_output_path);
                    if (is_iso_displacement)
                    {
                        canva.generate_iso_displacement_waiting_iso = false;
                        canva.generate_iso_displacement_waiting_render = true;
                        canva.generate_iso_displacement_status = "Projecting ISO displacement";
                        canva.generate_iso_displacement_npz_path =
                            state.sketch2iso.last_output_path;
                        state.canva_iso_displacement_npz_path =
                            state.sketch2iso.last_output_path;
                        state.canva_iso_displacement_model_input_path =
                            canva.generate_iso_displacement_model_input_path.string();
                        state.canva_iso_displacement_path.clear();
                        canva.generate_iso_displacement_render_request_id =
                            ++state.canva_iso_displacement_request_id;
                        console_add_line(
                            state,
                            "Canva: Create ISO Displacement Map queued renderer projection");
                    }
                }
                else
                {
                    console_add_line(state, std::string("Canva: Failed to load Sketch2ISO result: ")
                        + (err.empty() ? "(unknown error)" : err));
                    if (is_iso_displacement)
                    {
                        canva.generate_iso_displacement_active = false;
                        canva.generate_iso_displacement_waiting_iso = false;
                        canva.generate_iso_displacement_status = "Failed";
                    }
                }
            }
            else if (is_iso_displacement)
            {
                canva.generate_iso_displacement_active = false;
                canva.generate_iso_displacement_waiting_iso = false;
                canva.generate_iso_displacement_status = "Failed";
                console_add_line(
                    state,
                    "Canva: Create ISO Displacement Map cancelled because Sketch2ISO failed");
            }
        }

        (void)canva_try_finish_extract_pose(state, canva);

        if (canva.rig_optimization_job_running && canva.rig_optimization_future.valid())
        {
            const auto ready = canva.rig_optimization_future.wait_for(std::chrono::milliseconds(0));
            if (ready == std::future_status::ready)
            {
                RigOptimizationJobResult result = canva.rig_optimization_future.get();
                canva.rig_optimization_job_running = false;
                canva.rig_optimization_active = false;
                if (canva.rig_optimization_discard_result)
                {
                    canva.rig_optimization_discard_result = false;
                    console_add_line(state, "Canva: Discarded Rig Optimization result after reset");
                }
                else
                {
                    if (result.ok)
                    {
                        std::string err;
                        if (canva_apply_rig_optimization_result(state, canva, result.output_json_path, &err))
                        {
                            canva.rig_optimization_status = "Done";
                            std::ostringstream oss;
                            oss << "Canva: Rig Optimization complete";
                            if (canva.rig_optimization_last_final_loss > 0.0f)
                                oss << " (loss " << canva.rig_optimization_last_initial_loss
                                    << " -> " << canva.rig_optimization_last_final_loss << ")";
                            console_add_line(state, oss.str());
                        }
                        else
                        {
                            canva.rig_optimization_status = "Failed";
                            canva.rig_optimization_last_error = err.empty() ? "failed to apply optimizer result" : err;
                            console_add_line(state, "Canva: Rig Optimization result import failed: " + canva.rig_optimization_last_error);
                        }
                    }
                    else
                    {
                        canva.rig_optimization_status = "Failed";
                        canva.rig_optimization_last_error = result.error.empty() ? "optimizer failed" : result.error;
                        console_add_line(state, "Canva: Rig Optimization failed: " + canva.rig_optimization_last_error);
                    }
                }
            }
        }

        if (state.canva_displacement_completed_id > canva.last_displacement_completed_id)
        {
            canva.last_displacement_completed_id = state.canva_displacement_completed_id;
            if (canva.generate_displacement_waiting_render
                && state.canva_displacement_completed_id >= canva.generate_displacement_render_request_id)
            {
                canva.generate_displacement_waiting_render = false;
                canva.generate_displacement_active = false;

                if (!state.canva_displacement_path.empty())
                {
                    std::string err;
                    const bool added_layer = canva.generate_displacement_model_input_path.empty()
                        ? canva_add_canvas_filled_png_layer(
                            canva,
                            state.canva_displacement_path,
                            "Displacement",
                            CanvaLayerKind::Displacement,
                            true,
                            &err)
                        : canva_add_preprocess_aligned_png_layer(
                            canva,
                            state.canva_displacement_path,
                            canva.generate_displacement_model_input_path,
                            "Displacement",
                            CanvaLayerKind::Displacement,
                            true,
                            &err);
                    if (added_layer)
                    {
                        canva.generate_displacement_status = "Done";
                        console_add_line(state, "Canva: Added Displacement layer");
                    }
                    else
                    {
                        canva.generate_displacement_status = "Failed";
                        console_add_line(state, std::string("Canva: Failed to add Displacement layer: ")
                            + (err.empty() ? "(unknown error)" : err));
                    }
                }
                else
                {
                    canva.generate_displacement_status = "Failed";
                    console_add_line(state, "Canva: Generate Displacement projection failed");
                }
            }
        }

        if (state.canva_iso_displacement_completed_id
            > canva.last_iso_displacement_completed_id)
        {
            canva.last_iso_displacement_completed_id =
                state.canva_iso_displacement_completed_id;
            if (canva.generate_iso_displacement_waiting_render
                && state.canva_iso_displacement_completed_id
                    >= canva.generate_iso_displacement_render_request_id)
            {
                canva.generate_iso_displacement_waiting_render = false;
                canva.generate_iso_displacement_active = false;

                if (!state.canva_iso_displacement_path.empty())
                {
                    std::string error;
                    const bool added_layer =
                        canva.generate_iso_displacement_model_input_path.empty()
                        ? canva_add_canvas_filled_png_layer(
                            canva,
                            state.canva_iso_displacement_path,
                            "ISO Displacement",
                            CanvaLayerKind::IsoDisplacement,
                            true,
                            &error)
                        : canva_add_preprocess_aligned_png_layer(
                            canva,
                            state.canva_iso_displacement_path,
                            canva.generate_iso_displacement_model_input_path,
                            "ISO Displacement",
                            CanvaLayerKind::IsoDisplacement,
                            true,
                            &error);
                    if (added_layer)
                    {
                        canva.generate_iso_displacement_status = "Done";
                        console_add_line(state, "Canva: Added ISO Displacement layer");
                    }
                    else
                    {
                        canva.generate_iso_displacement_status = "Failed";
                        console_add_line(
                            state,
                            "Canva: Failed to add ISO Displacement layer: "
                                + (error.empty() ? std::string("(unknown error)") : error));
                    }
                }
                else
                {
                    canva.generate_iso_displacement_status = "Failed";
                    console_add_line(
                        state,
                        "Canva: Create ISO Displacement Map projection failed");
                }
            }
        }

        if (state.cloth_render_completed_id > canva.last_cloth_render_completed_id)
        {
            canva.last_cloth_render_completed_id = state.cloth_render_completed_id;
            if (canva.render_cloth_active
                && state.cloth_render_completed_id >= canva.render_cloth_request_id)
            {
                canva.render_cloth_active = false;
                if (!state.cloth_render_path.empty())
                {
                    std::string err;
                    if (canva_add_rendered_cloth_layer(
                            canva,
                            state.cloth_render_path,
                            canva.render_cloth_model_input_path,
                            canva.render_cloth_fine,
                            &err))
                    {
                        canva.render_cloth_status = "Done";
                        console_add_line(state,
                            canva.render_cloth_fine
                                ? "Canva: Added Fine cloth render layer"
                                : "Canva: Added Rendered cloth layer");
                    }
                    else
                    {
                        canva.render_cloth_status = "Failed";
                        console_add_line(state, "Canva: Failed to add cloth render layer: "
                            + (err.empty() ? std::string("(unknown error)") : err));
                    }
                }
                else
                {
                    canva.render_cloth_status = "Failed";
                    console_add_line(state,
                        canva.render_cloth_fine
                            ? "Canva: Fine cloth render failed"
                            : "Canva: Cloth render failed");
                }
                canva.render_cloth_model_input_path.clear();
            }

            if (canva.generate_cloth_waiting_render
                && state.cloth_render_completed_id >= canva.generate_cloth_render_request_id)
            {
                canva.generate_cloth_waiting_render = false;
                canva.generate_cloth_active = false;

                if (!state.cloth_render_path.empty())
                {
                    std::string err;
                    if (canva_add_generated_cloth_layer(
                            canva,
                            state.cloth_render_path,
                            canva.generate_cloth_model_input_path,
                            &err))
                    {
                        console_add_line(state, "Canva: Added Generated cloth layer");
                        canva.generate_cloth_status = "Done";
                        canva.generate_cloth_failed_step = -1;
                        canva.generate_cloth_model_input_path.clear();
                    }
                    else
                    {
                        console_add_line(state, std::string("Canva: Failed to add Generated cloth layer: ")
                            + (err.empty() ? "(unknown error)" : err));
                        canva.generate_cloth_status = "Failed";
                        canva.generate_cloth_failed_step = 3;
                        canva.generate_cloth_model_input_path.clear();
                    }
                }
                else
                {
                    console_add_line(state, "Canva: Generate Cloth render failed");
                    canva.generate_cloth_status = "Failed";
                    canva.generate_cloth_failed_step = 3;
                    canva.generate_cloth_model_input_path.clear();
                }
            }
        }

        if (canva.generate_cloth_waiting_pose)
        {
            const bool pose_selected =
                state.pending_pose_selection.empty()
                && state.selection.pose_index >= 0
                && state.selection.pose_index < (int)state.selection.poses.size()
                && state.selection.poses[(size_t)state.selection.pose_index] == canva.generate_cloth_pose_name;

            if (pose_selected)
            {
                if (state.selection.garment_index < 0
                    || state.selection.garment_index >= (int)state.selection.garments.size())
                {
                    console_add_line(state, "Canva: Generate Cloth cancelled because no garment mesh is selected");
                    canva.generate_cloth_waiting_pose = false;
                    canva.generate_cloth_active = false;
                    canva.generate_cloth_status = "Failed";
                    canva.generate_cloth_failed_step = 1;
                }
                else
                {
                    state.show_smpl = true;
                    state.show_smpl_character = true;
                    state.show_garment_mesh = true;
                    state.controls.current_frame = 0;
                    state.controls.playing = false;
                    state.controls.simulating = true;
                    state.cloth_simulate_request_id++;
                    canva.generate_cloth_waiting_pose = false;
                    canva.generate_cloth_waiting_simulation = true;
                    canva.generate_cloth_status = "Simulating";
                    canva.generate_cloth_failed_step = -1;
                    console_add_line(state, "Canva: Generate Cloth queued XPBD simulation");
                }
            }
        }

        if (canva.generate_cloth_waiting_simulation && !state.controls.simulating)
        {
            canva.generate_cloth_waiting_simulation = false;
            if (state.controls.frame_count <= 0)
            {
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Failed";
                canva.generate_cloth_failed_step = 2;
                console_add_line(state, "Canva: Generate Cloth cancelled because XPBD produced no frames");
            }
            else
            {
                state.controls.playing = false;
                state.controls.current_frame = std::max(0, state.controls.frame_count - 1);
                state.cloth_render_path.clear();
                state.cloth_render_fine = false;
                state.cloth_render_use_current_camera = false;
                state.cloth_render_stroke_reference_path = canva.generate_cloth_model_input_path.string();
                canva.generate_cloth_render_request_id = ++state.cloth_render_request_id;
                canva.generate_cloth_waiting_render = true;
                canva.generate_cloth_status = "Rendering";
                canva.generate_cloth_failed_step = -1;
                console_add_line(state, "Canva: Generate Cloth queued transparent cloth render");
            }
        }

        // Global shortcuts for this tab (works even when hovering the controls).
        {
            ImGuiIO& io = ImGui::GetIO();
            const bool focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
            if (focused && io.KeyCtrl && !io.WantTextInput)
            {
                // Transform tool toggle (Ctrl+T)
                if (ImGui::IsKeyPressed(GLFW_KEY_T))
                {
                    canva.tool = (canva.tool == CanvaTool::Brush) ? CanvaTool::Transform : CanvaTool::Brush;
                    canva_cancel_interactions(canva);
                }
            }
        }

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float spacing_x = ImGui::GetStyle().ItemSpacing.x;
        const bool show_generate_cloth_steps =
            canva.generate_cloth_active
            || canva.generate_cloth_status == "Done"
            || canva.generate_cloth_status == "Failed";
        float right_w = 420.0f;
        if (avail.x < right_w + 220.0f)
            right_w = std::max(200.0f, avail.x * 0.35f);
        const float left_w = std::max(1.0f, avail.x - right_w - spacing_x);

        // Left: drawing canvas view.
        ImGui::BeginChild("##canva_view", ImVec2(left_w, 0.0f), false,
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        {
            const ImVec2 view_pos = ImGui::GetCursorScreenPos();
            const ImVec2 view_size = ImGui::GetContentRegionAvail();

            if (view_size.x > 1.0f && view_size.y > 1.0f)
            {
                ImGui::InvisibleButton("##canva_invisible", view_size,
                    ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);

                const bool hovered = ImGui::IsItemHovered();
                ImGuiIO& io = ImGui::GetIO();
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const bool shift = io.KeyShift;

                const ImVec2 view_center = ImVec2(view_pos.x + view_size.x * 0.5f, view_pos.y + view_size.y * 0.5f);

                // Zoom (mouse wheel) around cursor.
                if (hovered)
                {
                    const float wheel = io.MouseWheel;
                    if (wheel != 0.0f)
                    {
                        const float zoom_before = canva.zoom;
                        const float zoom_mul = (wheel > 0.0f) ? 1.12f : (1.0f / 1.12f);
                        canva.zoom = std::clamp(canva.zoom * zoom_mul, 0.10f, 20.0f);
                        const float zoom_after = canva.zoom;

                        const ImVec2 mouse = io.MousePos;
                        const ImVec2 canvas_size_before = ImVec2((float)canva.width * zoom_before, (float)canva.height * zoom_before);
                        const ImVec2 origin_centered_before = ImVec2(
                            view_center.x - canvas_size_before.x * 0.5f,
                            view_center.y - canvas_size_before.y * 0.5f);
                        const ImVec2 origin_before = ImVec2(origin_centered_before.x + canva.pan.x, origin_centered_before.y + canva.pan.y);
                        const ImVec2 p_canvas = ImVec2(
                            (zoom_before > 1e-6f) ? ((mouse.x - origin_before.x) / zoom_before) : 0.0f,
                            (zoom_before > 1e-6f) ? ((mouse.y - origin_before.y) / zoom_before) : 0.0f);

                        const ImVec2 canvas_size_after = ImVec2((float)canva.width * zoom_after, (float)canva.height * zoom_after);
                        const ImVec2 origin_centered_after = ImVec2(
                            view_center.x - canvas_size_after.x * 0.5f,
                            view_center.y - canvas_size_after.y * 0.5f);
                        const ImVec2 origin_after = ImVec2(mouse.x - p_canvas.x * zoom_after, mouse.y - p_canvas.y * zoom_after);

                        canva.pan.x = origin_after.x - origin_centered_after.x;
                        canva.pan.y = origin_after.y - origin_centered_after.y;
                    }
                }

                // Pan (middle mouse drag).
                if (hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
                {
                    const ImVec2 d = io.MouseDelta;
                    canva.pan.x += d.x;
                    canva.pan.y += d.y;
                }

                const ImVec2 canvas_size = ImVec2((float)canva.width * canva.zoom, (float)canva.height * canva.zoom);
                const ImVec2 origin_centered = ImVec2(view_center.x - canvas_size.x * 0.5f, view_center.y - canvas_size.y * 0.5f);
                const ImVec2 origin = ImVec2(origin_centered.x + canva.pan.x, origin_centered.y + canva.pan.y);
                const ImVec2 p0 = origin;
                const ImVec2 p1 = ImVec2(origin.x + canvas_size.x, origin.y + canvas_size.y);

                const ImVec2 mouse = io.MousePos;
                const ImVec2 local = ImVec2((mouse.x - origin.x) / canva.zoom, (mouse.y - origin.y) / canva.zoom);
                const bool in_canvas = (local.x >= 0.0f && local.y >= 0.0f && local.x < (float)canva.width && local.y < (float)canva.height);

                dl->PushClipRect(view_pos, ImVec2(view_pos.x + view_size.x, view_pos.y + view_size.y), true);

                // Canvas background (white).
                dl->AddRectFilled(p0, p1, IM_COL32(255, 255, 255, 255));

                // Clip layers to canvas bounds.
                dl->PushClipRect(p0, p1, true);

                // Draw layers bottom->top (vector is top->bottom).
                for (int i = (int)canva.layers.size() - 1; i >= 0; --i)
                {
                    const auto& layer = canva.layers[(size_t)i];
                    if (!layer.enabled || layer.texture_id == 0)
                        continue;
                    ImTextureID tid = (ImTextureID)(intptr_t)layer.texture_id;

                    const ImVec2 lp0 = ImVec2(p0.x + layer.offset.x * canva.zoom, p0.y + layer.offset.y * canva.zoom);
                    const float s = std::max(1e-6f, layer.scale);
                    const ImVec2 lp1 = ImVec2(lp0.x + canvas_size.x * s, lp0.y + canvas_size.y * s);
                    dl->AddImage(tid, lp0, lp1);
                }

                dl->PopClipRect();

                dl->AddRect(p0, p1, IM_COL32(0, 0, 0, 140));

                if (show_generate_cloth_steps)
                {
                    const ImVec2 steps_anchor(
                        p1.x + 28.0f * canva.zoom,
                        p0.y + 16.0f * canva.zoom);
                    draw_generate_cloth_steps_in_view(state, canva, dl, steps_anchor);
                }

                // Transform tool: pick + drag active layer.
                if (canva.tool == CanvaTool::Transform)
                {
                    int hovered_handle = -1;
                    ImVec2 handle_pos[4] = { ImVec2(0, 0), ImVec2(0, 0), ImVec2(0, 0), ImVec2(0, 0) };
                    CanvaBounds active_bounds;

                    if (!canva.layers.empty())
                    {
                        // Draw active layer bounds.
                        CanvaLayer& layer = canva.layers[(size_t)canva.active_layer];
                        active_bounds = canva_layer_content_bounds(layer, canva.width, canva.height);

                        const float s = std::max(1e-6f, layer.scale);
                        const ImVec2 tl_canvas = ImVec2(layer.offset.x + (float)active_bounds.min_x * s, layer.offset.y + (float)active_bounds.min_y * s);
                        const ImVec2 br_canvas = ImVec2(layer.offset.x + (float)active_bounds.max_x_excl * s, layer.offset.y + (float)active_bounds.max_y_excl * s);
                        const ImVec2 lp0 = ImVec2(p0.x + tl_canvas.x * canva.zoom, p0.y + tl_canvas.y * canva.zoom);
                        const ImVec2 lp1 = ImVec2(p0.x + br_canvas.x * canva.zoom, p0.y + br_canvas.y * canva.zoom);
                        dl->AddRect(lp0, lp1, IM_COL32(40, 40, 40, 200), 0.0f, 0, 1.5f);

                        // Corner scale handles.
                        handle_pos[0] = lp0;                  // TL
                        handle_pos[1] = ImVec2(lp1.x, lp0.y); // TR
                        handle_pos[2] = lp1;                  // BR
                        handle_pos[3] = ImVec2(lp0.x, lp1.y); // BL

                        const float hr = 6.0f;
                        const float hr2 = hr * hr;
                        for (int i = 0; i < 4; ++i)
                        {
                            const float dx = mouse.x - handle_pos[i].x;
                            const float dy = mouse.y - handle_pos[i].y;
                            if (dx * dx + dy * dy <= hr2)
                                hovered_handle = i;
                        }

                        for (int i = 0; i < 4; ++i)
                        {
                            const bool hi = (i == hovered_handle);
                            const ImU32 fill = hi ? IM_COL32(30, 30, 30, 255) : IM_COL32(50, 50, 50, 230);
                            const ImU32 outl = IM_COL32(220, 220, 220, 230);
                            const ImVec2 a = ImVec2(handle_pos[i].x - hr, handle_pos[i].y - hr);
                            const ImVec2 b = ImVec2(handle_pos[i].x + hr, handle_pos[i].y + hr);
                            dl->AddRectFilled(a, b, fill, 2.0f);
                            dl->AddRect(a, b, outl, 2.0f, 0, 1.0f);
                        }
                    }

                    // Pick layer on click, but don't steal clicks intended for the scale handles.
                    if (hovered && in_canvas && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hovered_handle < 0)
                    {
                        const int picked = canva_pick_layer_at(canva, local);
                        if (picked >= 0)
                            canva.active_layer = picked;
                    }

                    if (hovered && in_canvas && ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDown(ImGuiMouseButton_Middle))
                    {
                        if (!canva.transforming)
                        {
                            // If starting from a handle, transform the already-active layer.
                            if (hovered_handle < 0)
                            {
                                const int picked = canva_pick_layer_at(canva, local);
                                if (picked >= 0)
                                    canva.active_layer = picked;
                            }

                            canva.transform_layer = canva.active_layer;
                            canva.transforming = true;
                            canva.scaling = (hovered_handle >= 0);
                            canva.transform_start_mouse = mouse;

                            CanvaLayer& layer = canva.layers[(size_t)canva.transform_layer];
                            canva.transform_start_offset = layer.offset;
                            canva.transform_start_scale = layer.scale;
                            canva_push_undo(canva, layer.id, layer.pixels, layer.offset, layer.scale);

                            if (canva.scaling)
                            {
                                const ImVec2 min_px = ImVec2((float)active_bounds.min_x, (float)active_bounds.min_y);
                                const ImVec2 max_px = ImVec2((float)active_bounds.max_x_excl, (float)active_bounds.max_y_excl);

                                // dragged corner (handle) and opposite anchor corner in *layer pixel* coords.
                                if (hovered_handle == 0) // TL
                                {
                                    canva.transform_drag_px = min_px;
                                    canva.transform_anchor_px = max_px;
                                }
                                else if (hovered_handle == 1) // TR
                                {
                                    canva.transform_drag_px = ImVec2(max_px.x, min_px.y);
                                    canva.transform_anchor_px = ImVec2(min_px.x, max_px.y);
                                }
                                else if (hovered_handle == 2) // BR
                                {
                                    canva.transform_drag_px = max_px;
                                    canva.transform_anchor_px = min_px;
                                }
                                else // BL
                                {
                                    canva.transform_drag_px = ImVec2(min_px.x, max_px.y);
                                    canva.transform_anchor_px = ImVec2(max_px.x, min_px.y);
                                }

                                const float ss = std::max(1e-6f, canva.transform_start_scale);
                                canva.transform_anchor_canvas = ImVec2(
                                    canva.transform_start_offset.x + canva.transform_anchor_px.x * ss,
                                    canva.transform_start_offset.y + canva.transform_anchor_px.y * ss);
                            }
                        }

                        if (canva.transform_layer >= 0 && canva.transform_layer < (int)canva.layers.size())
                        {
                            CanvaLayer& layer = canva.layers[(size_t)canva.transform_layer];
                            if (!canva.scaling)
                            {
                                const ImVec2 d_screen = ImVec2(mouse.x - canva.transform_start_mouse.x, mouse.y - canva.transform_start_mouse.y);
                                const ImVec2 d_canvas = ImVec2(d_screen.x / canva.zoom, d_screen.y / canva.zoom);
                                layer.offset = ImVec2(canva.transform_start_offset.x + d_canvas.x, canva.transform_start_offset.y + d_canvas.y);
                            }
                            else
                            {
                                // Scale uniformly by dragging a corner handle. The opposite corner stays fixed.
                                const ImVec2 mouse_canvas = local;
                                const ImVec2 dp = ImVec2(canva.transform_drag_px.x - canva.transform_anchor_px.x,
                                    canva.transform_drag_px.y - canva.transform_anchor_px.y);
                                const ImVec2 dc = ImVec2(mouse_canvas.x - canva.transform_anchor_canvas.x,
                                    mouse_canvas.y - canva.transform_anchor_canvas.y);

                                float s_new = canva.transform_start_scale;
                                const bool have_x = std::abs(dp.x) > 1e-6f;
                                const bool have_y = std::abs(dp.y) > 1e-6f;
                                float sx = 0.0f;
                                float sy = 0.0f;
                                if (have_x)
                                    sx = dc.x / dp.x;
                                if (have_y)
                                    sy = dc.y / dp.y;

                                if (have_x && have_y)
                                    s_new = (std::abs(sx) > std::abs(sy)) ? sx : sy;
                                else if (have_x)
                                    s_new = sx;
                                else if (have_y)
                                    s_new = sy;

                                s_new = std::clamp(std::abs(s_new), 0.05f, 20.0f);

                                layer.scale = s_new;
                                layer.offset = ImVec2(
                                    canva.transform_anchor_canvas.x - canva.transform_anchor_px.x * s_new,
                                    canva.transform_anchor_canvas.y - canva.transform_anchor_px.y * s_new);
                            }
                        }
                    }
                    else
                    {
                        canva.transforming = false;
                        canva.scaling = false;
                        canva.transform_layer = -1;
                    }

                    // No drawing in transform tool.
                    canva.drawing = false;
                    canva.sizing = false;
                }

                // Brush tool: Shift+drag changes brush size.
                if (canva.tool == CanvaTool::Brush)
                {
                    const bool left_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);
                    const bool middle_down = ImGui::IsMouseDown(ImGuiMouseButton_Middle);

                    if (hovered && in_canvas && shift && left_down && !middle_down)
                    {
                        if (!canva.sizing)
                        {
                            canva.sizing = true;
                            canva.size_start_mouse = mouse;
                            canva.size_start_radius = canva.brush_radius;
                        }

                        const float dx_canvas = (mouse.x - canva.size_start_mouse.x) / std::max(1e-6f, canva.zoom);
                        const float new_radius = canva.size_start_radius + dx_canvas;
                        canva.brush_radius = std::clamp(new_radius, 0.5f, 250.0f);

                        // While sizing, do not draw.
                        canva.drawing = false;
                    }
                    else
                    {
                        canva.sizing = false;
                    }

                    // Drawing (left mouse).
                    if (hovered && in_canvas && left_down && !middle_down && !shift)
                    {
                        CanvaLayer& layer = canva.layers[(size_t)canva.active_layer];
                        const float s = std::max(1e-6f, layer.scale);
                        const ImVec2 layer_local = ImVec2((local.x - layer.offset.x) / s, (local.y - layer.offset.y) / s);
                        const bool in_layer = (layer_local.x >= 0.0f && layer_local.y >= 0.0f
                            && layer_local.x < (float)canva.width && layer_local.y < (float)canva.height);

                        if (in_layer)
                        {
                            if (!canva.drawing)
                            {
                                // Snapshot for undo at the start of a stroke.
                                canva_push_undo(canva, layer.id, layer.pixels, layer.offset, layer.scale);

                                canva.drawing = true;
                                canva.last_canvas_pos = layer_local;
                                const float r_layer = canva.brush_radius / s;
                                canva_stamp_circle(layer, canva.width, canva.height, layer_local.x, layer_local.y, r_layer, canva.brush_color);
                            }
                            else
                            {
                                const float r_layer = canva.brush_radius / s;
                                canva_draw_segment(layer, canva.width, canva.height, canva.last_canvas_pos, layer_local, r_layer, canva.brush_color);
                                canva.last_canvas_pos = layer_local;
                            }
                        }
                        else
                        {
                            canva.drawing = false;
                        }
                    }
                    else
                    {
                        canva.drawing = false;
                    }

                    // Custom cursor: brush size preview circle.
                    if (hovered && in_canvas)
                    {
                        const float r_px = std::max(0.5f, canva.brush_radius * canva.zoom);
                        const ImU32 outline = IM_COL32(0, 0, 0, 210);

                        ImVec4 solid = canva.brush_color;
                        solid.w = 1.0f;
                        const ImU32 fill = ImGui::ColorConvertFloat4ToU32(solid);

                        dl->AddCircle(mouse, r_px, outline, 0, 1.25f);
                        dl->AddCircleFilled(mouse, 1.5f, outline);
                        dl->AddCircleFilled(mouse, 1.0f, fill);
                    }
                }

                dl->PopClipRect();

                // Upload textures if needed (after painting).
                for (auto& layer : canva.layers)
                {
                    if (layer.dirty)
                        canva_upload_layer(layer, canva.width, canva.height);
                }
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        // Right: brush + layers controls.
        ImGui::BeginChild("##canva_controls", ImVec2(0.0f, 0.0f), true,
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.16f, 0.16f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.70f, 0.20f, 0.20f, 1.0f));
            if (ImGui::Button("Reset Everything"))
                ImGui::OpenPopup("Reset Canva?");
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Clear canvas layers, rig, UV/maps, generated renders, pose, shape, and simulation.");

            if (ImGui::BeginPopupModal("Reset Canva?", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
            {
                ImGui::TextUnformatted("Reset all computed Canva data?");
                ImGui::TextUnformatted("Generated files on disk will be kept.");
                ImGui::Spacing();
                if (ImGui::Button("Reset", ImVec2(120.0f, 0.0f)))
                {
                    canva_reset_all(state, canva, renderer);
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel", ImVec2(120.0f, 0.0f)))
                    ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Brush");
            ImGui::Separator();

            ImGui::BeginDisabled(canva.tool != CanvaTool::Brush);
            ImGui::SliderFloat("Size", &canva.brush_radius, 0.5f, 250.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
            ImGui::ColorEdit4("Color", (float*)&canva.brush_color, ImGuiColorEditFlags_AlphaBar);
            ImGui::EndDisabled();

            // Persist brush settings via AppState -> imgui.ini handler.
            state.canva_brush_radius = std::clamp(canva.brush_radius, 0.5f, 250.0f);
            state.canva_brush_color = {
                std::clamp(canva.brush_color.x, 0.0f, 1.0f),
                std::clamp(canva.brush_color.y, 0.0f, 1.0f),
                std::clamp(canva.brush_color.z, 0.0f, 1.0f),
                std::clamp(canva.brush_color.w, 0.0f, 1.0f)
            };

            if (ImGui::Button("Reset view"))
            {
                canva.zoom = 1.0f;
                canva.pan = ImVec2(0.0f, 0.0f);
            }
            ImGui::SameLine();
            ImGui::Text("Zoom: %.2fx", (double)canva.zoom);

            ImGui::Separator();
            ImGui::TextUnformatted("Tool");
            ImGui::Separator();
            if (ImGui::RadioButton("Brush", canva.tool == CanvaTool::Brush))
            {
                canva.tool = CanvaTool::Brush;
                canva_cancel_interactions(canva);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Transform", canva.tool == CanvaTool::Transform))
            {
                canva.tool = CanvaTool::Transform;
                canva_cancel_interactions(canva);
            }

            ImGui::TextUnformatted("Layers");
            ImGui::Separator();

            if (ImGui::Button("Load##canva_load_image"))
            {
                const char* patterns[] = {
                    "*.png", "*.PNG",
                    "*.jpg", "*.JPG",
                    "*.jpeg", "*.JPEG"
                };

                std::error_code ec;
                std::filesystem::path initial_dir = std::filesystem::path("assets") / "sketches";
                if (!std::filesystem::exists(initial_dir, ec))
                    initial_dir = std::filesystem::path("assets") / "sketch";
                const std::string initial_dir_string =
                    std::filesystem::exists(initial_dir, ec) ? initial_dir.string() : std::string();

                const char* path = tinyfd_openFileDialog(
                    "Load image",
                    initial_dir_string.c_str(),
                    (int)(sizeof(patterns) / sizeof(patterns[0])),
                    patterns,
                    "Image files (*.png, *.jpg, *.jpeg)",
                    0);

                // The blocking dialog can desync GLFW's cached cursor mode from
                // the actual X11/OS cursor state. Force a full restore next frame.
                state.cursor_restore_request = true;

                if (path && path[0] != '\0')
                {
                    std::string err;
                    if (canva_add_png_layer(canva, path, &err))
                    {
                        console_add_line(state, std::string("Canva: Loaded image layer: ") + canva.layers.front().name);
                    }
                    else
                    {
                        console_add_line(state, std::string("Canva: Image load failed: ") + (err.empty() ? "(unknown error)" : err));
                    }
                }
                else
                {
                    console_add_line(state, "Canva: Load cancelled");
                }
            }

            ImGui::SameLine();
            if (ImGui::Button("Save##canva_save_image"))
            {
                std::filesystem::path out_path;
                std::string err;
                if (canva_save_visible_layers_png(canva, &out_path, &err))
                {
                    console_add_line(state, std::string("Canva: Saved PNG: ") + out_path.string());
                }
                else
                {
                    console_add_line(state, std::string("Canva: Save failed: ") + (err.empty() ? "(unknown error)" : err));
                }
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Sketch2Rig");
            ImGui::SameLine();
            ImGui::TextColored(model_inference_status_color(state.sketch2rig.status), "%s",
                model_inference_status_text(state.sketch2rig.status));

            if (!state.sketch2rig.message.empty())
                ImGui::TextWrapped("%s", state.sketch2rig.message.c_str());

            const bool can_process =
                state.sketch2rig.service != nullptr &&
                state.sketch2rig.status == ModelInferenceStatus::Ready;
            ImGui::BeginDisabled(!can_process);
            if (ImGui::Button("Extract Rig"))
            {
                std::filesystem::path sketch_path;
                std::string err;
                if (canva_save_model_input_png(canva, &sketch_path, &err))
                {
                    const std::filesystem::path output_path =
                        sketch_path.parent_path() / ("sketch_rig.json");
                    const std::uint64_t request_id = state.sketch2rig.next_request_id++;

                    state.sketch2rig.last_sketch_path = sketch_path.string();
                    console_add_line(state, std::string("Canva: Saved normalized sketch for Sketch2Rig: ") + sketch_path.string());

                    std::string request_err;
                    if (state.sketch2rig.service->request_inference(request_id, sketch_path, output_path, &request_err))
                    {
                        state.sketch2rig.status = ModelInferenceStatus::Busy;
                        state.sketch2rig.active_request_id = request_id;
                        state.sketch2rig.message = "Processing sketch with Sketch2Rig...";
                    }
                    else
                    {
                        state.sketch2rig.status = ModelInferenceStatus::Error;
                        state.sketch2rig.message = request_err.empty() ? "Failed to queue inference" : request_err;
                        console_add_line(state, "Sketch2Rig: " + state.sketch2rig.message);
                    }
                }
                else
                {
                    console_add_line(state, std::string("Canva: Process preprocessing failed: ") + (err.empty() ? "(unknown error)" : err));
                }
            }
            ImGui::EndDisabled();

            if (!state.sketch2rig.last_output_path.empty())
                ImGui::TextWrapped("Output: %s", state.sketch2rig.last_output_path.c_str());

            ImGui::Separator();
            ImGui::TextUnformatted("Sketch2Betas");
            ImGui::SameLine();
            ImGui::TextColored(model_inference_status_color(state.sketch2betas.status), "%s",
                model_inference_status_text(state.sketch2betas.status));

            if (!state.sketch2betas.message.empty())
                ImGui::TextWrapped("%s", state.sketch2betas.message.c_str());

            const bool can_extract_betas =
                state.sketch2betas.service != nullptr &&
                state.sketch2betas.status == ModelInferenceStatus::Ready;
            ImGui::BeginDisabled(!can_extract_betas);
            if (ImGui::Button("Extract betas"))
            {
                std::filesystem::path sketch_path;
                std::string err;
                if (canva_save_model_input_png(canva, &sketch_path, &err))
                {
                    const std::filesystem::path output_path =
                        sketch_path.parent_path() / ("sketch_betas.json");
                    const std::uint64_t request_id = state.sketch2betas.next_request_id++;

                    state.sketch2betas.last_sketch_path = sketch_path.string();
                    console_add_line(state, std::string("Canva: Saved normalized sketch for Sketch2Betas: ") + sketch_path.string());

                    std::string request_err;
                    if (state.sketch2betas.service->request_inference(request_id, sketch_path, output_path, &request_err))
                    {
                        state.sketch2betas.status = ModelInferenceStatus::Busy;
                        state.sketch2betas.active_request_id = request_id;
                        state.sketch2betas.message = "Processing sketch with Sketch2Betas...";
                    }
                    else
                    {
                        state.sketch2betas.status = ModelInferenceStatus::Error;
                        state.sketch2betas.message = request_err.empty() ? "Failed to queue inference" : request_err;
                        console_add_line(state, "Sketch2Betas: " + state.sketch2betas.message);
                    }
                }
                else
                {
                    console_add_line(state, std::string("Canva: Extract betas preprocessing failed: ") + (err.empty() ? "(unknown error)" : err));
                }
            }
            ImGui::EndDisabled();

            if (!state.sketch2betas.last_output_path.empty())
                ImGui::TextWrapped("Output: %s", state.sketch2betas.last_output_path.c_str());

            ImGui::Separator();
            ImGui::TextUnformatted("Sketch2UV");
            ImGui::SameLine();
            ImGui::TextColored(model_inference_status_color(state.sketch2uv.status), "%s",
                model_inference_status_text(state.sketch2uv.status));

            if (!state.sketch2uv.message.empty())
                ImGui::TextWrapped("%s", state.sketch2uv.message.c_str());

            const bool can_predict_uv =
                state.sketch2uv.service != nullptr &&
                state.sketch2uv.status == ModelInferenceStatus::Ready;
            ImGui::BeginDisabled(!can_predict_uv);
            if (ImGui::Button("Predict UV"))
            {
                std::filesystem::path sketch_path;
                std::string err;
                if (canva_save_model_input_png(canva, &sketch_path, &err))
                {
                    const std::filesystem::path output_path =
                        sketch_path.parent_path() / ("sketch_uv.json");
                    const std::uint64_t request_id = state.sketch2uv.next_request_id++;

                    state.sketch2uv.last_sketch_path = sketch_path.string();
                    console_add_line(state, std::string("Canva: Saved normalized sketch for Sketch2UV: ") + sketch_path.string());

                    std::string request_err;
                    if (state.sketch2uv.service->request_inference(request_id, sketch_path, output_path, &request_err))
                    {
                        state.sketch2uv.status = ModelInferenceStatus::Busy;
                        state.sketch2uv.active_request_id = request_id;
                        state.sketch2uv.message = "Processing sketch with Sketch2UV...";
                    }
                    else
                    {
                        state.sketch2uv.status = ModelInferenceStatus::Error;
                        state.sketch2uv.message = request_err.empty() ? "Failed to queue inference" : request_err;
                        console_add_line(state, "Sketch2UV: " + state.sketch2uv.message);
                    }
                }
                else
                {
                    console_add_line(state, std::string("Canva: Predict UV preprocessing failed: ") + (err.empty() ? "(unknown error)" : err));
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Load UV##canva_load_uv"))
            {
                const char* patterns[] = { "*.png", "*.PNG" };

                std::error_code ec;
                std::filesystem::path initial_dir = std::filesystem::path("renders");
                if (!std::filesystem::exists(initial_dir, ec))
                    initial_dir = std::filesystem::path("assets") / "sketches";
                const std::string initial_dir_string =
                    std::filesystem::exists(initial_dir, ec) ? initial_dir.string() : std::string();

                const char* path = tinyfd_openFileDialog(
                    "Load UV PNG",
                    initial_dir_string.c_str(),
                    (int)(sizeof(patterns) / sizeof(patterns[0])),
                    patterns,
                    "UV PNG files (*.png)",
                    0);

                state.cursor_restore_request = true;

                if (path && path[0] != '\0')
                {
                    std::string err;
                    const std::string layer_name = std::string("Loaded UV: ") + canva_basename(path);
                    if (canva_add_canvas_filled_png_layer(
                            canva,
                            path,
                            layer_name,
                            CanvaLayerKind::ManualUV,
                            true,
                            &err))
                    {
                        state.canva_manual_uv_png_path = path;
                        console_add_line(state, std::string("Canva: Loaded manual UV PNG: ") + path);
                    }
                    else
                    {
                        console_add_line(state, std::string("Canva: UV PNG load failed: ")
                            + (err.empty() ? "(unknown error)" : err));
                    }
                }
                else
                {
                    console_add_line(state, "Canva: Load UV cancelled");
                }
            }

            if (!state.sketch2uv.last_output_path.empty())
                ImGui::TextWrapped("Output: %s", state.sketch2uv.last_output_path.c_str());
            if (!state.canva_manual_uv_png_path.empty())
            {
                ImGui::TextWrapped("Loaded UV: %s", state.canva_manual_uv_png_path.c_str());
                ImGui::SameLine();
                if (ImGui::Button("Clear##canva_clear_loaded_uv"))
                {
                    state.canva_manual_uv_png_path.clear();
                    console_add_line(state, "Canva: Cleared loaded UV PNG");
                }
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Sketch2ISO");
            ImGui::SameLine();
            ImGui::TextColored(model_inference_status_color(state.sketch2iso.status), "%s",
                model_inference_status_text(state.sketch2iso.status));

            if (!state.sketch2iso.message.empty())
                ImGui::TextWrapped("%s", state.sketch2iso.message.c_str());

            const bool can_predict_iso =
                state.sketch2iso.service != nullptr
                && state.sketch2iso.status == ModelInferenceStatus::Ready;
            ImGui::BeginDisabled(!can_predict_iso);
            if (ImGui::Button("Predict ISO"))
            {
                std::filesystem::path sketch_path;
                std::string err;
                if (canva_save_model_input_png(canva, &sketch_path, &err))
                {
                    const std::filesystem::path output_path =
                        sketch_path.parent_path() / "sketch_iso.npz";
                    const std::uint64_t request_id = state.sketch2iso.next_request_id++;

                    state.sketch2iso.last_sketch_path = sketch_path.string();
                    console_add_line(state, "Canva: Saved normalized sketch for Sketch2ISO: "
                        + sketch_path.string());

                    std::string request_err;
                    if (state.sketch2iso.service->request_inference(
                            request_id, sketch_path, output_path, &request_err))
                    {
                        state.sketch2iso.status = ModelInferenceStatus::Busy;
                        state.sketch2iso.active_request_id = request_id;
                        state.sketch2iso.message = "Processing sketch with Sketch2ISO...";
                    }
                    else
                    {
                        state.sketch2iso.status = ModelInferenceStatus::Error;
                        state.sketch2iso.message =
                            request_err.empty() ? "Failed to queue inference" : request_err;
                        console_add_line(state, "Sketch2ISO: " + state.sketch2iso.message);
                    }
                }
                else
                {
                    console_add_line(state, std::string("Canva: Predict ISO preprocessing failed: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Load ISO##canva_load_iso"))
            {
                const char* patterns[] = { "*.npz", "*.NPZ" };
                const std::filesystem::path initial_dir =
                    assets_dir() / "sketches" / "sketches_iso";
                std::error_code ec;
                const std::string initial_dir_string =
                    std::filesystem::exists(initial_dir, ec) ? initial_dir.string() : std::string();

                const char* path = tinyfd_openFileDialog(
                    "Load ISO NPZ",
                    initial_dir_string.c_str(),
                    (int)(sizeof(patterns) / sizeof(patterns[0])),
                    patterns,
                    "ISO NPZ files (*.npz)",
                    0);

                state.cursor_restore_request = true;
                if (path && path[0] != '\0')
                {
                    std::string err;
                    const std::string layer_name =
                        std::string("Loaded ISO Debug: ") + canva_basename(path);
                    if (canva_add_iso_debug_layer(canva, path, {}, layer_name, true, &err))
                    {
                        state.canva_manual_iso_npz_path = path;
                        console_add_line(state, std::string("Canva: Loaded ISO NPZ: ") + path);
                    }
                    else
                    {
                        console_add_line(state, std::string("Canva: ISO NPZ load failed: ")
                            + (err.empty() ? "(unknown error)" : err));
                    }
                }
                else
                {
                    console_add_line(state, "Canva: Load ISO cancelled");
                }
            }

            if (!state.sketch2iso.last_output_path.empty())
                ImGui::TextWrapped("Output: %s", state.sketch2iso.last_output_path.c_str());
            if (!state.canva_manual_iso_npz_path.empty())
            {
                ImGui::TextWrapped("Loaded ISO: %s", state.canva_manual_iso_npz_path.c_str());
                ImGui::SameLine();
                if (ImGui::Button("Clear##canva_clear_loaded_iso"))
                {
                    state.canva_manual_iso_npz_path.clear();
                    console_add_line(state, "Canva: Cleared loaded ISO NPZ");
                }
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Rig Optimization");
            std::string rig_opt_pose_error;
            std::filesystem::path rig_opt_pose_path;
            const bool rig_opt_pose_available =
                canva_current_predicted_pose_path(state, &rig_opt_pose_path, &rig_opt_pose_error);
            std::error_code rig_opt_ec;
            const bool rig_opt_have_uv =
                !state.sketch2uv.last_output_path.empty()
                && std::filesystem::exists(std::filesystem::path(state.sketch2uv.last_output_path), rig_opt_ec);
            const bool rig_opt_can_queue_uv =
                state.sketch2uv.service != nullptr
                && state.sketch2uv.status == ModelInferenceStatus::Ready;
            const bool can_optimize_rig =
                rig_opt_pose_available
                && !canva.rig_optimization_active
                && !canva.rig_optimization_job_running
                && (rig_opt_have_uv || rig_opt_can_queue_uv);

            ImGui::PushItemWidth(112.0f);
            ImGui::InputInt("Iterations##rig_opt_iterations", &canva.rig_optimization_iterations, 10, 25);
            canva.rig_optimization_iterations = std::clamp(canva.rig_optimization_iterations, 1, 1000);
            ImGui::SameLine();
            ImGui::InputInt("Render##rig_opt_render_size", &canva.rig_optimization_render_size, 16, 32);
            canva.rig_optimization_render_size = std::clamp(canva.rig_optimization_render_size, 64, 384);
            ImGui::InputFloat("LR##rig_opt_lr", &canva.rig_optimization_lr, 0.005f, 0.02f, "%.4f");
            canva.rig_optimization_lr = std::clamp(canva.rig_optimization_lr, 0.0001f, 1.0f);
            ImGui::SameLine();
            ImGui::InputFloat("Pose reg##rig_opt_pose_reg", &canva.rig_optimization_pose_reg, 0.002f, 0.01f, "%.4f");
            canva.rig_optimization_pose_reg = std::clamp(canva.rig_optimization_pose_reg, 0.0f, 1.0f);
            ImGui::InputFloat("Beta reg##rig_opt_beta_reg", &canva.rig_optimization_beta_reg, 0.002f, 0.01f, "%.4f");
            canva.rig_optimization_beta_reg = std::clamp(canva.rig_optimization_beta_reg, 0.0f, 1.0f);
            ImGui::PopItemWidth();

            ImGui::BeginDisabled(!can_optimize_rig);
            if (ImGui::Button("Rig Optimization"))
            {
                std::string err;
                if (!canva_queue_rig_optimization(state, canva, &err))
                {
                    canva.rig_optimization_active = false;
                    canva.rig_optimization_job_running = false;
                    canva.rig_optimization_waiting_uv = false;
                    canva.rig_optimization_status = "Failed";
                    canva.rig_optimization_last_error = err.empty() ? "(unknown error)" : err;
                    console_add_line(state, "Canva: Rig Optimization failed to queue: " + canva.rig_optimization_last_error);
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Text("Status: %s", canva.rig_optimization_status.c_str());
            if (!rig_opt_pose_available && !canva.rig_optimization_active)
                ImGui::TextWrapped("Run Extract Pose first.");
            else if (!rig_opt_have_uv && !rig_opt_can_queue_uv && !canva.rig_optimization_active)
                ImGui::TextWrapped("Sketch2UV is required for the contour mask.");
            if (canva.rig_optimization_waiting_uv)
                ImGui::TextUnformatted("Rig Optimization: predicting UV mask...");
            if (canva.rig_optimization_job_running)
                ImGui::TextUnformatted("Rig Optimization: optimizing differentiable silhouette...");
            if (canva.rig_optimization_last_final_loss > 0.0f)
                ImGui::Text("Loss: %.5f -> %.5f",
                    canva.rig_optimization_last_initial_loss,
                    canva.rig_optimization_last_final_loss);
            if (!canva.rig_optimization_last_output.empty())
                ImGui::TextWrapped("Output: %s", canva.rig_optimization_last_output.c_str());
            if (!canva.rig_optimization_last_error.empty())
                ImGui::TextWrapped("Error: %s", canva.rig_optimization_last_error.c_str());

            ImGui::Separator();
            const bool displacement_camera_available =
                state.scene_view_target_valid
                && state.scene_view_snapshot_completed_id > 0
                && !state.scene_view_snapshot_path.empty();
            std::filesystem::path displacement_loaded_uv_path;
            const bool displacement_have_loaded_uv =
                canva_manual_uv_source_path(state, &displacement_loaded_uv_path);
            const bool displacement_can_queue_uv =
                state.sketch2uv.service != nullptr
                && state.sketch2uv.status == ModelInferenceStatus::Ready;
            const bool can_generate_displacement =
                !canva.generate_displacement_active
                && !canva.generate_iso_displacement_active
                && displacement_camera_available
                && (displacement_have_loaded_uv || displacement_can_queue_uv);
            ImGui::BeginDisabled(!can_generate_displacement);
            if (ImGui::Button("Generate Displacement"))
            {
                std::string err;
                if (!canva_queue_generate_displacement(state, canva, &err))
                {
                    canva.generate_displacement_active = false;
                    canva.generate_displacement_status = "Failed";
                    console_add_line(state, std::string("Canva: Generate Displacement failed to queue: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Text("Status: %s",
                !displacement_camera_available ? "Predict camera view first"
                : !displacement_have_loaded_uv && !displacement_can_queue_uv ? "Need UV"
                : canva.generate_displacement_status.c_str());
            if (canva.generate_displacement_waiting_uv)
                ImGui::TextUnformatted("Generate Displacement: generating UV map...");
            if (canva.generate_displacement_waiting_render)
                ImGui::TextUnformatted("Generate Displacement: projecting offsets...");

            ImGui::SetNextItemWidth(140.0f);
            ImGui::InputFloat(
                "ISO distance threshold##canva_iso_distance",
                &state.iso_distance_threshold,
                0.01f,
                0.05f,
                "%.4f");
            state.iso_distance_threshold =
                std::clamp(state.iso_distance_threshold, 0.0f, 10.0f);

            std::filesystem::path displacement_loaded_iso_path;
            const bool displacement_have_loaded_iso =
                canva_manual_iso_source_path(state, &displacement_loaded_iso_path);
            const bool displacement_can_queue_iso =
                state.sketch2iso.service != nullptr
                && state.sketch2iso.status == ModelInferenceStatus::Ready;
            const bool can_generate_iso_displacement =
                !canva.generate_iso_displacement_active
                && !canva.generate_displacement_active
                && displacement_camera_available
                && (displacement_have_loaded_iso || displacement_can_queue_iso);
            ImGui::BeginDisabled(!can_generate_iso_displacement);
            if (ImGui::Button("Create ISO Displacement Map"))
            {
                std::string error;
                if (!canva_queue_generate_iso_displacement(state, canva, &error))
                {
                    canva.generate_iso_displacement_active = false;
                    canva.generate_iso_displacement_status = "Failed";
                    console_add_line(
                        state,
                        "Canva: Create ISO Displacement Map failed to queue: "
                            + (error.empty() ? std::string("(unknown error)") : error));
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Text(
                "Status: %s",
                !displacement_camera_available ? "Predict camera view first"
                : !displacement_have_loaded_iso && !displacement_can_queue_iso ? "Need ISO"
                : canva.generate_iso_displacement_status.c_str());
            if (canva.generate_iso_displacement_waiting_iso)
                ImGui::TextUnformatted("ISO displacement: generating ISO map...");
            if (canva.generate_iso_displacement_waiting_render)
                ImGui::TextUnformatted("ISO displacement: projecting offsets...");

            ImGui::Separator();
            const bool can_extract_pose =
                !canva.extract_pose_active &&
                state.sketch2rig.service != nullptr &&
                state.sketch2betas.service != nullptr &&
                state.sketch2rig.status == ModelInferenceStatus::Ready &&
                state.sketch2betas.status == ModelInferenceStatus::Ready;
            ImGui::BeginDisabled(!can_extract_pose);
            if (ImGui::Button("Extract Pose"))
            {
                std::string err;
                if (!canva_queue_extract_pose(state, canva, false, false, &err))
                    console_add_line(state, std::string("Canva: Extract Pose failed to queue: ")
                        + (err.empty() ? "(unknown error)" : err));
            }
            ImGui::EndDisabled();

            const bool garment_selected =
                state.selection.garment_index >= 0
                && state.selection.garment_index < (int)state.selection.garments.size();
            const bool can_render_cloth =
                garment_selected
                && !canva.render_cloth_active
                && !canva.generate_cloth_active
                && !state.controls.simulating;
            ImGui::BeginDisabled(!can_render_cloth);
            if (ImGui::Button("Render Cloth"))
            {
                std::string err;
                if (!canva_queue_render_cloth(state, canva, false, &err))
                {
                    canva.render_cloth_active = false;
                    canva.render_cloth_status = "Failed";
                    console_add_line(state, "Canva: Render Cloth failed to queue: "
                        + (err.empty() ? std::string("(unknown error)") : err));
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Fine Render"))
            {
                std::string err;
                if (!canva_queue_render_cloth(state, canva, true, &err))
                {
                    canva.render_cloth_active = false;
                    canva.render_cloth_status = "Failed";
                    console_add_line(state, "Canva: Fine Render failed to queue: "
                        + (err.empty() ? std::string("(unknown error)") : err));
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Text("Status: %s",
                !garment_selected && !canva.render_cloth_active
                    ? "Select cloth mesh"
                    : canva.render_cloth_status.c_str());

            ImGui::SetNextItemWidth(150.0f);
            ImGui::InputFloat(
                "Target detail size",
                &state.cloth_render_target_detail_size,
                0.001f,
                0.005f,
                "%.4f");
            state.cloth_render_target_detail_size =
                std::clamp(state.cloth_render_target_detail_size, 0.0001f, 0.25f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Maximum fine-render mesh edge length in scene units.");

            const bool can_generate_cloth =
                garment_selected &&
                !canva.generate_cloth_active &&
                !canva.render_cloth_active &&
                !state.controls.simulating;
            ImGui::BeginDisabled(!can_generate_cloth);
            if (ImGui::Button("Generate Cloth"))
            {
                std::string err;
                if (!canva_queue_generate_cloth_current_pose(state, canva, &err))
                {
                    canva.generate_cloth_active = false;
                    canva.generate_cloth_status = "Failed";
                    canva.generate_cloth_failed_step = 1;
                    console_add_line(state, std::string("Canva: Generate Cloth failed to queue: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();

            const bool can_full_generation =
                can_extract_pose &&
                garment_selected &&
                !canva.generate_cloth_active &&
                !canva.render_cloth_active &&
                !state.controls.simulating;
            ImGui::BeginDisabled(!can_full_generation);
            if (ImGui::Button("Full Generation"))
            {
                std::string err;
                if (canva_queue_extract_pose(state, canva, true, true, &err))
                {
                    canva.generate_cloth_active = true;
                    canva.generate_cloth_waiting_pose = false;
                    canva.generate_cloth_waiting_simulation = false;
                    canva.generate_cloth_waiting_render = false;
                    canva.generate_cloth_status = "Inference";
                    canva.generate_cloth_failed_step = -1;
                    console_add_line(state, "Canva: Full Generation queued pose extraction");
                }
                else
                {
                    canva.generate_cloth_active = false;
                    canva.generate_cloth_status = "Failed";
                    canva.generate_cloth_failed_step = 0;
                    console_add_line(state, std::string("Canva: Full Generation failed to queue: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
            ImGui::EndDisabled();
            ImGui::Text("Status: %s",
                (!garment_selected && !canva.generate_cloth_active) ? "Select cloth mesh" : canva.generate_cloth_status.c_str());
            if (canva.extract_pose_active)
                ImGui::TextUnformatted("Full Generation: waiting for rig and betas...");
            if (canva.generate_cloth_waiting_pose)
                ImGui::TextUnformatted("Full Generation: waiting for pose reload...");
            if (canva.generate_cloth_waiting_simulation)
            {
                const std::string step = state.controls.sim_stage.empty()
                    ? std::string("waiting for XPBD simulation")
                    : state.controls.sim_stage;
                ImGui::Text("Generate Cloth: %s...", step.c_str());
            }
            if (canva.generate_cloth_waiting_render)
                ImGui::TextUnformatted("Generate Cloth: rendering cloth layer...");

            const bool cloth_layer_selected =
                canva.active_layer >= 0
                && canva.active_layer < (int)canva.layers.size()
                && canva_layer_is_generated_cloth(canva.layers[(size_t)canva.active_layer]);
            std::filesystem::path cloth_fitting_uv_path;
            const bool cloth_fitting_have_uv = canva_current_uv_source_path(state, &cloth_fitting_uv_path);
            const bool cloth_fitting_can_queue_uv =
                state.sketch2uv.service != nullptr
                && state.sketch2uv.status == ModelInferenceStatus::Ready;
            const bool can_fit_cloth =
                cloth_layer_selected
                && (cloth_fitting_have_uv || cloth_fitting_can_queue_uv)
                && !canva.cloth_fitting_active
                && !canva.generate_cloth_active
                && !canva.generate_displacement_active
                && !canva.generate_iso_displacement_active;
            ImGui::BeginDisabled(!can_fit_cloth);
            if (ImGui::Button("cloth Fitting"))
            {
                std::string err;
                if (cloth_fitting_have_uv)
                {
                    if (!canva_run_cloth_fitting(state, canva, renderer, cloth_fitting_uv_path, canva.active_layer, &err))
                    {
                        canva.cloth_fitting_status = "Failed";
                        console_add_line(state, std::string("Canva: cloth Fitting failed: ")
                            + (err.empty() ? "(unknown error)" : err));
                    }
                    else
                    {
                        canva.cloth_fitting_status = "Done";
                    }
                }
                else if (!canva_queue_cloth_fitting_uv(state, canva, canva.layers[(size_t)canva.active_layer], &err))
                {
                    canva.cloth_fitting_active = false;
                    canva.cloth_fitting_status = "Failed";
                    console_add_line(state, std::string("Canva: cloth Fitting failed to queue Sketch2UV: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Text("Status: %s",
                !cloth_layer_selected ? "Select Generated cloth layer"
                : canva.cloth_fitting_active ? canva.cloth_fitting_status.c_str()
                : !cloth_fitting_have_uv && !cloth_fitting_can_queue_uv ? "Need UV"
                : "Ready");
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderFloat("Depth margin", &state.cloth_fitting_depth_margin, 0.0f, 0.02f, "%.5f");
            state.cloth_fitting_depth_margin = std::clamp(state.cloth_fitting_depth_margin, 0.0f, 0.05f);

            if (ImGui::Button("Add layer"))
            {
                const std::string name = std::string("Layer ") + std::to_string(canva.next_layer_number++);
                canva.layers.insert(canva.layers.begin(),
                    canva_make_layer(canva.width, canva.height, canva.next_layer_id++, name));
                canva.active_layer = 0;
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(canva.layers.empty());
            if (ImGui::Button("Remove layer"))
            {
                const int idx = canva.active_layer;
                if (idx >= 0 && idx < (int)canva.layers.size())
                {
                    const int removed_id = canva.layers[(size_t)idx].id;
                    canva_delete_layer_texture(canva.layers[(size_t)idx]);
                    canva.layers.erase(canva.layers.begin() + idx);
                    canva_prune_undo_for_layer(canva.undo, removed_id);
                    canva.active_layer = std::clamp(canva.active_layer, 0, std::max(0, (int)canva.layers.size() - 1));
                    canva_cancel_interactions(canva);
                }
            }
            ImGui::EndDisabled();

            ImGui::BeginDisabled(canva.layers.size() <= 1);
            if (ImGui::Button("Up") && canva.active_layer > 0)
            {
                std::swap(canva.layers[(size_t)canva.active_layer], canva.layers[(size_t)canva.active_layer - 1]);
                canva.active_layer--;
            }
            ImGui::SameLine();
            if (ImGui::Button("Down") && canva.active_layer + 1 < (int)canva.layers.size())
            {
                std::swap(canva.layers[(size_t)canva.active_layer], canva.layers[(size_t)canva.active_layer + 1]);
                canva.active_layer++;
            }
            ImGui::EndDisabled();

            ImGui::Separator();

            // Layer list (top->bottom).
            for (int i = 0; i < (int)canva.layers.size(); ++i)
            {
                CanvaLayer& layer = canva.layers[(size_t)i];

                ImGui::PushID(i);
                ImGui::Checkbox("##vis", &layer.enabled);
                ImGui::SameLine();

                const bool selected = (canva.active_layer == i);
                if (ImGui::Selectable(layer.name.c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick))
                    canva.active_layer = i;

                // Drag-drop reorder (swap-based).
                if (ImGui::BeginDragDropSource())
                {
                    ImGui::SetDragDropPayload("CANVA_LAYER", &i, sizeof(int));
                    ImGui::TextUnformatted(layer.name.c_str());
                    ImGui::EndDragDropSource();
                }
                if (ImGui::BeginDragDropTarget())
                {
                    if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CANVA_LAYER"))
                    {
                        if (payload->DataSize == sizeof(int))
                        {
                            const int from = *(const int*)payload->Data;
                            const int to = i;
                            if (from >= 0 && from < (int)canva.layers.size() && to >= 0 && to < (int)canva.layers.size() && from != to)
                            {
                                std::swap(canva.layers[(size_t)from], canva.layers[(size_t)to]);
                                if (canva.active_layer == from)
                                    canva.active_layer = to;
                                else if (canva.active_layer == to)
                                    canva.active_layer = from;
                            }
                        }
                    }
                    ImGui::EndDragDropTarget();
                }

                ImGui::PopID();
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Controls:");
            ImGui::BulletText("Brush tool: left mouse");
            ImGui::BulletText("Brush size: Shift + drag (on canvas)");
            ImGui::BulletText("Transform tool: Ctrl+T, drag layer (or corner handles to scale)");
            ImGui::BulletText("Undo: Ctrl+Z");
            ImGui::BulletText("Pan: middle mouse drag");
            ImGui::BulletText("Zoom: mouse wheel");
        }
        ImGui::EndChild();
    }
}
