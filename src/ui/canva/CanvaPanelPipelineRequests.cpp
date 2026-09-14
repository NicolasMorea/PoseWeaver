#include "ui/canva/CanvaPanelInternal.h"

#include "ui/canva/CanvaPreprocess.h"
#include "core/Paths.h"
#include "render/Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cerrno>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

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

    bool canva_composite_visible_layers_rgba(
        const CanvaState& canva,
        std::vector<unsigned char>* out_rgba,
        std::string* out_error,
        bool sketch_layers_only)
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

    std::filesystem::path canva_render_frame_dir(int frame_index)
    {
        return std::filesystem::path("renders")
            / ("frame_" + std::to_string(std::max(0, frame_index)));
    }

    std::filesystem::path canva_render_cloth_frame_dir(int frame_index, int cloth_frame_index)
    {
        return canva_render_frame_dir(frame_index)
            / ("cloth_frame_" + std::to_string(std::max(0, cloth_frame_index)));
    }

    int canva_count_cloth_frames_for_sketch(const CanvaState& canva, int sketch_frame_index)
    {
        if (sketch_frame_index >= 0
            && sketch_frame_index < (int)canva.sequence_sketch_frames.size())
            return (int)canva.sequence_sketch_frames[(size_t)sketch_frame_index].cloth_frame_indices.size();

        int count = 0;
        for (const CanvaSequenceTrackFrame& frame : canva.sequence_cloth_frames)
        {
            if (frame.sketch_frame_index == sketch_frame_index)
                ++count;
        }
        return count;
    }

    void canva_prepare_cloth_output_dirs(
        AppState& state,
        CanvaState& canva,
        int sketch_frame_index,
        int cloth_frame_index)
    {
        const std::filesystem::path frame_dir = canva_render_frame_dir(sketch_frame_index);
        const std::filesystem::path cloth_frame_dir =
            canva_render_cloth_frame_dir(sketch_frame_index, cloth_frame_index);
        canva.sequence_current_frame_dir = frame_dir;
        canva.sequence_current_cloth_frame_dir = cloth_frame_dir;
        state.cloth_render_output_dir = cloth_frame_dir.string();
        state.cloth_fitting_output_dir = cloth_frame_dir.string();
    }

    void canva_delete_render_frame_dirs()
    {
        const std::filesystem::path renders_dir = "renders";
        std::error_code ec;
        if (!std::filesystem::exists(renders_dir, ec))
            return;

        for (const std::filesystem::directory_entry& entry
             : std::filesystem::directory_iterator(renders_dir, ec))
        {
            if (ec)
                return;
            if (!entry.is_directory(ec))
                continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind("frame_", 0) != 0)
                continue;
            std::filesystem::remove_all(entry.path(), ec);
            ec.clear();
        }
    }

     bool canva_save_visible_layers_png(const CanvaState& canva, std::filesystem::path* out_path, std::string* out_error)
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

     bool canva_save_model_input_png(
        const CanvaState& canva,
        std::filesystem::path* out_path,
        std::string* out_error)
    {
        return canva_save_model_input_png_to_dir(
            canva,
            canva_render_frame_dir(0),
            "canva_model_input",
            out_path,
            out_error);
    }

    bool canva_save_model_input_png_to_dir(
        const CanvaState& canva,
        const std::filesystem::path& output_dir,
        const std::string& stem,
        std::filesystem::path* out_path,
        std::string* out_error)
    {
        std::vector<unsigned char> rgba;
        if (!canva_composite_visible_layers_rgba(canva, &rgba, out_error))
            return false;

        const std::filesystem::path dir = output_dir.empty()
            ? canva_render_frame_dir(0)
            : output_dir;
        const std::string filename = stem.empty() ? "canva_model_input" : stem;
        return save_normalized_model_input_png_to_path(
            canva.width,
            canva.height,
            rgba,
            dir / (filename + ".png"),
            out_path,
            out_error);
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

    bool canva_queue_render_cloth(
        AppState& state,
        CanvaState& canva,
        bool fine,
        bool fit_after,
        bool refit_after,
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
        if (refit_after && !canva_has_cloth_refit_maps(canva))
        {
            if (out_error)
                *out_error = "fit a cloth layer once to create reusable fitting maps";
            return false;
        }

        canva.render_cloth_active = true;
        canva.render_cloth_fine = fine;
        canva.render_cloth_fit_after = fit_after;
        canva.render_cloth_refit_after = refit_after;
        canva.render_cloth_model_input_path = canva_cloth_render_alignment_input(state, canva);
        canva.render_cloth_status = fine ? "Refining" : "Rendering";

        state.show_garment_mesh = true;
        state.cloth_render_fine = fine;
        state.cloth_render_use_current_camera = true;
        state.cloth_render_stroke_reference_path = canva.render_cloth_model_input_path.string();
        state.cloth_render_camera_rig_path.clear();
        state.cloth_render_target_detail_size =
            std::clamp(state.cloth_render_target_detail_size, 0.0001f, 0.25f);
        state.cloth_render_path.clear();
        canva_prepare_cloth_output_dirs(state, canva, 0, 0);
        canva.render_cloth_request_id = ++state.cloth_render_request_id;

        console_add_line(state,
            refit_after ? "Canva: queued cloth render with cached-map refit"
            : fine ? "Canva: queued fine cloth render"
            : "Canva: queued cloth render");
        return true;
    }

     bool canva_queue_extract_pose_custom_from_sketch(
        AppState& state,
        CanvaState& canva,
        const std::filesystem::path& sketch_path,
        bool generate_cloth_after,
        bool hide_rig_layer,
        std::string* out_error)
    {
        if (state.sketch2rig.service == nullptr || state.sketch2betas.service == nullptr)
        {
            if (out_error)
                *out_error = "custom rig/betas workers are not available";
            return false;
        }
        if (state.sketch2rig.status != ModelInferenceStatus::Ready
            || state.sketch2betas.status != ModelInferenceStatus::Ready)
        {
            if (out_error)
                *out_error = "custom rig/betas workers are not ready";
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
        canva.extract_pose_using_nlf = false;
        canva.extract_pose_nlf_fallback_allowed = false;
        canva.extract_pose_direct_nlf = false;
        canva.bone_prediction_debug_mode = 0;
        canva.bone_prediction_nlf_confidence = -1.0f;
        canva.bone_prediction_debug_status = "Custom pipeline";
        canva.extract_pose_rig_request_id = rig_request_id;
        canva.extract_pose_betas_request_id = betas_request_id;
        canva.extract_pose_model_input_path = sketch_path;
        canva.extract_pose_rig_path = rig_output_path;
        canva.extract_pose_betas_path = betas_output_path;
        if (generate_cloth_after)
        {
            canva_generate_cloth_reset_timings(canva, false);
            canva_generate_cloth_start_step(canva, GenerateClothStepId::Inference);
            canva.generate_cloth_model_input_path = sketch_path;
            canva.generate_cloth_rig_json_path = rig_output_path;
            canva.generate_cloth_status = "Inference";
            canva.generate_cloth_failed_step = -1;
            canva.generate_cloth_extract_rig_start = std::chrono::steady_clock::now();
            canva.generate_cloth_extract_betas_start = canva.generate_cloth_extract_rig_start;
        }
        state.scene_view_snap_to_target_request_id++;

        console_add_line(state, std::string("Canva: Extract Pose queued from sketch: ") + sketch_path.string());
        return true;
    }

     bool canva_queue_extract_pose(
        AppState& state,
        CanvaState& canva,
        bool generate_cloth_after,
        bool hide_rig_layer,
        std::string* out_error,
        const std::filesystem::path& output_dir)
    {
        std::filesystem::path sketch_path;
        std::string err;
        const bool saved = output_dir.empty()
            ? canva_save_model_input_png(canva, &sketch_path, &err)
            : canva_save_model_input_png_to_dir(
                canva,
                output_dir,
                "canva_model_input",
                &sketch_path,
                &err);
        if (!saved)
        {
            canva.bone_prediction_debug_mode = 4;
            canva.bone_prediction_debug_status = "Bone prediction preprocessing failed";
            if (out_error)
                *out_error = err.empty() ? "preprocessing failed" : err;
            return false;
        }

        const int preferred = std::clamp(state.canva_preferred_bone_prediction, 0, 1);
        if (preferred == 0)
            return canva_queue_extract_pose_custom_from_sketch(
                state,
                canva,
                sketch_path,
                generate_cloth_after,
                hide_rig_layer,
                out_error);

        if (state.nlf.service == nullptr)
        {
            canva.bone_prediction_debug_mode = 4;
            canva.bone_prediction_debug_status = "NLF unavailable";
            if (out_error)
                *out_error = "NLF worker is not available";
            return false;
        }
        if (state.nlf.status != ModelInferenceStatus::Ready)
        {
            canva.bone_prediction_debug_mode = 4;
            canva.bone_prediction_debug_status = "NLF not ready";
            if (out_error)
                *out_error = "NLF worker is not ready";
            return false;
        }

        const std::filesystem::path nlf_output_path =
            sketch_path.parent_path() / "sketch_nlf.json";
        const std::uint64_t nlf_request_id = state.nlf.next_request_id++;
        std::string nlf_request_err;
        if (!state.nlf.service->request_inference(
                nlf_request_id,
                sketch_path,
                nlf_output_path,
                &nlf_request_err))
        {
            canva.bone_prediction_debug_mode = 4;
            canva.bone_prediction_debug_status = "NLF queue failed";
            if (out_error)
                *out_error = nlf_request_err.empty() ? "NLF request failed" : nlf_request_err;
            return false;
        }

        state.nlf.last_sketch_path = sketch_path.string();
        state.nlf.status = ModelInferenceStatus::Busy;
        state.nlf.active_request_id = nlf_request_id;
        state.nlf.message = "Predicting bones and betas with NLF...";

        canva.extract_pose_active = true;
        canva.extract_pose_rig_done = false;
        canva.extract_pose_betas_done = false;
        canva.extract_pose_hide_rig_layer = hide_rig_layer;
        canva.extract_pose_generate_cloth_after = generate_cloth_after;
        canva.extract_pose_using_nlf = true;
        canva.extract_pose_nlf_fallback_allowed = true;
        canva.extract_pose_direct_nlf = false;
        canva.bone_prediction_debug_mode = 3;
        canva.bone_prediction_nlf_confidence = -1.0f;
        canva.bone_prediction_debug_status = "NLF running";
        canva.extract_pose_rig_request_id = nlf_request_id;
        canva.extract_pose_betas_request_id = nlf_request_id;
        canva.extract_pose_model_input_path = sketch_path;
        canva.extract_pose_rig_path = nlf_output_path;
        canva.extract_pose_betas_path = nlf_output_path;
        if (generate_cloth_after)
        {
            canva_generate_cloth_reset_timings(canva, false);
            canva_generate_cloth_start_step(canva, GenerateClothStepId::Inference);
            canva.generate_cloth_model_input_path = sketch_path;
            canva.generate_cloth_rig_json_path = nlf_output_path;
            canva.generate_cloth_status = "Inference";
            canva.generate_cloth_failed_step = -1;
            canva.generate_cloth_extract_rig_start = std::chrono::steady_clock::now();
            canva.generate_cloth_extract_betas_start = canva.generate_cloth_extract_rig_start;
        }
        state.scene_view_snap_to_target_request_id++;

        console_add_line(state, std::string("Canva: Extract Pose queued with NLF from sketch: ") + sketch_path.string());
        return true;
    }

     bool canva_queue_generate_displacement(
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

        if (state.sketch2uv.service == nullptr
            || (state.sketch2uv.status != ModelInferenceStatus::NotStarted
                && state.sketch2uv.status != ModelInferenceStatus::Ready))
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

     bool canva_queue_generate_iso_displacement(
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

    bool canva_queue_cloth_fitting_predictions_from_sketch(
        AppState& state,
        CanvaState& canva,
        const std::filesystem::path& sketch_path,
        const std::filesystem::path& rig_json_path,
        int source_layer_id,
        bool wait_for_source_layer,
        std::string* out_error)
    {
        if (canva.cloth_fitting_active)
        {
            if (out_error)
                *out_error = "cloth fitting is already active";
            return false;
        }
        if (state.sketch2iso.service == nullptr || state.sketch2iso.status != ModelInferenceStatus::Ready)
        {
            if (out_error)
                *out_error = "Sketch2ISO is not ready";
            return false;
        }
        if (state.sketch2depth.service == nullptr || state.sketch2depth.status != ModelInferenceStatus::Ready)
        {
            if (out_error)
                *out_error = "Sketch2Depth is not ready";
            return false;
        }
        if (sketch_path.empty())
        {
            if (out_error)
                *out_error = "missing sketch path for cloth fitting";
            return false;
        }
        const int fitting_sketch_index =
            (canva.sequence_generating && canva.sequence_active_sketch_index >= 0)
                ? canva.sequence_active_sketch_index
                : 0;
        canva_prepare_cloth_output_dirs(
            state,
            canva,
            fitting_sketch_index,
            canva_count_cloth_frames_for_sketch(canva, fitting_sketch_index));

        const std::filesystem::path iso_output_path =
            sketch_path.parent_path() / "sketch_iso_cloth_fitting.npz";
        const std::filesystem::path depth_output_path =
            sketch_path.parent_path() / "sketch_depth_cloth_fitting.json";
        const std::uint64_t iso_request_id = state.sketch2iso.next_request_id++;
        const std::uint64_t depth_request_id = state.sketch2depth.next_request_id++;

        std::string request_err;
        if (!state.sketch2depth.service->request_inference(
                depth_request_id, sketch_path, depth_output_path, &request_err))
        {
            if (out_error)
                *out_error = request_err.empty() ? "failed to queue Sketch2Depth" : request_err;
            return false;
        }
        if (!state.sketch2iso.service->request_inference(
                iso_request_id, sketch_path, iso_output_path, &request_err))
        {
            canva.ignore_sketch2depth_request_id = depth_request_id;
            if (out_error)
                *out_error = request_err.empty() ? "failed to queue Sketch2ISO" : request_err;
            return false;
        }

        state.sketch2iso.last_sketch_path = sketch_path.string();
        state.sketch2iso.status = ModelInferenceStatus::Busy;
        state.sketch2iso.active_request_id = iso_request_id;
        state.sketch2iso.message = "Processing sketch ISO values for cloth fitting...";
        state.sketch2depth.last_sketch_path = sketch_path.string();
        state.sketch2depth.status = ModelInferenceStatus::Busy;
        state.sketch2depth.active_request_id = depth_request_id;
        state.sketch2depth.message = "Predicting body depth for cloth fitting...";

        canva.cloth_fitting_active = true;
        canva.cloth_fitting_waiting_iso = true;
        canva.cloth_fitting_waiting_depth = true;
        canva.cloth_fitting_iso_request_id = iso_request_id;
        canva.cloth_fitting_depth_request_id = depth_request_id;
        canva.cloth_fitting_model_input_path = sketch_path;
        canva.cloth_fitting_iso_path.clear();
        canva.cloth_fitting_depth_path.clear();
        canva.cloth_fitting_rig_json_path = rig_json_path;
        canva.cloth_fitting_source_layer_id = source_layer_id;
        canva.cloth_fitting_waiting_source_layer = wait_for_source_layer;
        canva.cloth_fitting_status = "Predicting ISO and depth";
        canva_reset_cloth_fitting_timings(canva);
        canva.cloth_fitting_iso_start = std::chrono::steady_clock::now();
        canva.cloth_fitting_depth_start = canva.cloth_fitting_iso_start;
        if (canva.generate_cloth_timing_started && canva.generate_cloth_include_fitting_steps)
            canva_generate_cloth_start_step(canva, GenerateClothStepId::FittingInference);

        console_add_line(state, std::string("Canva: cloth Fitting queued Sketch2ISO and Sketch2Depth from sketch: ")
            + sketch_path.string());
        return true;
    }

     bool canva_queue_cloth_fitting_predictions(
        AppState& state,
        CanvaState& canva,
        const CanvaLayer& source_layer,
        std::string* out_error)
    {
        const int source_layer_index = canva_find_layer_index_by_id(canva, source_layer.id);
        if (source_layer_index < 0)
        {
            if (out_error)
                *out_error = "selected cloth layer is no longer available";
            return false;
        }
        canva.layers[(size_t)source_layer_index].enabled = false;

        std::filesystem::path inference_dir = canva.generate_cloth_model_input_path.parent_path();
        if (inference_dir.empty())
            inference_dir = canva_render_frame_dir(0);

        std::filesystem::path sketch_path;
        std::string err;
        if (!canva_save_model_input_png_to_dir(
                canva,
                inference_dir,
                "canva_model_input",
                &sketch_path,
                &err))
        {
            if (out_error)
                *out_error = err.empty() ? "preprocessing failed" : err;
            return false;
        }

        return canva_queue_cloth_fitting_predictions_from_sketch(
            state,
            canva,
            sketch_path,
            canva.generate_cloth_rig_json_path,
            source_layer.id,
            false,
            out_error);
    }

     bool canva_queue_generate_cloth_current_pose(
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
        canva.generate_cloth_rig_json_path.clear();
        canva.generate_cloth_pose_name.clear();
        canva.generate_cloth_status = "Simulating";
        canva.generate_cloth_failed_step = -1;
        canva.generate_cloth_fit_after_render = false;
        canva_generate_cloth_reset_timings(canva, false);
        canva_generate_cloth_finish_step(canva, GenerateClothStepId::Inference, true);
        canva_generate_cloth_finish_step(canva, GenerateClothStepId::Pose, true);
        canva_generate_cloth_start_step(canva, GenerateClothStepId::Sequence);
        canva.generate_cloth_last_sim_phase = 0;

        state.show_smpl = true;
        state.show_smpl_character = true;
        state.show_garment_mesh = true;
        state.controls.current_frame = 0;
        state.controls.playing = false;
        state.controls.simulating = true;
        state.cloth_simulate_play_on_finish = false;
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

     bool canva_current_predicted_pose_path(
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

     bool canva_start_rig_optimization(
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

     bool canva_queue_rig_optimization(
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

        if (state.sketch2uv.service == nullptr
            || (state.sketch2uv.status != ModelInferenceStatus::NotStarted
                && state.sketch2uv.status != ModelInferenceStatus::Ready))
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

     bool canva_apply_rig_optimization_result(
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

}
