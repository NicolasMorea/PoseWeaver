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

namespace poseweaver::ui_canva
{
    static void canva_prediction_from_cloth_transform(
        const CanvaLayer& cloth_layer,
        const std::filesystem::path& model_input_path,
        float* out_scale,
        float* out_offset_x,
        float* out_offset_y)
    {
        ImVec2 prediction_offset(0.0f, 0.0f);
        float prediction_scale = 1.0f;

        if (!model_input_path.empty())
        {
            CanvaPreprocessTransform transform;
            std::string ignored_error;
            if (canva_load_preprocess_transform(model_input_path, &transform, &ignored_error))
            {
                prediction_scale = 1.0f / std::max(1e-6f, transform.scale);
                prediction_offset = ImVec2(
                    transform.src_center.x - transform.dst_center.x * prediction_scale,
                    transform.src_center.y - transform.dst_center.y * prediction_scale);
            }
        }

        prediction_scale = std::max(1e-6f, prediction_scale);
        *out_scale = cloth_layer.scale / prediction_scale;
        *out_offset_x = (cloth_layer.offset.x - prediction_offset.x) / prediction_scale;
        *out_offset_y = (cloth_layer.offset.y - prediction_offset.y) / prediction_scale;
    }

     bool canva_layer_is_generated_cloth(const CanvaLayer& layer)
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
        std::vector<unsigned char> warped_pixels,
        std::vector<unsigned char> debug_pixels,
        std::vector<unsigned char> warped_debug_pixels,
        std::vector<unsigned char> occlusion_pixels,
        std::vector<unsigned char> scene_occlusion_pixels,
        std::vector<unsigned char> aligned_predicted_depth_occlusion_pixels,
        std::vector<unsigned char> iso_occlusion_pixels,
        std::vector<unsigned char> scene_debug_pixels,
        std::vector<unsigned char> deformation_pixels,
        std::vector<unsigned char> close_to_body_pixels,
        bool include_hidden_output_layers,
        std::string* out_error)
    {
        if ((int)pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "fitted pixel buffer is too small";
            return false;
        }
        if ((int)warped_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "warped fitted pixel buffer is too small";
            return false;
        }
        if ((int)debug_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "debug pixel buffer is too small";
            return false;
        }
        if ((int)warped_debug_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "warped debug pixel buffer is too small";
            return false;
        }
        if ((int)occlusion_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "occlusion pixel buffer is too small";
            return false;
        }
        if ((int)scene_occlusion_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "true-depth occlusion pixel buffer is too small";
            return false;
        }
        if ((int)aligned_predicted_depth_occlusion_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "aligned predicted-depth occlusion pixel buffer is too small";
            return false;
        }
        if ((int)iso_occlusion_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "ISO occlusion pixel buffer is too small";
            return false;
        }
        if ((int)scene_debug_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "3D scene debug pixel buffer is too small";
            return false;
        }
        if ((int)deformation_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "deformation map pixel buffer is too small";
            return false;
        }
        if ((int)close_to_body_pixels.size() < canva.width * canva.height * 4)
        {
            if (out_error)
                *out_error = "close-to-body pixel buffer is too small";
            return false;
        }
        const ImVec2 source_offset = source_layer.offset;
        const float source_scale = source_layer.scale;
        const int source_layer_id = source_layer.id;
        const bool remove_hidden_source_layer =
            !source_layer.enabled && canva_layer_is_generated_cloth(source_layer);

        CanvaLayer warped_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "Modified cloth (warped)");
        const int warped_layer_id = warped_layer.id;
        warped_layer.kind = CanvaLayerKind::ClothFittingWarped;
        warped_layer.pixels = std::move(warped_pixels);
        warped_layer.offset = source_offset;
        warped_layer.scale = source_scale;
        warped_layer.enabled = true;
        warped_layer.dirty = true;
        warped_layer.bounds_dirty = true;

        canva.layers.insert(canva.layers.begin(), std::move(warped_layer));

        if (include_hidden_output_layers)
        {
            auto add_hidden_layer = [&](std::vector<unsigned char> layer_pixels, const char* name, CanvaLayerKind kind)
            {
                CanvaLayer layer = canva_make_layer(
                    canva.width,
                    canva.height,
                    canva.next_layer_id++,
                    name);
                layer.kind = kind;
                layer.pixels = std::move(layer_pixels);
                layer.offset = source_offset;
                layer.scale = source_scale;
                layer.enabled = false;
                layer.dirty = true;
                layer.bounds_dirty = true;
                const auto insert_pos = canva.layers.begin() + std::min<int>(1, (int)canva.layers.size());
                canva.layers.insert(insert_pos, std::move(layer));
            };

            add_hidden_layer(std::move(pixels), "Modified cloth", CanvaLayerKind::ClothFitting);
            add_hidden_layer(std::move(debug_pixels), "Cloth fitting debug", CanvaLayerKind::ClothFittingDebug);
            add_hidden_layer(std::move(warped_debug_pixels), "Cloth fitting debug (warped)", CanvaLayerKind::ClothFittingWarpedDebug);
            add_hidden_layer(std::move(occlusion_pixels), "Cloth fitting occlusion", CanvaLayerKind::ClothFittingOcclusionDebug);
            add_hidden_layer(std::move(scene_occlusion_pixels), "Cloth fitting true-depth visibility", CanvaLayerKind::ClothFittingSceneOcclusionDebug);
            add_hidden_layer(std::move(aligned_predicted_depth_occlusion_pixels), "Cloth fitting predicted-depth visibility", CanvaLayerKind::ClothFittingAlignedPredictedDepthOcclusionDebug);
            add_hidden_layer(std::move(iso_occlusion_pixels), "Cloth fitting ISO visibility", CanvaLayerKind::ClothFittingIsoOcclusionDebug);
            add_hidden_layer(std::move(scene_debug_pixels), "Cloth fitting 3D scene debug", CanvaLayerKind::ClothFittingSceneDebug);
            add_hidden_layer(std::move(deformation_pixels), "Cloth fitting deformation", CanvaLayerKind::ClothFittingDeformation);
            add_hidden_layer(std::move(close_to_body_pixels), "Cloth fitting close to body", CanvaLayerKind::ClothFittingCloseToBodyDebug);
        }

        if (remove_hidden_source_layer)
        {
            for (int i = (int)canva.layers.size() - 1; i >= 0; --i)
            {
                CanvaLayer& layer = canva.layers[(size_t)i];
                if (layer.id != source_layer_id)
                    continue;
                canva_delete_layer_texture(layer);
                canva_prune_undo_for_layer(canva.undo, source_layer_id);
                canva.layers.erase(canva.layers.begin() + i);
                break;
            }
        }

        canva.active_layer = canva_find_layer_index_by_id(canva, warped_layer_id);
        if (canva.active_layer < 0)
            canva.active_layer = 0;
        canva_cancel_interactions(canva);
        return true;
    }

    static bool canva_run_cloth_fitting(
        AppState& state,
        CanvaState& canva,
        Renderer& renderer,
        const std::filesystem::path& iso_npz_path,
        const std::filesystem::path& depth_prediction_path,
        const std::filesystem::path& model_input_path,
        const std::filesystem::path& rig_json_path,
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
        if (iso_npz_path.empty())
        {
            if (out_error)
                *out_error = "missing ISO source";
            return false;
        }
        if (depth_prediction_path.empty())
        {
            if (out_error)
                *out_error = "missing depth prediction";
            return false;
        }

        std::vector<unsigned char> fitted;
        std::vector<unsigned char> warped_fitted;
        std::vector<unsigned char> debug;
        std::vector<unsigned char> warped_debug;
        std::vector<unsigned char> occlusion;
        std::vector<unsigned char> scene_occlusion;
        std::vector<unsigned char> aligned_predicted_depth_occlusion;
        std::vector<unsigned char> iso_occlusion;
        std::vector<unsigned char> scene_debug;
        std::vector<unsigned char> deformation;
        std::vector<unsigned char> close_to_body;
        int tested_count = 0;
        int removed_count = 0;
        float prediction_from_cloth_scale = 1.0f;
        float prediction_from_cloth_offset_x = 0.0f;
        float prediction_from_cloth_offset_y = 0.0f;
        canva_prediction_from_cloth_transform(
            source,
            model_input_path,
            &prediction_from_cloth_scale,
            &prediction_from_cloth_offset_x,
            &prediction_from_cloth_offset_y);
        std::string fit_err;
        const bool timing_fitting =
            canva.generate_cloth_timing_started
            && canva.generate_cloth_include_fitting_steps;
        const auto renderer_start = std::chrono::steady_clock::now();
        if (!renderer.fit_cloth_render_to_smpl_visibility(
                state,
                source.pixels,
                canva.width,
                canva.height,
                iso_npz_path.string(),
                depth_prediction_path.string(),
                rig_json_path.string(),
                prediction_from_cloth_scale,
                prediction_from_cloth_offset_x,
                prediction_from_cloth_offset_y,
                fitted,
                warped_fitted,
                debug,
                warped_debug,
                occlusion,
                scene_occlusion,
                aligned_predicted_depth_occlusion,
                iso_occlusion,
                scene_debug,
                deformation,
                close_to_body,
                tested_count,
                removed_count,
                &fit_err))
        {
            if (out_error)
                *out_error = fit_err.empty() ? "renderer fitting failed" : fit_err;
            return false;
        }
        if (timing_fitting)
            canva_generate_cloth_append_timed_detail(
                canva,
                GenerateClothStepId::Fitting,
                "Renderer depth/visibility",
                renderer_start);

        std::string add_err;
        const auto layer_start = std::chrono::steady_clock::now();
        const bool include_hidden_output_layers =
            !canva.sequence_generating
            && !canva.sequence_waiting_render;
        if (!canva_add_cloth_fitting_layers(
                canva,
                source,
                std::move(fitted),
                std::move(warped_fitted),
                std::move(debug),
                std::move(warped_debug),
                std::move(occlusion),
                std::move(scene_occlusion),
                std::move(aligned_predicted_depth_occlusion),
                std::move(iso_occlusion),
                std::move(scene_debug),
                std::move(deformation),
                std::move(close_to_body),
                include_hidden_output_layers,
                &add_err))
        {
            if (out_error)
                *out_error = add_err.empty() ? "failed to create output layer" : add_err;
            return false;
        }
        if (timing_fitting)
        {
            canva_generate_cloth_append_timed_detail(
                canva,
                GenerateClothStepId::Fitting,
                "Canvas layer creation",
                layer_start);
            canva_generate_cloth_append_detail(
                canva,
                GenerateClothStepId::Fitting,
                "Pixels: tested=" + std::to_string(tested_count)
                    + ", removed=" + std::to_string(removed_count));
        }

        console_add_line(state,
            "Canva: cloth Fitting added warped cloth layer (saved debug files, removed="
                + std::to_string(removed_count)
                + ", tested=" + std::to_string(tested_count) + ")");
        return true;
    }

     void canva_finish_cloth_fitting_if_ready(
        AppState& state,
        CanvaState& canva,
        Renderer& renderer)
    {
        if (!canva.cloth_fitting_active
            || canva.cloth_fitting_waiting_iso
            || canva.cloth_fitting_waiting_depth)
            return;

        canva.cloth_fitting_status = "Depth testing";
        if (canva.generate_cloth_timing_started && canva.generate_cloth_include_fitting_steps)
        {
            canva_generate_cloth_finish_step(canva, GenerateClothStepId::FittingInference);
            canva_generate_cloth_start_step(canva, GenerateClothStepId::Fitting);
        }
        const int source_idx = canva_find_layer_index_by_id(canva, canva.cloth_fitting_source_layer_id);
        std::string fit_err;
        if (canva_run_cloth_fitting(
                state,
                canva,
                renderer,
                canva.cloth_fitting_iso_path,
                canva.cloth_fitting_depth_path,
                canva.cloth_fitting_model_input_path,
                canva.cloth_fitting_rig_json_path,
                source_idx,
                &fit_err))
        {
            canva.cloth_fitting_active = false;
            canva.cloth_fitting_status = "Done";
            if (canva.sequence_waiting_render || canva.sequence_generating || canva.sequence_cloth_frames.empty())
                canva_sequence_capture_current_frame(state, canva);
            if (canva.generate_cloth_timing_started && canva.generate_cloth_include_fitting_steps)
            {
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Done";
                canva.generate_cloth_failed_step = -1;
                canva_generate_cloth_finish_step(canva, GenerateClothStepId::Fitting);
                canva_generate_cloth_finish_pipeline(canva);
            }
        }
        else
        {
            canva.cloth_fitting_active = false;
            canva.cloth_fitting_status = "Failed";
            if (canva.generate_cloth_timing_started && canva.generate_cloth_include_fitting_steps)
            {
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Failed";
                canva.generate_cloth_failed_step = generate_cloth_step_index(GenerateClothStepId::Fitting);
                canva_generate_cloth_finish_step(canva, GenerateClothStepId::Fitting);
                canva_generate_cloth_finish_pipeline(canva);
            }
            console_add_line(state, std::string("Canva: cloth Fitting failed: ")
                + (fit_err.empty() ? "(unknown error)" : fit_err));
        }
    }

    bool canva_queue_cloth_fitting_cached_predictions(
        AppState& state,
        CanvaState& canva,
        Renderer& renderer,
        const CanvaLayer& source_layer,
        const std::filesystem::path& iso_path,
        const std::filesystem::path& depth_path,
        const std::filesystem::path& model_input_path,
        const std::filesystem::path& rig_json_path,
        std::string* out_error)
    {
        if (canva.cloth_fitting_active)
        {
            if (out_error)
                *out_error = "cloth fitting is already active";
            return false;
        }
        if (iso_path.empty())
        {
            if (out_error)
                *out_error = "missing cached ISO prediction";
            return false;
        }
        if (depth_path.empty())
        {
            if (out_error)
                *out_error = "missing cached depth prediction";
            return false;
        }
        if (state.cloth_fitting_output_dir.empty())
        {
            const int sketch_dir_index = std::max(0, canva.sequence_active_sketch_index);
            canva_prepare_cloth_output_dirs(
                state,
                canva,
                sketch_dir_index,
                canva_count_cloth_frames_for_sketch(canva, sketch_dir_index));
        }

        canva.cloth_fitting_active = true;
        canva.cloth_fitting_waiting_iso = false;
        canva.cloth_fitting_waiting_depth = false;
        canva.cloth_fitting_iso_request_id = 0;
        canva.cloth_fitting_depth_request_id = 0;
        canva.cloth_fitting_model_input_path = model_input_path;
        canva.cloth_fitting_iso_path = iso_path;
        canva.cloth_fitting_depth_path = depth_path;
        canva.cloth_fitting_rig_json_path = rig_json_path;
        canva.cloth_fitting_source_layer_id = source_layer.id;
        canva.cloth_fitting_status = "Depth testing";

        canva_finish_cloth_fitting_if_ready(state, canva, renderer);
        if (canva.cloth_fitting_status != "Done")
        {
            if (out_error)
                *out_error = "cached cloth fitting failed";
            return false;
        }
        return true;
    }

}
