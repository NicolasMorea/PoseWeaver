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
        const ImVec2 source_offset = source_layer.offset;
        const float source_scale = source_layer.scale;

        CanvaLayer layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "Modified cloth (no warp)");
        layer.kind = CanvaLayerKind::ClothFitting;
        layer.pixels = std::move(pixels);
        layer.offset = source_offset;
        layer.scale = source_scale;
        layer.enabled = false;
        layer.dirty = true;
        layer.bounds_dirty = true;

        canva.layers.insert(canva.layers.begin(), std::move(layer));

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

        CanvaLayer debug_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "cloth fitting debug (ISO mapping colors)");
        debug_layer.kind = CanvaLayerKind::ClothFittingDebug;
        debug_layer.pixels = std::move(debug_pixels);
        debug_layer.offset = source_offset;
        debug_layer.scale = source_scale;
        debug_layer.enabled = false;
        debug_layer.dirty = true;
        debug_layer.bounds_dirty = true;

        canva.layers.insert(canva.layers.begin(), std::move(debug_layer));

        CanvaLayer warped_debug_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "cloth Fitting Debug (warped)");
        warped_debug_layer.kind = CanvaLayerKind::ClothFittingWarpedDebug;
        warped_debug_layer.pixels = std::move(warped_debug_pixels);
        warped_debug_layer.offset = source_offset;
        warped_debug_layer.scale = source_scale;
        warped_debug_layer.enabled = false;
        warped_debug_layer.dirty = true;
        warped_debug_layer.bounds_dirty = true;

        canva.layers.insert(canva.layers.begin(), std::move(warped_debug_layer));

        CanvaLayer occlusion_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "cloth occlusion (final colors)");
        occlusion_layer.kind = CanvaLayerKind::ClothFittingOcclusionDebug;
        occlusion_layer.pixels = std::move(occlusion_pixels);
        occlusion_layer.offset = source_offset;
        occlusion_layer.scale = source_scale;
        occlusion_layer.enabled = false;
        occlusion_layer.dirty = true;
        occlusion_layer.bounds_dirty = true;

        canva.layers.insert(canva.layers.begin(), std::move(occlusion_layer));

        CanvaLayer scene_occlusion_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "cloth occlusion (true 3D depth)");
        scene_occlusion_layer.kind = CanvaLayerKind::ClothFittingSceneOcclusionDebug;
        scene_occlusion_layer.pixels = std::move(scene_occlusion_pixels);
        scene_occlusion_layer.offset = source_offset;
        scene_occlusion_layer.scale = source_scale;
        scene_occlusion_layer.enabled = false;
        scene_occlusion_layer.dirty = true;
        scene_occlusion_layer.bounds_dirty = true;

        canva.layers.insert(canva.layers.begin(), std::move(scene_occlusion_layer));

        CanvaLayer aligned_predicted_depth_occlusion_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "cloth occlusion (aligned predicted depth)");
        aligned_predicted_depth_occlusion_layer.kind =
            CanvaLayerKind::ClothFittingAlignedPredictedDepthOcclusionDebug;
        aligned_predicted_depth_occlusion_layer.pixels =
            std::move(aligned_predicted_depth_occlusion_pixels);
        aligned_predicted_depth_occlusion_layer.offset = source_offset;
        aligned_predicted_depth_occlusion_layer.scale = source_scale;
        aligned_predicted_depth_occlusion_layer.enabled = false;
        aligned_predicted_depth_occlusion_layer.dirty = true;
        aligned_predicted_depth_occlusion_layer.bounds_dirty = true;
        canva.layers.insert(canva.layers.begin(), std::move(aligned_predicted_depth_occlusion_layer));

        CanvaLayer iso_occlusion_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "cloth occlusion (ISO mapping only)");
        iso_occlusion_layer.kind = CanvaLayerKind::ClothFittingIsoOcclusionDebug;
        iso_occlusion_layer.pixels = std::move(iso_occlusion_pixels);
        iso_occlusion_layer.offset = source_offset;
        iso_occlusion_layer.scale = source_scale;
        iso_occlusion_layer.enabled = false;
        iso_occlusion_layer.dirty = true;
        iso_occlusion_layer.bounds_dirty = true;
        canva.layers.insert(canva.layers.begin(), std::move(iso_occlusion_layer));

        CanvaLayer scene_debug_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "cloth Fitting 3D Scene Debug");
        scene_debug_layer.kind = CanvaLayerKind::ClothFittingSceneDebug;
        scene_debug_layer.pixels = std::move(scene_debug_pixels);
        scene_debug_layer.offset = source_offset;
        scene_debug_layer.scale = source_scale;
        scene_debug_layer.enabled = false;
        scene_debug_layer.dirty = true;
        scene_debug_layer.bounds_dirty = true;

        canva.layers.insert(canva.layers.begin(), std::move(scene_debug_layer));

        CanvaLayer deformation_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "deformation map");
        deformation_layer.kind = CanvaLayerKind::ClothFittingDeformation;
        deformation_layer.pixels = std::move(deformation_pixels);
        deformation_layer.offset = source_offset;
        deformation_layer.scale = source_scale;
        deformation_layer.enabled = false;
        deformation_layer.dirty = true;
        deformation_layer.bounds_dirty = true;
        canva.layers.insert(canva.layers.begin(), std::move(deformation_layer));

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
            "Canva: cloth Fitting added no-warp and warped cloth/debug layers (removed="
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
                source_idx,
                &fit_err))
        {
            canva.cloth_fitting_active = false;
            canva.cloth_fitting_status = "Done";
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

}
