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
#include <deque>
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

    bool canva_smooth_warped_cloth_layer(
        const CanvaLayer& source,
        CanvaLayer& destination,
        int width,
        int height,
        std::string* out_error)
    {
        const size_t expected_size = (size_t)width * (size_t)height * 4u;
        if (width <= 0 || height <= 0 || source.pixels.size() < expected_size)
        {
            if (out_error)
                *out_error = "warped cloth smoothing source is invalid";
            return false;
        }

        destination.pixels.assign(expected_size, 0u);
        int min_x = width;
        int min_y = height;
        int max_x = -1;
        int max_y = -1;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (source.pixels[((size_t)y * (size_t)width + (size_t)x) * 4u + 3u] == 0u)
                    continue;
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }
        if (max_x < min_x || max_y < min_y)
        {
            destination.dirty = true;
            destination.bounds_dirty = true;
            return true;
        }

        const float filter_strength =
            std::clamp(destination.smoothing_filter_strength, 0.0f, 1.0f);
        const int contour_radius =
            std::clamp(destination.smoothing_contour_radius, 1, 32);
        const int line_gap_radius =
            std::clamp(destination.smoothing_line_gap_radius, 0, 12);
        const int cleanup_radius = std::max(
            1,
            (int)std::lround((float)contour_radius * (0.10f + 0.20f * filter_strength)));
        const int crop_margin = contour_radius + cleanup_radius + line_gap_radius + 2;
        min_x = std::max(0, min_x - crop_margin);
        min_y = std::max(0, min_y - crop_margin);
        max_x = std::min(width - 1, max_x + crop_margin);
        max_y = std::min(height - 1, max_y + crop_margin);
        const int crop_width = max_x - min_x + 1;
        const int crop_height = max_y - min_y + 1;
        const size_t crop_pixels = (size_t)crop_width * (size_t)crop_height;

        std::vector<float> samples(crop_pixels * 4u, 0.0f);
        for (int y = 0; y < crop_height; ++y)
        {
            for (int x = 0; x < crop_width; ++x)
            {
                const size_t source_index =
                    ((size_t)(y + min_y) * (size_t)width + (size_t)(x + min_x)) * 4u;
                const size_t crop_index =
                    ((size_t)y * (size_t)crop_width + (size_t)x) * 4u;
                const float alpha = (float)source.pixels[source_index + 3u] / 255.0f;
                for (int channel = 0; channel < 3; ++channel)
                    samples[crop_index + (size_t)channel] =
                        (float)source.pixels[source_index + (size_t)channel] * alpha;
                samples[crop_index + 3u] = (float)source.pixels[source_index + 3u];
            }
        }

        const int kernel_size = contour_radius * 2 + 1;
        std::vector<float> horizontal(crop_pixels * 4u, 0.0f);
        std::vector<float> blurred(crop_pixels * 4u, 0.0f);
        for (int y = 0; y < crop_height; ++y)
        {
            float sums[4] = {};
            for (int x = 0; x <= contour_radius && x < crop_width; ++x)
            {
                const size_t index = ((size_t)y * (size_t)crop_width + (size_t)x) * 4u;
                for (int channel = 0; channel < 4; ++channel)
                    sums[channel] += samples[index + (size_t)channel];
            }
            for (int x = 0; x < crop_width; ++x)
            {
                const size_t index = ((size_t)y * (size_t)crop_width + (size_t)x) * 4u;
                for (int channel = 0; channel < 4; ++channel)
                    horizontal[index + (size_t)channel] = sums[channel] / (float)kernel_size;
                const int remove_x = x - contour_radius;
                const int add_x = x + contour_radius + 1;
                if (remove_x >= 0)
                {
                    const size_t remove_index =
                        ((size_t)y * (size_t)crop_width + (size_t)remove_x) * 4u;
                    for (int channel = 0; channel < 4; ++channel)
                        sums[channel] -= samples[remove_index + (size_t)channel];
                }
                if (add_x < crop_width)
                {
                    const size_t add_index =
                        ((size_t)y * (size_t)crop_width + (size_t)add_x) * 4u;
                    for (int channel = 0; channel < 4; ++channel)
                        sums[channel] += samples[add_index + (size_t)channel];
                }
            }
        }
        for (int x = 0; x < crop_width; ++x)
        {
            float sums[4] = {};
            for (int y = 0; y <= contour_radius && y < crop_height; ++y)
            {
                const size_t index = ((size_t)y * (size_t)crop_width + (size_t)x) * 4u;
                for (int channel = 0; channel < 4; ++channel)
                    sums[channel] += horizontal[index + (size_t)channel];
            }
            for (int y = 0; y < crop_height; ++y)
            {
                const size_t index = ((size_t)y * (size_t)crop_width + (size_t)x) * 4u;
                for (int channel = 0; channel < 4; ++channel)
                    blurred[index + (size_t)channel] = sums[channel] / (float)kernel_size;
                const int remove_y = y - contour_radius;
                const int add_y = y + contour_radius + 1;
                if (remove_y >= 0)
                {
                    const size_t remove_index =
                        ((size_t)remove_y * (size_t)crop_width + (size_t)x) * 4u;
                    for (int channel = 0; channel < 4; ++channel)
                        sums[channel] -= horizontal[remove_index + (size_t)channel];
                }
                if (add_y < crop_height)
                {
                    const size_t add_index =
                        ((size_t)add_y * (size_t)crop_width + (size_t)x) * 4u;
                    for (int channel = 0; channel < 4; ++channel)
                        sums[channel] += horizontal[add_index + (size_t)channel];
                }
            }
        }

        auto morph_axis = [crop_width, crop_height](
            const std::vector<unsigned char>& input,
            int radius,
            bool horizontal_axis,
            bool dilate)
        {
            if (radius <= 0)
                return input;
            std::vector<unsigned char> output(input.size(), 0u);
            const int major_count = horizontal_axis ? crop_height : crop_width;
            const int minor_count = horizontal_axis ? crop_width : crop_height;
            const int full_window = radius * 2 + 1;
            std::vector<int> prefix((size_t)minor_count + 1u, 0);
            for (int major = 0; major < major_count; ++major)
            {
                prefix[0] = 0;
                for (int minor = 0; minor < minor_count; ++minor)
                {
                    const int x = horizontal_axis ? minor : major;
                    const int y = horizontal_axis ? major : minor;
                    prefix[(size_t)minor + 1u] =
                        prefix[(size_t)minor]
                        + (input[(size_t)y * (size_t)crop_width + (size_t)x] != 0u ? 1 : 0);
                }
                for (int minor = 0; minor < minor_count; ++minor)
                {
                    const int begin = std::max(0, minor - radius);
                    const int finish = std::min(minor_count, minor + radius + 1);
                    const int count = prefix[(size_t)finish] - prefix[(size_t)begin];
                    const bool on = dilate ? count > 0 : count == full_window;
                    const int x = horizontal_axis ? minor : major;
                    const int y = horizontal_axis ? major : minor;
                    output[(size_t)y * (size_t)crop_width + (size_t)x] = on ? 1u : 0u;
                }
            }
            return output;
        };
        auto morph = [&](const std::vector<unsigned char>& input, int radius, bool dilate)
        {
            return morph_axis(
                morph_axis(input, radius, true, dilate),
                radius,
                false,
                dilate);
        };

        std::vector<unsigned char> mask(crop_pixels, 0u);
        for (size_t pixel_index = 0; pixel_index < crop_pixels; ++pixel_index)
        {
            mask[pixel_index] =
                blurred[pixel_index * 4u + 3u] >= 0.5f * 255.0f ? 1u : 0u;
        }
        mask = morph(morph(mask, cleanup_radius, true), cleanup_radius, false);
        mask = morph(morph(mask, cleanup_radius, false), cleanup_radius, true);

        const int far_distance = std::numeric_limits<int>::max();
        std::vector<int> distance(crop_pixels, far_distance);
        std::deque<size_t> queue;
        for (int y = 0; y < crop_height; ++y)
        {
            for (int x = 0; x < crop_width; ++x)
            {
                const size_t index = (size_t)y * (size_t)crop_width + (size_t)x;
                if (mask[index] == 0u)
                    continue;
                const bool boundary =
                    x == 0 || y == 0 || x + 1 == crop_width || y + 1 == crop_height
                    || mask[index - 1u] == 0u
                    || mask[index + 1u] == 0u
                    || mask[index - (size_t)crop_width] == 0u
                    || mask[index + (size_t)crop_width] == 0u;
                if (boundary)
                {
                    distance[index] = 1;
                    queue.push_back(index);
                }
            }
        }
        while (!queue.empty())
        {
            const size_t index = queue.front();
            queue.pop_front();
            const int x = (int)(index % (size_t)crop_width);
            const int y = (int)(index / (size_t)crop_width);
            if (x > 0 && mask[index - 1u] != 0u && distance[index - 1u] > distance[index] + 1)
            {
                distance[index - 1u] = distance[index] + 1;
                queue.push_back(index - 1u);
            }
            if (x + 1 < crop_width && mask[index + 1u] != 0u
                && distance[index + 1u] > distance[index] + 1)
            {
                distance[index + 1u] = distance[index] + 1;
                queue.push_back(index + 1u);
            }
            if (y > 0 && mask[index - (size_t)crop_width] != 0u
                && distance[index - (size_t)crop_width] > distance[index] + 1)
            {
                distance[index - (size_t)crop_width] = distance[index] + 1;
                queue.push_back(index - (size_t)crop_width);
            }
            if (y + 1 < crop_height && mask[index + (size_t)crop_width] != 0u
                && distance[index + (size_t)crop_width] > distance[index] + 1)
            {
                distance[index + (size_t)crop_width] = distance[index] + 1;
                queue.push_back(index + (size_t)crop_width);
            }
        }

        constexpr std::array<float, 3> shader_tones{ 0.78f, 0.91f, 1.0f };
        std::array<std::array<unsigned char, 3>, 3> palette{};
        for (size_t tone = 0; tone < palette.size(); ++tone)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                palette[tone][(size_t)channel] = (unsigned char)std::lround(
                    std::clamp(destination.smoothing_cloth_color[(size_t)channel], 0.0f, 1.0f)
                    * shader_tones[tone]
                    * 255.0f);
            }
        }

        const float outline_width = std::max(0.0f, destination.smoothing_inner_line_width);
        std::vector<float> filtered_rgb(crop_pixels * 3u, 0.0f);
        std::vector<unsigned char> line_mask(crop_pixels, 0u);
        for (size_t pixel_index = 0; pixel_index < crop_pixels; ++pixel_index)
        {
            if (mask[pixel_index] == 0u)
                continue;
            const int x = (int)(pixel_index % (size_t)crop_width);
            const int y = (int)(pixel_index / (size_t)crop_width);
            const size_t source_index =
                ((size_t)(y + min_y) * (size_t)width + (size_t)(x + min_x)) * 4u;
            const size_t blur_index = pixel_index * 4u;
            const float blur_alpha = std::max(blurred[blur_index + 3u] / 255.0f, 1e-5f);
            float shader_rgb[3] = {};
            for (int channel = 0; channel < 3; ++channel)
            {
                const float blur_color = blurred[blur_index + (size_t)channel] / blur_alpha;
                shader_rgb[channel] =
                    source.pixels[source_index + 3u] == 0u
                    ? blur_color
                    : (float)source.pixels[source_index + (size_t)channel];
                filtered_rgb[pixel_index * 3u + (size_t)channel] =
                    shader_rgb[channel]
                    + (blur_color - shader_rgb[channel]) * (0.15f * filter_strength);
            }

            if (source.pixels[source_index + 3u] == 0u)
                continue;
            float black_distance = 0.0f;
            float shader_tone_distance = std::numeric_limits<float>::max();
            for (int channel = 0; channel < 3; ++channel)
                black_distance += shader_rgb[channel] * shader_rgb[channel];
            for (const auto& tone : palette)
            {
                float tone_distance = 0.0f;
                for (int channel = 0; channel < 3; ++channel)
                {
                    const float delta =
                        shader_rgb[channel] - (float)tone[(size_t)channel];
                    tone_distance += delta * delta;
                }
                shader_tone_distance = std::min(shader_tone_distance, tone_distance);
            }
            line_mask[pixel_index] =
                black_distance <= shader_tone_distance ? 1u : 0u;
        }
        if (line_gap_radius > 0)
            line_mask = morph(morph(line_mask, line_gap_radius, true), line_gap_radius, false);

        for (int y = 0; y < crop_height; ++y)
        {
            for (int x = 0; x < crop_width; ++x)
            {
                const size_t pixel_index = (size_t)y * (size_t)crop_width + (size_t)x;
                if (mask[pixel_index] == 0u)
                    continue;
                const size_t output_index =
                    ((size_t)(y + min_y) * (size_t)width + (size_t)(x + min_x)) * 4u;
                unsigned char* output = destination.pixels.data() + output_index;
                const bool outline =
                    outline_width > 0.0f && (float)distance[pixel_index] <= outline_width;
                if (outline || line_mask[pixel_index] != 0u)
                {
                    output[0] = 0u;
                    output[1] = 0u;
                    output[2] = 0u;
                }
                else
                {
                    size_t selected_tone = 0;
                    float selected_distance = std::numeric_limits<float>::max();
                    for (size_t tone = 0; tone < palette.size(); ++tone)
                    {
                        float color_distance = 0.0f;
                        for (int channel = 0; channel < 3; ++channel)
                        {
                            const float delta =
                                filtered_rgb[pixel_index * 3u + (size_t)channel]
                                - (float)palette[tone][(size_t)channel];
                            color_distance += delta * delta;
                        }
                        if (color_distance < selected_distance)
                        {
                            selected_distance = color_distance;
                            selected_tone = tone;
                        }
                    }
                    output[0] = palette[selected_tone][0];
                    output[1] = palette[selected_tone][1];
                    output[2] = palette[selected_tone][2];
                }
                output[3] = 255u;
            }
        }

        destination.dirty = true;
        destination.bounds_dirty = true;
        return true;
    }
    static bool canva_add_cloth_fitting_layers(
        CanvaState& canva,
        const CanvaLayer& source_layer,
        const AppState& state,
        ClothFittingLayers fitting,
        bool include_hidden_output_layers,
        std::string* out_error)
    {
        const size_t expected_size = (size_t)canva.width * (size_t)canva.height * 4u;
        const auto require_pixels = [&](const std::vector<unsigned char>& pixels, const char* label)
        {
            if (pixels.size() >= expected_size)
                return true;
            if (out_error)
                *out_error = std::string(label) + " pixel buffer is too small";
            return false;
        };
        if (!require_pixels(fitting.occluded_cloth, "occluded cloth")
            || !require_pixels(fitting.warped_cloth, "warped cloth")
            || !require_pixels(fitting.warped_occluded_cloth, "warped occluded cloth")
            || !require_pixels(fitting.raw_iso_matching, "raw ISO matching")
            || !require_pixels(fitting.filtered_iso_matching, "filtered ISO matching")
            || !require_pixels(fitting.predicted_depth_visibility, "predicted-depth visibility")
            || !require_pixels(fitting.true_depth_visibility, "true-depth visibility")
            || !require_pixels(fitting.iso_visibility, "ISO visibility")
            || !require_pixels(fitting.raw_final_occlusion, "raw final occlusion")
            || !require_pixels(fitting.filtered_final_occlusion, "filtered final occlusion"))
            return false;
        for (const auto& step : fitting.deformation_steps)
            if (!require_pixels(step, "deformation step"))
                return false;

        const std::vector<unsigned char> raw_cloth = source_layer.pixels;
        const ImVec2 source_offset = source_layer.offset;
        const float source_scale = source_layer.scale;
        const int source_layer_id = source_layer.id;
        const bool remove_hidden_source_layer =
            !source_layer.enabled && canva_layer_is_generated_cloth(source_layer);

        CanvaLayer warped_occluded_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "Warped occluded cloth");
        const int warped_occluded_layer_id = warped_occluded_layer.id;
        warped_occluded_layer.kind = CanvaLayerKind::ClothFittingWarped;
        warped_occluded_layer.pixels = std::move(fitting.warped_occluded_cloth);
        warped_occluded_layer.offset = source_offset;
        warped_occluded_layer.scale = source_scale;
        warped_occluded_layer.enabled = false;
        warped_occluded_layer.dirty = true;
        warped_occluded_layer.bounds_dirty = true;

        CanvaLayer smoothed_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "Smooth result");
        const int smoothed_layer_id = smoothed_layer.id;
        smoothed_layer.kind = CanvaLayerKind::ClothFittingWarpedSmoothed;
        smoothed_layer.offset = source_offset;
        smoothed_layer.scale = source_scale;
        smoothed_layer.smoothing_source_layer_id = warped_occluded_layer_id;
        smoothed_layer.smoothing_filter_strength = state.cloth_smoothing_filter_strength;
        smoothed_layer.smoothing_inner_line_width = state.cloth_smoothing_inner_line_width;
        smoothed_layer.smoothing_contour_radius = state.cloth_smoothing_contour_radius;
        smoothed_layer.smoothing_line_gap_radius = state.cloth_smoothing_line_gap_radius;
        smoothed_layer.smoothing_cloth_color = state.cloth_ink_color;
        if (!canva_smooth_warped_cloth_layer(
                warped_occluded_layer,
                smoothed_layer,
                canva.width,
                canva.height,
                out_error))
        {
            canva_delete_layer_texture(warped_occluded_layer);
            canva_delete_layer_texture(smoothed_layer);
            return false;
        }

        const std::filesystem::path output_dir = state.cloth_fitting_output_dir.empty()
            ? std::filesystem::path("renders")
            : std::filesystem::path(state.cloth_fitting_output_dir);
        const std::filesystem::path smooth_path = output_dir / "cloth_fitting_smooth_result.png";
        if (stbi_write_png(
                smooth_path.string().c_str(),
                canva.width,
                canva.height,
                4,
                smoothed_layer.pixels.data(),
                canva.width * 4) == 0)
        {
            canva_delete_layer_texture(warped_occluded_layer);
            canva_delete_layer_texture(smoothed_layer);
            if (out_error)
                *out_error = "could not write " + smooth_path.string();
            return false;
        }

        canva.layers.insert(canva.layers.begin(), std::move(warped_occluded_layer));
        canva.layers.insert(canva.layers.begin(), std::move(smoothed_layer));

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

            add_hidden_layer(raw_cloth, "Raw cloth", CanvaLayerKind::ClothFitting);
            add_hidden_layer(std::move(fitting.raw_iso_matching), "Raw ISO matching", CanvaLayerKind::ClothFittingDebug);
            add_hidden_layer(std::move(fitting.filtered_iso_matching), "Filtered ISO matching", CanvaLayerKind::ClothFittingDebug);
            add_hidden_layer(std::move(fitting.predicted_depth_visibility), "Pred depth visibility", CanvaLayerKind::ClothFittingAlignedPredictedDepthOcclusionDebug);
            add_hidden_layer(std::move(fitting.true_depth_visibility), "True depth visibility", CanvaLayerKind::ClothFittingSceneOcclusionDebug);
            add_hidden_layer(std::move(fitting.iso_visibility), "ISO matching visibility", CanvaLayerKind::ClothFittingIsoOcclusionDebug);
            add_hidden_layer(std::move(fitting.raw_final_occlusion), "Raw final occlusion", CanvaLayerKind::ClothFittingOcclusionDebug);
            add_hidden_layer(std::move(fitting.filtered_final_occlusion), "Filtered final occlusion", CanvaLayerKind::ClothFittingOcclusionDebug);
            for (int i = 0; i < (int)fitting.deformation_steps.size(); ++i)
            {
                std::ostringstream name;
                name << "Deformation step " << std::setw(2) << std::setfill('0') << i;
                add_hidden_layer(
                    std::move(fitting.deformation_steps[(size_t)i]),
                    name.str().c_str(),
                    CanvaLayerKind::ClothFittingDeformation);
            }
            add_hidden_layer(std::move(fitting.occluded_cloth), "Occluded cloth", CanvaLayerKind::ClothFitting);
            add_hidden_layer(std::move(fitting.warped_cloth), "Warped cloth", CanvaLayerKind::ClothFittingWarped);
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

        canva.active_layer = canva_find_layer_index_by_id(canva, smoothed_layer_id);
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

        ClothFittingLayers fitting;
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
                fitting,
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
        const int tested_count = fitting.tested_count;
        const int removed_count = fitting.removed_count;
        if (!canva_add_cloth_fitting_layers(
                canva,
                source,
                state,
                std::move(fitting),
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
            "Canva: cloth fitting added smooth result layer (saved fitting layers, removed="
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
        if (canva.cloth_fitting_waiting_source_layer)
        {
            canva.cloth_fitting_status = "Waiting for cloth render";
            return;
        }

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
        canva.cloth_fitting_waiting_source_layer = false;
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
