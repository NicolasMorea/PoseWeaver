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

    bool canva_layer_is_visibility_map(const CanvaLayer& layer)
    {
        return layer.kind == CanvaLayerKind::ClothFittingOcclusionDebug
            || layer.kind == CanvaLayerKind::ClothFittingSceneOcclusionDebug
            || layer.kind == CanvaLayerKind::ClothFittingAlignedPredictedDepthOcclusionDebug
            || layer.kind == CanvaLayerKind::ClothFittingIsoOcclusionDebug;
    }

    bool canva_stylize_warped_cloth_layer(
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
                *out_error = "warped cloth stylization source is invalid";
            return false;
        }

        if (std::clamp(destination.stylization_method, 0, 3) == 3)
        {
            destination.pixels.assign(source.pixels.begin(), source.pixels.begin() + expected_size);
            destination.dirty = true;
            destination.bounds_dirty = true;
            return true;
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

        const float softness =
            std::clamp(destination.stylization_softness, 0.0f, 1.0f);
        const int contour_radius =
            std::clamp(destination.stylization_contour_radius, 1, 24);
        const int line_gap_radius =
            std::clamp(destination.stylization_line_gap_radius, 0, 6);
        const int cleanup_radius = std::max(
            1,
            (int)std::lround((float)contour_radius * 0.2f));
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

        const int method = std::clamp(destination.stylization_method, 0, 3);
        const float outline_width =
            std::clamp(destination.stylization_outline_width, 0.0f, 12.0f);
        const float wrinkle_width =
            std::clamp(destination.stylization_wrinkle_width, 0.0f, 8.0f);
        const float edge_threshold =
            std::clamp(destination.stylization_edge_threshold, 0.01f, 0.5f);
        const float shadow_threshold =
            std::clamp(destination.stylization_shadow_threshold, 0.5f, 1.0f);
        const float shadow_strength =
            std::clamp(destination.stylization_shadow_strength, 0.0f, 0.8f);
        std::array<float, 3> base_rgb{};
        std::array<unsigned char, 3> line_rgb{};
        for (int channel = 0; channel < 3; ++channel)
        {
            base_rgb[(size_t)channel] =
                std::clamp(destination.stylization_cloth_color[(size_t)channel], 0.0f, 1.0f)
                * 255.0f;
            line_rgb[(size_t)channel] = (unsigned char)std::lround(
                std::clamp(destination.stylization_line_color[(size_t)channel], 0.0f, 1.0f)
                * 255.0f);
        }
        const float base_luma = std::max(
            1.0f,
            base_rgb[0] * 0.2126f + base_rgb[1] * 0.7152f + base_rgb[2] * 0.0722f);

        std::vector<float> raw_luma(crop_pixels, 0.0f);
        std::vector<float> source_luma_samples;
        source_luma_samples.reserve(crop_pixels);
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
            std::array<float, 3> sample_rgb{};
            for (int channel = 0; channel < 3; ++channel)
            {
                sample_rgb[(size_t)channel] = source.pixels[source_index + 3u] == 0u
                    ? blurred[blur_index + (size_t)channel] / blur_alpha
                    : (float)source.pixels[source_index + (size_t)channel];
            }
            const float luma =
                sample_rgb[0] * 0.2126f + sample_rgb[1] * 0.7152f + sample_rgb[2] * 0.0722f;
            raw_luma[pixel_index] = luma;
            if (source.pixels[source_index + 3u] != 0u)
            {
                source_luma_samples.push_back(luma);
            }
        }
        std::sort(source_luma_samples.begin(), source_luma_samples.end());
        const float source_base_luma = source_luma_samples.empty()
            ? base_luma
            : std::max(
                1.0f,
                source_luma_samples[(size_t)std::floor(
                    (float)(source_luma_samples.size() - 1u) * 0.90f)]);
        std::vector<float> intensity(crop_pixels, 1.0f);
        for (size_t pixel_index = 0; pixel_index < crop_pixels; ++pixel_index)
        {
            if (mask[pixel_index] == 0u)
                continue;
            intensity[pixel_index] =
                std::clamp(raw_luma[pixel_index] / source_base_luma, 0.0f, 1.25f);
        }

        const auto box_blur_scalar = [crop_width, crop_height](
            const std::vector<float>& input,
            int radius)
        {
            if (radius <= 0)
                return input;
            std::vector<float> horizontal(input.size(), 0.0f);
            std::vector<float> output(input.size(), 0.0f);
            for (int y = 0; y < crop_height; ++y)
            {
                float sum = 0.0f;
                for (int x = 0; x <= radius && x < crop_width; ++x)
                    sum += input[(size_t)y * (size_t)crop_width + (size_t)x];
                for (int x = 0; x < crop_width; ++x)
                {
                    const int begin = std::max(0, x - radius);
                    const int end = std::min(crop_width - 1, x + radius);
                    horizontal[(size_t)y * (size_t)crop_width + (size_t)x] =
                        sum / (float)(end - begin + 1);
                    if (x - radius >= 0)
                        sum -= input[(size_t)y * (size_t)crop_width + (size_t)(x - radius)];
                    if (x + radius + 1 < crop_width)
                        sum += input[(size_t)y * (size_t)crop_width + (size_t)(x + radius + 1)];
                }
            }
            for (int x = 0; x < crop_width; ++x)
            {
                float sum = 0.0f;
                for (int y = 0; y <= radius && y < crop_height; ++y)
                    sum += horizontal[(size_t)y * (size_t)crop_width + (size_t)x];
                for (int y = 0; y < crop_height; ++y)
                {
                    const int begin = std::max(0, y - radius);
                    const int end = std::min(crop_height - 1, y + radius);
                    output[(size_t)y * (size_t)crop_width + (size_t)x] =
                        sum / (float)(end - begin + 1);
                    if (y - radius >= 0)
                        sum -= horizontal[(size_t)(y - radius) * (size_t)crop_width + (size_t)x];
                    if (y + radius + 1 < crop_height)
                        sum += horizontal[(size_t)(y + radius + 1) * (size_t)crop_width + (size_t)x];
                }
            }
            return output;
        };
        const int tone_blur_radius = method == 1
            ? 1 + (int)std::lround(softness * 3.0f)
            : 1;
        const std::vector<float> filtered_intensity =
            box_blur_scalar(intensity, tone_blur_radius);

        const int contour_exclusion =
            std::max(2, (int)std::ceil(outline_width) + contour_radius / 2);
        const int wrinkle_blur_radius = std::clamp(contour_radius / 3, 2, 5);
        const std::vector<float> wrinkle_intensity =
            box_blur_scalar(
                box_blur_scalar(intensity, wrinkle_blur_radius),
                wrinkle_blur_radius);
        const std::vector<float> wrinkle_background =
            box_blur_scalar(
                box_blur_scalar(wrinkle_intensity, wrinkle_blur_radius + 3),
                wrinkle_blur_radius + 3);
        std::vector<float> ridge_strength(crop_pixels, 0.0f);
        for (size_t index = 0; index < crop_pixels; ++index)
        {
            if (mask[index] != 0u && distance[index] > contour_exclusion)
                ridge_strength[index] = std::max(
                    0.0f,
                    wrinkle_background[index] - wrinkle_intensity[index]);
        }

        std::vector<unsigned char> weak_lines(crop_pixels, 0u);
        std::vector<unsigned char> strong_lines(crop_pixels, 0u);
        for (int y = 2; y + 2 < crop_height; ++y)
        {
            for (int x = 2; x + 2 < crop_width; ++x)
            {
                const size_t index = (size_t)y * (size_t)crop_width + (size_t)x;
                if (mask[index] == 0u || distance[index] <= contour_exclusion)
                    continue;

                const size_t north = index - (size_t)crop_width;
                const size_t south = index + (size_t)crop_width;
                const float center = wrinkle_intensity[index];
                const float dxx = wrinkle_intensity[index + 1u] - 2.0f * center
                    + wrinkle_intensity[index - 1u];
                const float dyy = wrinkle_intensity[south] - 2.0f * center
                    + wrinkle_intensity[north];
                const float dxy = 0.25f * (
                    wrinkle_intensity[south + 1u] - wrinkle_intensity[south - 1u]
                    - wrinkle_intensity[north + 1u] + wrinkle_intensity[north - 1u]);
                const float half_trace = 0.5f * (dxx + dyy);
                const float discriminant = std::hypot(0.5f * (dxx - dyy), dxy);
                const float normal_curvature = half_trace + discriminant;
                const float tangent_curvature = half_trace - discriminant;
                if (normal_curvature <= 0.0f
                    || normal_curvature < std::abs(tangent_curvature) * 1.25f)
                {
                    continue;
                }

                float normal_x = dxy;
                float normal_y = normal_curvature - dxx;
                if (std::abs(normal_x) + std::abs(normal_y) < 1e-6f)
                {
                    normal_x = 1.0f;
                    normal_y = 0.0f;
                }
                int step_x = 0;
                int step_y = 0;
                if (std::abs(normal_x) > std::abs(normal_y) * 2.0f)
                    step_x = 1;
                else if (std::abs(normal_y) > std::abs(normal_x) * 2.0f)
                    step_y = 1;
                else
                {
                    step_x = 1;
                    step_y = normal_x * normal_y >= 0.0f ? 1 : -1;
                }
                const size_t before = (size_t)(y - step_y) * (size_t)crop_width
                    + (size_t)(x - step_x);
                const size_t after = (size_t)(y + step_y) * (size_t)crop_width
                    + (size_t)(x + step_x);
                const bool ridge_peak = ridge_strength[index] >= ridge_strength[before]
                    && ridge_strength[index] >= ridge_strength[after];

                weak_lines[index] = ridge_peak
                    && ridge_strength[index] >= edge_threshold * 0.26f ? 1u : 0u;
                strong_lines[index] = ridge_peak
                    && ridge_strength[index] >= edge_threshold * 0.48f ? 1u : 0u;
            }
        }

        std::vector<unsigned char> line_mask(crop_pixels, 0u);
        std::deque<size_t> line_queue;
        for (size_t index = 0; index < crop_pixels; ++index)
        {
            if (strong_lines[index] == 0u)
                continue;
            line_mask[index] = 1u;
            line_queue.push_back(index);
        }
        while (!line_queue.empty())
        {
            const size_t index = line_queue.front();
            line_queue.pop_front();
            const int x = (int)(index % (size_t)crop_width);
            const int y = (int)(index / (size_t)crop_width);
            for (int oy = -1; oy <= 1; ++oy)
            {
                for (int ox = -1; ox <= 1; ++ox)
                {
                    if ((ox == 0 && oy == 0) || x + ox < 0 || x + ox >= crop_width
                        || y + oy < 0 || y + oy >= crop_height)
                    {
                        continue;
                    }
                    const size_t neighbor = (size_t)(y + oy) * (size_t)crop_width
                        + (size_t)(x + ox);
                    if (weak_lines[neighbor] != 0u && line_mask[neighbor] == 0u)
                    {
                        line_mask[neighbor] = 1u;
                        line_queue.push_back(neighbor);
                    }
                }
            }
        }

        std::vector<unsigned char> separated_lines = line_mask;
        for (int y = 1; y + 1 < crop_height; ++y)
        {
            for (int x = 1; x + 1 < crop_width; ++x)
            {
                const size_t index = (size_t)y * (size_t)crop_width + (size_t)x;
                if (line_mask[index] == 0u)
                    continue;
                int neighbor_count = 0;
                for (int oy = -1; oy <= 1; ++oy)
                {
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        if (ox != 0 || oy != 0)
                        {
                            const size_t neighbor =
                                (size_t)(y + oy) * (size_t)crop_width + (size_t)(x + ox);
                            neighbor_count += line_mask[neighbor] != 0u ? 1 : 0;
                        }
                    }
                }
                if (neighbor_count >= 4)
                    separated_lines[index] = 0u;
            }
        }
        line_mask = std::move(separated_lines);

        std::vector<unsigned char> visited(crop_pixels, 0u);
        const int minimum_line_pixels = std::max(12, contour_radius * 2);
        for (size_t seed = 0; seed < crop_pixels; ++seed)
        {
            if (line_mask[seed] == 0u || visited[seed] != 0u)
                continue;
            std::vector<size_t> component;
            line_queue.push_back(seed);
            visited[seed] = 1u;
            while (!line_queue.empty())
            {
                const size_t index = line_queue.front();
                line_queue.pop_front();
                component.push_back(index);
                const int x = (int)(index % (size_t)crop_width);
                const int y = (int)(index / (size_t)crop_width);
                for (int oy = -1; oy <= 1; ++oy)
                {
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        if ((ox == 0 && oy == 0) || x + ox < 0 || x + ox >= crop_width
                            || y + oy < 0 || y + oy >= crop_height)
                        {
                            continue;
                        }
                        const size_t neighbor = (size_t)(y + oy) * (size_t)crop_width
                            + (size_t)(x + ox);
                        if (line_mask[neighbor] != 0u && visited[neighbor] == 0u)
                        {
                            visited[neighbor] = 1u;
                            line_queue.push_back(neighbor);
                        }
                    }
                }
            }
            int component_min_x = crop_width;
            int component_min_y = crop_height;
            int component_max_x = -1;
            int component_max_y = -1;
            int endpoint_count = 0;
            for (size_t index : component)
            {
                const int x = (int)(index % (size_t)crop_width);
                const int y = (int)(index / (size_t)crop_width);
                component_min_x = std::min(component_min_x, x);
                component_min_y = std::min(component_min_y, y);
                component_max_x = std::max(component_max_x, x);
                component_max_y = std::max(component_max_y, y);
                int neighbor_count = 0;
                for (int oy = -1; oy <= 1; ++oy)
                {
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        if ((ox == 0 && oy == 0) || x + ox < 0 || x + ox >= crop_width
                            || y + oy < 0 || y + oy >= crop_height)
                        {
                            continue;
                        }
                        const size_t neighbor = (size_t)(y + oy) * (size_t)crop_width
                            + (size_t)(x + ox);
                        neighbor_count += line_mask[neighbor] != 0u ? 1 : 0;
                    }
                }
                endpoint_count += neighbor_count <= 1 ? 1 : 0;
            }
            const int component_span = std::max(
                component_max_x - component_min_x + 1,
                component_max_y - component_min_y + 1);
            if ((int)component.size() < minimum_line_pixels
                || component_span < minimum_line_pixels
                || endpoint_count == 0
                || endpoint_count > 4)
            {
                for (size_t index : component)
                    line_mask[index] = 0u;
            }
        }
        if (line_gap_radius > 0)
            line_mask = morph(morph(line_mask, line_gap_radius, true), line_gap_radius, false);
        const int wrinkle_radius =
            std::max(0, (int)std::floor(wrinkle_width * 0.5f));
        if (wrinkle_radius > 0)
        {
            std::vector<int> endpoint_distance(crop_pixels, far_distance);
            for (int y = 1; y + 1 < crop_height; ++y)
            {
                for (int x = 1; x + 1 < crop_width; ++x)
                {
                    const size_t index = (size_t)y * (size_t)crop_width + (size_t)x;
                    if (line_mask[index] == 0u)
                        continue;
                    int neighbor_count = 0;
                    for (int oy = -1; oy <= 1; ++oy)
                    {
                        for (int ox = -1; ox <= 1; ++ox)
                        {
                            if (ox != 0 || oy != 0)
                            {
                                const size_t neighbor =
                                    (size_t)(y + oy) * (size_t)crop_width + (size_t)(x + ox);
                                neighbor_count += line_mask[neighbor] != 0u ? 1 : 0;
                            }
                        }
                    }
                    if (neighbor_count <= 1)
                    {
                        endpoint_distance[index] = 0;
                        line_queue.push_back(index);
                    }
                }
            }
            while (!line_queue.empty())
            {
                const size_t index = line_queue.front();
                line_queue.pop_front();
                const int x = (int)(index % (size_t)crop_width);
                const int y = (int)(index / (size_t)crop_width);
                for (int oy = -1; oy <= 1; ++oy)
                {
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        if ((ox == 0 && oy == 0) || x + ox < 0 || x + ox >= crop_width
                            || y + oy < 0 || y + oy >= crop_height)
                        {
                            continue;
                        }
                        const size_t neighbor = (size_t)(y + oy) * (size_t)crop_width
                            + (size_t)(x + ox);
                        if (line_mask[neighbor] != 0u
                            && endpoint_distance[neighbor] > endpoint_distance[index] + 1)
                        {
                            endpoint_distance[neighbor] = endpoint_distance[index] + 1;
                            line_queue.push_back(neighbor);
                        }
                    }
                }
            }

            std::vector<unsigned char> expanded_lines = line_mask;
            for (int y = 0; y < crop_height; ++y)
            {
                for (int x = 0; x < crop_width; ++x)
                {
                    const size_t index = (size_t)y * (size_t)crop_width + (size_t)x;
                    if (line_mask[index] == 0u)
                        continue;
                    const int radius = std::min(
                        wrinkle_radius,
                        endpoint_distance[index] == far_distance
                            ? wrinkle_radius
                            : endpoint_distance[index] / 2);
                    for (int oy = -radius; oy <= radius; ++oy)
                    {
                        for (int ox = -radius; ox <= radius; ++ox)
                        {
                            if (ox * ox + oy * oy > radius * radius
                                || x + ox < 0 || x + ox >= crop_width
                                || y + oy < 0 || y + oy >= crop_height)
                            {
                                continue;
                            }
                            const size_t neighbor = (size_t)(y + oy) * (size_t)crop_width
                                + (size_t)(x + ox);
                            if (mask[neighbor] != 0u)
                                expanded_lines[neighbor] = 1u;
                        }
                    }
                }
            }
            line_mask = std::move(expanded_lines);
        }
        for (size_t index = 0; index < crop_pixels; ++index)
        {
            if (distance[index] <= contour_exclusion)
                line_mask[index] = 0u;
        }

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
                    output[0] = line_rgb[0];
                    output[1] = line_rgb[1];
                    output[2] = line_rgb[2];
                }
                else if (method == 2)
                {
                    output[0] = 255u;
                    output[1] = 255u;
                    output[2] = 255u;
                }
                else
                {
                    float tone = 1.0f;
                    const float source_tone = filtered_intensity[pixel_index];
                    if (method == 0)
                    {
                        if (source_tone < shadow_threshold - 0.08f)
                            tone = 1.0f - shadow_strength;
                        else if (source_tone < shadow_threshold + 0.06f)
                            tone = 1.0f - shadow_strength * 0.52f;
                    }
                    else
                    {
                        const float feather = 0.015f + softness * 0.12f;
                        float transition = std::clamp(
                            (source_tone - (shadow_threshold - feather)) / (2.0f * feather),
                            0.0f,
                            1.0f);
                        transition = transition * transition * (3.0f - 2.0f * transition);
                        transition = std::round(transition * 3.0f) / 3.0f;
                        tone = 1.0f - shadow_strength * (1.0f - transition);
                    }
                    for (int channel = 0; channel < 3; ++channel)
                    {
                        output[channel] = (unsigned char)std::lround(
                            std::clamp(base_rgb[(size_t)channel] * tone, 0.0f, 255.0f));
                    }
                }
                output[3] = 255u;
            }
        }

        destination.dirty = true;
        destination.bounds_dirty = true;
        return true;
    }

    bool canva_has_cloth_refit_maps(const CanvaState& canva)
    {
        const size_t pixel_count = (size_t)canva.width * (size_t)canva.height;
        const int visibility_index = canva_find_layer_index_by_id(
            canva.layers,
            canva.cloth_refit_visibility_layer_id);
        const int deformation_index = canva_find_layer_index_by_id(
            canva.layers,
            canva.cloth_refit_deformation_layer_id);
        return canva.cloth_refit_map_width == canva.width
            && canva.cloth_refit_map_height == canva.height
            && visibility_index >= 0
            && deformation_index >= 0
            && canva_layer_is_visibility_map(canva.layers[(size_t)visibility_index])
            && canva.layers[(size_t)visibility_index].pixels.size() >= pixel_count * 4u
            && canva.layers[(size_t)deformation_index].kind
                == CanvaLayerKind::ClothFittingDeformation
            && canva.layers[(size_t)deformation_index].deformation_source_x.size() >= pixel_count
            && canva.layers[(size_t)deformation_index].deformation_source_y.size() >= pixel_count;
    }

    bool canva_refit_cloth_with_cached_maps(
        AppState& state,
        CanvaState& canva,
        Renderer& renderer,
        const CanvaLayer& source_layer,
        std::string* out_error)
    {
        if (!canva_has_cloth_refit_maps(canva))
        {
            if (out_error)
                *out_error = "saved fitting maps are unavailable";
            return false;
        }
        if (!canva_layer_is_generated_cloth(source_layer))
        {
            if (out_error)
                *out_error = "rendered source is not a cloth layer";
            return false;
        }

        std::vector<unsigned char> warped_cloth;
        std::vector<unsigned char> warped_occluded_cloth;
        const int visibility_index = canva_find_layer_index_by_id(
            canva.layers,
            canva.cloth_refit_visibility_layer_id);
        const int deformation_index = canva_find_layer_index_by_id(
            canva.layers,
            canva.cloth_refit_deformation_layer_id);
        const CanvaLayer& visibility_layer = canva.layers[(size_t)visibility_index];
        const CanvaLayer& deformation_layer = canva.layers[(size_t)deformation_index];
        if (!renderer.refit_cloth_render_with_maps(
                source_layer.pixels,
                canva.width,
                canva.height,
                visibility_layer.pixels,
                deformation_layer.deformation_source_x,
                deformation_layer.deformation_source_y,
                warped_cloth,
                warped_occluded_cloth,
                out_error))
        {
            return false;
        }

        const int source_layer_id = source_layer.id;
        const ImVec2 source_offset = source_layer.offset;
        const float source_scale = source_layer.scale;
        CanvaLayer warped_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "Refitted cloth");
        const int warped_layer_id = warped_layer.id;
        warped_layer.kind = CanvaLayerKind::ClothFittingWarped;
        warped_layer.pixels = std::move(warped_occluded_cloth);
        warped_layer.offset = source_offset;
        warped_layer.scale = source_scale;
        warped_layer.enabled = false;
        warped_layer.dirty = true;
        warped_layer.bounds_dirty = true;

        CanvaLayer stylized_layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "Stylized cloth (refit)");
        const int stylized_layer_id = stylized_layer.id;
        stylized_layer.kind = CanvaLayerKind::ClothFittingWarpedSmoothed;
        stylized_layer.offset = source_offset;
        stylized_layer.scale = source_scale;
        stylized_layer.smoothing_source_layer_id = warped_layer_id;
        stylized_layer.stylization_method = state.cloth_stylization_method;
        stylized_layer.stylization_outline_width = state.cloth_stylization_outline_width;
        stylized_layer.stylization_wrinkle_width = state.cloth_stylization_wrinkle_width;
        stylized_layer.stylization_edge_threshold = state.cloth_stylization_edge_threshold;
        stylized_layer.stylization_shadow_threshold = state.cloth_stylization_shadow_threshold;
        stylized_layer.stylization_shadow_strength = state.cloth_stylization_shadow_strength;
        stylized_layer.stylization_softness = state.cloth_stylization_softness;
        stylized_layer.stylization_contour_radius = state.cloth_stylization_contour_radius;
        stylized_layer.stylization_line_gap_radius = state.cloth_stylization_line_gap_radius;
        stylized_layer.stylization_cloth_color = state.cloth_ink_color;
        stylized_layer.stylization_line_color = state.cloth_stylization_line_color;
        if (!canva_stylize_warped_cloth_layer(
                warped_layer,
                stylized_layer,
                canva.width,
                canva.height,
                out_error))
        {
            canva_delete_layer_texture(warped_layer);
            canva_delete_layer_texture(stylized_layer);
            return false;
        }

        canva.layers.insert(canva.layers.begin(), std::move(warped_layer));
        canva.layers.insert(canva.layers.begin(), std::move(stylized_layer));
        for (CanvaLayer& layer : canva.layers)
        {
            if (layer.id == source_layer_id)
            {
                layer.enabled = false;
                break;
            }
        }
        canva.active_layer = canva_find_layer_index_by_id(canva, stylized_layer_id);
        canva_cancel_interactions(canva);
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
        if (!require_pixels(fitting.dumb_render, "dumb render")
            || !require_pixels(fitting.occluded_cloth, "occluded cloth")
            || !require_pixels(fitting.warped_cloth, "warped cloth")
            || !require_pixels(fitting.warped_occluded_cloth, "warped occluded cloth")
            || !require_pixels(fitting.raw_iso_matching, "raw ISO matching")
            || !require_pixels(fitting.filtered_iso_matching, "filtered ISO matching")
            || !require_pixels(fitting.predicted_depth_visibility, "predicted-depth visibility")
            || !require_pixels(fitting.true_depth_visibility, "true-depth visibility")
            || !require_pixels(fitting.corrected_true_depth_visibility, "corrected true-depth visibility")
            || !require_pixels(fitting.iso_visibility, "ISO visibility")
            || !require_pixels(fitting.raw_final_occlusion, "raw final occlusion")
            || !require_pixels(fitting.filtered_final_occlusion, "filtered final occlusion"))
            return false;
        for (const auto& step : fitting.deformation_steps)
            if (!require_pixels(step, "deformation step"))
                return false;
        const size_t pixel_count = (size_t)canva.width * (size_t)canva.height;
        if (fitting.deformation_source_x.size() < pixel_count
            || fitting.deformation_source_y.size() < pixel_count)
        {
            if (out_error)
                *out_error = "deformation field does not match the canvas size";
            return false;
        }

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
            "Stylized cloth");
        const int smoothed_layer_id = smoothed_layer.id;
        smoothed_layer.kind = CanvaLayerKind::ClothFittingWarpedSmoothed;
        smoothed_layer.offset = source_offset;
        smoothed_layer.scale = source_scale;
        smoothed_layer.smoothing_source_layer_id = warped_occluded_layer_id;
        smoothed_layer.stylization_method = state.cloth_stylization_method;
        smoothed_layer.stylization_outline_width = state.cloth_stylization_outline_width;
        smoothed_layer.stylization_wrinkle_width = state.cloth_stylization_wrinkle_width;
        smoothed_layer.stylization_edge_threshold = state.cloth_stylization_edge_threshold;
        smoothed_layer.stylization_shadow_threshold = state.cloth_stylization_shadow_threshold;
        smoothed_layer.stylization_shadow_strength = state.cloth_stylization_shadow_strength;
        smoothed_layer.stylization_softness = state.cloth_stylization_softness;
        smoothed_layer.stylization_contour_radius = state.cloth_stylization_contour_radius;
        smoothed_layer.stylization_line_gap_radius = state.cloth_stylization_line_gap_radius;
        smoothed_layer.stylization_cloth_color = state.cloth_ink_color;
        smoothed_layer.stylization_line_color = state.cloth_stylization_line_color;
        if (!canva_stylize_warped_cloth_layer(
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
                const int layer_id = layer.id;
                canva.layers.insert(insert_pos, std::move(layer));
                return layer_id;
            };

            CanvaLayer dumb_render_layer = canva_make_layer(
                canva.width,
                canva.height,
                canva.next_layer_id++,
                "dumb render");
            dumb_render_layer.kind = CanvaLayerKind::ClothFittingDebug;
            dumb_render_layer.pixels = std::move(fitting.dumb_render);
            dumb_render_layer.offset = source_offset;
            dumb_render_layer.scale = source_scale;
            dumb_render_layer.enabled = false;
            dumb_render_layer.dirty = true;
            dumb_render_layer.bounds_dirty = true;

            CanvaLayer dumb_smoothed_layer = canva_make_layer(
                canva.width,
                canva.height,
                canva.next_layer_id++,
                "dumb layer smoothed");
            dumb_smoothed_layer.kind = CanvaLayerKind::ClothFittingDebug;
            dumb_smoothed_layer.offset = source_offset;
            dumb_smoothed_layer.scale = source_scale;
            dumb_smoothed_layer.enabled = false;
            dumb_smoothed_layer.stylization_method = state.cloth_stylization_method;
            dumb_smoothed_layer.stylization_outline_width = state.cloth_stylization_outline_width;
            dumb_smoothed_layer.stylization_wrinkle_width = state.cloth_stylization_wrinkle_width;
            dumb_smoothed_layer.stylization_edge_threshold = state.cloth_stylization_edge_threshold;
            dumb_smoothed_layer.stylization_shadow_threshold = state.cloth_stylization_shadow_threshold;
            dumb_smoothed_layer.stylization_shadow_strength = state.cloth_stylization_shadow_strength;
            dumb_smoothed_layer.stylization_softness = state.cloth_stylization_softness;
            dumb_smoothed_layer.stylization_contour_radius = state.cloth_stylization_contour_radius;
            dumb_smoothed_layer.stylization_line_gap_radius = state.cloth_stylization_line_gap_radius;
            dumb_smoothed_layer.stylization_cloth_color = state.cloth_ink_color;
            dumb_smoothed_layer.stylization_line_color = state.cloth_stylization_line_color;
            if (!canva_stylize_warped_cloth_layer(
                    dumb_render_layer,
                    dumb_smoothed_layer,
                    canva.width,
                    canva.height,
                    out_error))
            {
                canva_delete_layer_texture(dumb_render_layer);
                canva_delete_layer_texture(dumb_smoothed_layer);
                return false;
            }

            const auto dumb_insert_pos =
                canva.layers.begin() + std::min<int>(1, (int)canva.layers.size());
            canva.layers.insert(dumb_insert_pos, std::move(dumb_render_layer));
            const auto dumb_smoothed_insert_pos =
                canva.layers.begin() + std::min<int>(1, (int)canva.layers.size());
            canva.layers.insert(dumb_smoothed_insert_pos, std::move(dumb_smoothed_layer));

            add_hidden_layer(raw_cloth, "Raw cloth", CanvaLayerKind::ClothFitting);
            add_hidden_layer(std::move(fitting.raw_iso_matching), "Raw ISO matching", CanvaLayerKind::ClothFittingDebug);
            add_hidden_layer(std::move(fitting.filtered_iso_matching), "Filtered ISO matching", CanvaLayerKind::ClothFittingDebug);
            add_hidden_layer(std::move(fitting.predicted_depth_visibility), "Pred depth visibility", CanvaLayerKind::ClothFittingAlignedPredictedDepthOcclusionDebug);
            add_hidden_layer(std::move(fitting.true_depth_visibility), "True depth visibility", CanvaLayerKind::ClothFittingSceneOcclusionDebug);
            add_hidden_layer(std::move(fitting.corrected_true_depth_visibility), "True depth corrected visibility", CanvaLayerKind::ClothFittingSceneOcclusionDebug);
            add_hidden_layer(std::move(fitting.iso_visibility), "ISO matching visibility", CanvaLayerKind::ClothFittingIsoOcclusionDebug);
            add_hidden_layer(std::move(fitting.raw_final_occlusion), "Raw final occlusion", CanvaLayerKind::ClothFittingOcclusionDebug);
            canva.cloth_refit_visibility_layer_id = add_hidden_layer(
                std::move(fitting.filtered_final_occlusion),
                "Filtered final occlusion",
                CanvaLayerKind::ClothFittingOcclusionDebug);

            CanvaLayer deformation_field = canva_make_layer(
                canva.width,
                canva.height,
                canva.next_layer_id++,
                "deformation field");
            deformation_field.kind = CanvaLayerKind::ClothFittingDeformation;
            deformation_field.deformation_source_x = std::move(fitting.deformation_source_x);
            deformation_field.deformation_source_y = std::move(fitting.deformation_source_y);
            deformation_field.offset = source_offset;
            deformation_field.scale = source_scale;
            deformation_field.enabled = false;
            canva_update_deformation_field_pixels(deformation_field, canva.width, canva.height);
            canva.cloth_refit_deformation_layer_id = deformation_field.id;
            const auto deformation_insert_pos =
                canva.layers.begin() + std::min<int>(1, (int)canva.layers.size());
            canva.layers.insert(deformation_insert_pos, std::move(deformation_field));
            canva.cloth_refit_map_width = canva.width;
            canva.cloth_refit_map_height = canva.height;

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
            canva.cloth_fitting_renderer_timings = std::move(fitting.timings_seconds);
            if (canva.cloth_fitting_timing_started)
                canva.cloth_fitting_step_seconds[(size_t)ClothFittingTimingStep::Visibility] =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - renderer_start).count();
            if (out_error)
                *out_error = fit_err.empty() ? "renderer fitting failed" : fit_err;
            return false;
        }
        canva.cloth_fitting_renderer_timings = fitting.timings_seconds;
        if (canva.cloth_fitting_timing_started)
            canva.cloth_fitting_step_seconds[(size_t)ClothFittingTimingStep::Visibility] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - renderer_start).count();
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
            if (canva.cloth_fitting_timing_started)
                canva.cloth_fitting_step_seconds[(size_t)ClothFittingTimingStep::LayerCreation] =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - layer_start).count();
            if (out_error)
                *out_error = add_err.empty() ? "failed to create output layer" : add_err;
            return false;
        }
        if (canva.cloth_fitting_timing_started)
            canva.cloth_fitting_step_seconds[(size_t)ClothFittingTimingStep::LayerCreation] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - layer_start).count();
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
            "Canva: cloth fitting added stylized cloth layer (saved fitting layers, removed="
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
            canva_finish_cloth_fitting_timings(canva);
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
            canva_finish_cloth_fitting_timings(canva);
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
        canva_reset_cloth_fitting_timings(canva);

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
