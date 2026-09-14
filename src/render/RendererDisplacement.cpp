#include "render/Renderer.h"
#include "core/Paths.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cnpy.h>
#include <glad/glad.h>
#include <igl/frustum.h>
#include <igl/look_at.h>
#include <stb_image.h>
#include <stb_image_write.h>

namespace poseweaver
{
    namespace
    {
        constexpr float kSketch2RigVerticalFovRad = 0.7348438501358032f;
        constexpr float kSketch2RigHorizontalFovRad = 0.7348438501358032f;
        constexpr float kSketch2RigVerticalFovDeg =
            kSketch2RigVerticalFovRad * 180.0f / 3.14159265358979323846f;
        constexpr int kIsoDimensions = 64;

        static bool nearest_mask_pixels(
            const std::vector<unsigned char>& mask,
            int width,
            int height,
            std::vector<int>& out_x,
            std::vector<int>& out_y);

        static bool rgba_is_color(
            const std::vector<unsigned char>& rgba,
            size_t pixel,
            unsigned char r,
            unsigned char g,
            unsigned char b)
        {
            const size_t i = pixel * 4u;
            return i + 3u < rgba.size()
                && rgba[i + 3u] != 0u
                && rgba[i + 0u] == r
                && rgba[i + 1u] == g
                && rgba[i + 2u] == b;
        }

        static std::uint32_t debug_color_at(const std::vector<unsigned char>& rgba, size_t pixel)
        {
            const size_t i = pixel * 4u;
            if (i + 3u >= rgba.size() || rgba[i + 3u] == 0u)
                return 0xffffffffu;
            return ((std::uint32_t)rgba[i] << 16u)
                | ((std::uint32_t)rgba[i + 1u] << 8u)
                | (std::uint32_t)rgba[i + 2u];
        }

        static void set_debug_color(
            std::vector<unsigned char>& rgba,
            size_t pixel,
            std::uint32_t color)
        {
            const size_t i = pixel * 4u;
            rgba[i] = (unsigned char)((color >> 16u) & 0xffu);
            rgba[i + 1u] = (unsigned char)((color >> 8u) & 0xffu);
            rgba[i + 2u] = (unsigned char)(color & 0xffu);
            rgba[i + 3u] = 255u;
        }

        static void filter_cloth_fitting_debug_colors(
            std::vector<unsigned char>& rgba,
            int width,
            int height,
            int smoothing_radius,
            int smoothing_passes,
            int min_patch_area,
            int error_fill_radius)
        {
            constexpr int first_semantic_color = 5;
            constexpr std::array<std::uint32_t, 12> colors{
                0x808080u, 0x0060ffu, 0x00c8dcu, 0x000000u,
                0x303030u, 0xff2020u, 0xff9000u, 0xbe0020u,
                0x20dc50u, 0xffe000u, 0xff00ffu, 0xa040ffu
            };
            const size_t count = (size_t)width * (size_t)height;
            if (width <= 0 || height <= 0 || rgba.size() < count * 4u)
                return;

            auto color_index = [&](std::uint32_t color)
            {
                for (int i = 0; i < (int)colors.size(); ++i)
                {
                    if (colors[(size_t)i] == color)
                        return i;
                }
                return -1;
            };
            const auto semantic_color_index = [&](std::uint32_t color)
            {
                const int index = color_index(color);
                return index >= first_semantic_color ? index : -1;
            };
            const auto visit_neighbors = [width, height](size_t pixel, bool diagonals, const auto& visit)
            {
                const int x = (int)(pixel % (size_t)width);
                const int y = (int)(pixel / (size_t)width);
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        if ((dx == 0 && dy == 0) || (!diagonals && dx != 0 && dy != 0))
                            continue;
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                            visit((size_t)ny * (size_t)width + (size_t)nx);
                    }
                }
            };

            const auto smooth_colors = [&](int radius, int passes)
            {
                radius = std::clamp(radius, 0, 8);
                passes = std::clamp(passes, 0, 8);
                const int radius_squared = radius * radius;
                for (int pass = 0; pass < passes && radius > 0; ++pass)
                {
                    std::vector<unsigned char> filtered = rgba;
                    for (int y = 0; y < height; ++y)
                    {
                        for (int x = 0; x < width; ++x)
                        {
                            const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                            const int current = semantic_color_index(debug_color_at(rgba, pixel));
                            if (current < 0)
                                continue;
                            std::array<int, colors.size()> votes{};
                            for (int dy = -radius; dy <= radius; ++dy)
                            {
                                const int ny = y + dy;
                                if (ny < 0 || ny >= height)
                                    continue;
                                for (int dx = -radius; dx <= radius; ++dx)
                                {
                                    const int nx = x + dx;
                                    if (nx < 0 || nx >= width || dx * dx + dy * dy > radius_squared)
                                        continue;
                                    const int candidate = semantic_color_index(debug_color_at(
                                        rgba, (size_t)ny * (size_t)width + (size_t)nx));
                                    if (candidate >= 0)
                                        ++votes[(size_t)candidate];
                                }
                            }
                            int best = current;
                            for (int candidate = 0; candidate < (int)colors.size(); ++candidate)
                            {
                                if (votes[(size_t)candidate] > votes[(size_t)best])
                                    best = candidate;
                            }
                            if (best != current)
                                set_debug_color(filtered, pixel, colors[(size_t)best]);
                        }
                    }
                    rgba = std::move(filtered);
                }
            };

            // Replace body-overlapping matching failures with the nearest reliable
            // semantic case before smoothing. Grey means there is no ISO body at
            // that pixel, so it remains an immutable boundary for the filter.
            std::vector<unsigned char> valid_mask(count, 0u);
            for (size_t pixel = 0; pixel < count; ++pixel)
                valid_mask[pixel] = semantic_color_index(debug_color_at(rgba, pixel)) >= 0 ? 1u : 0u;

            std::vector<int> nearest_x;
            std::vector<int> nearest_y;
            if (nearest_mask_pixels(valid_mask, width, height, nearest_x, nearest_y))
            {
                const int max_distance_squared = error_fill_radius > 0
                    ? error_fill_radius * error_fill_radius
                    : std::numeric_limits<int>::max();
                std::vector<unsigned char> filled = rgba;
                for (size_t pixel = 0; pixel < count; ++pixel)
                {
                    const std::uint32_t source_color = debug_color_at(rgba, pixel);
                    if (source_color == 0xffffffffu
                        || source_color == colors[0]
                        || valid_mask[pixel] != 0u)
                        continue;
                    const int x = (int)(pixel % (size_t)width);
                    const int y = (int)(pixel / (size_t)width);
                    const int nx = nearest_x[pixel];
                    const int ny = nearest_y[pixel];
                    if (nx < 0 || ny < 0)
                        continue;
                    const int dx = nx - x;
                    const int dy = ny - y;
                    if (dx * dx + dy * dy > max_distance_squared)
                        continue;
                    const std::uint32_t color = debug_color_at(
                        rgba, (size_t)ny * (size_t)width + (size_t)nx);
                    if (semantic_color_index(color) >= 0)
                        set_debug_color(filled, pixel, color);
                }
                rgba = std::move(filled);
            }

            smoothing_radius = std::clamp(smoothing_radius, 0, 8);
            smoothing_passes = std::clamp(smoothing_passes, 0, 8);
            smooth_colors(smoothing_radius, smoothing_passes);

            min_patch_area = std::clamp(min_patch_area, 1, 4096);
            std::vector<unsigned char> visited(count, 0u);
            std::vector<size_t> component;
            std::vector<size_t> queue;
            const auto remove_small_patches = [&]()
            {
                bool changed = false;
                std::fill(visited.begin(), visited.end(), 0u);
                for (size_t seed = 0; seed < count; ++seed)
                {
                    const std::uint32_t source_color = debug_color_at(rgba, seed);
                    if (visited[seed] != 0u || semantic_color_index(source_color) < 0)
                        continue;
                    visited[seed] = 1u;
                    component.clear();
                    queue.assign(1u, seed);
                    for (size_t head = 0; head < queue.size(); ++head)
                    {
                        const size_t pixel = queue[head];
                        component.push_back(pixel);
                        visit_neighbors(pixel, false, [&](size_t neighbor)
                        {
                            if (visited[neighbor] == 0u && debug_color_at(rgba, neighbor) == source_color)
                            {
                                visited[neighbor] = 1u;
                                queue.push_back(neighbor);
                            }
                        });
                    }
                    if ((int)component.size() >= min_patch_area)
                        continue;
                    std::array<int, colors.size()> boundary_votes{};
                    for (const size_t pixel : component)
                    {
                        visit_neighbors(pixel, true, [&](size_t neighbor)
                        {
                            const int candidate = semantic_color_index(debug_color_at(rgba, neighbor));
                            if (candidate >= 0 && colors[(size_t)candidate] != source_color)
                                ++boundary_votes[(size_t)candidate];
                        });
                    }
                    int best = -1;
                    for (int candidate = 0; candidate < (int)colors.size(); ++candidate)
                    {
                        if (boundary_votes[(size_t)candidate] > 0
                            && (best < 0 || boundary_votes[(size_t)candidate] > boundary_votes[(size_t)best]))
                            best = candidate;
                    }
                    if (best >= 0)
                    {
                        for (const size_t pixel : component)
                            set_debug_color(rgba, pixel, colors[(size_t)best]);
                        changed = true;
                    }
                }
                return changed;
            };
            for (int pass = 0; pass < 3 && remove_small_patches(); ++pass)
            {}

            if (smoothing_radius > 0 && smoothing_passes > 0)
                smooth_colors(std::max(1, smoothing_radius - 1), 1);
            for (int pass = 0; pass < 3 && remove_small_patches(); ++pass)
            {}
        }

        // Exact squared-Euclidean nearest-site transform. The vertical pass is
        // separable; the horizontal pass uses the lower envelope of parabolas.
        static bool nearest_mask_pixels(
            const std::vector<unsigned char>& mask,
            int width,
            int height,
            std::vector<int>& out_x,
            std::vector<int>& out_y)
        {
            const size_t count = (size_t)width * (size_t)height;
            out_x.assign(count, -1);
            out_y.assign(count, -1);
            if (width <= 0 || height <= 0 || mask.size() < count)
                return false;

            bool have_site = false;
            for (size_t i = 0; i < count; ++i)
                have_site = have_site || mask[i] != 0u;
            if (!have_site)
                return false;

            constexpr float inf = 1.0e20f;
            std::vector<float> vertical_distance(count, inf);
            std::vector<int> vertical_site_y(count, -1);
            for (int x = 0; x < width; ++x)
            {
                int nearest = -1;
                for (int y = 0; y < height; ++y)
                {
                    const size_t i = (size_t)y * (size_t)width + (size_t)x;
                    if (mask[i] != 0u)
                        nearest = y;
                    if (nearest >= 0)
                    {
                        const float dy = (float)(y - nearest);
                        vertical_distance[i] = dy * dy;
                        vertical_site_y[i] = nearest;
                    }
                }
                nearest = -1;
                for (int y = height - 1; y >= 0; --y)
                {
                    const size_t i = (size_t)y * (size_t)width + (size_t)x;
                    if (mask[i] != 0u)
                        nearest = y;
                    if (nearest >= 0)
                    {
                        const float dy = (float)(y - nearest);
                        const float distance = dy * dy;
                        if (distance < vertical_distance[i])
                        {
                            vertical_distance[i] = distance;
                            vertical_site_y[i] = nearest;
                        }
                    }
                }
            }

            std::vector<int> sites((size_t)width, -1);
            std::vector<float> boundaries((size_t)width + 1u, inf);
            for (int y = 0; y < height; ++y)
            {
                int envelope_size = 0;
                for (int q = 0; q < width; ++q)
                {
                    const size_t qi = (size_t)y * (size_t)width + (size_t)q;
                    if (!(vertical_distance[qi] < inf * 0.5f))
                        continue;

                    float boundary = -inf;
                    while (envelope_size > 0)
                    {
                        const int p = sites[(size_t)envelope_size - 1u];
                        const float fq = vertical_distance[qi] + (float)(q * q);
                        const float fp = vertical_distance[(size_t)y * (size_t)width + (size_t)p]
                            + (float)(p * p);
                        boundary = (fq - fp) / (2.0f * (float)(q - p));
                        if (boundary > boundaries[(size_t)envelope_size - 1u])
                            break;
                        --envelope_size;
                    }
                    if (envelope_size == 0)
                        boundary = -inf;
                    sites[(size_t)envelope_size] = q;
                    boundaries[(size_t)envelope_size] = boundary;
                    ++envelope_size;
                    boundaries[(size_t)envelope_size] = inf;
                }

                int k = 0;
                for (int x = 0; x < width; ++x)
                {
                    while (k + 1 < envelope_size && boundaries[(size_t)k + 1u] < (float)x)
                        ++k;
                    const int site_x = sites[(size_t)k];
                    const size_t i = (size_t)y * (size_t)width + (size_t)x;
                    const size_t site_i = (size_t)y * (size_t)width + (size_t)site_x;
                    out_x[i] = site_x;
                    out_y[i] = vertical_site_y[site_i];
                }
            }
            return true;
        }

        static void box_blur_scalar(
            const std::vector<float>& source,
            int width,
            int height,
            int radius,
            std::vector<float>& out)
        {
            const size_t count = (size_t)width * (size_t)height;
            if (source.size() < count || width <= 0 || height <= 0 || radius <= 0)
            {
                out = source;
                return;
            }

            std::vector<float> horizontal(count, 0.0f);
            std::vector<float> prefix((size_t)std::max(width, height) + 1u, 0.0f);
            for (int y = 0; y < height; ++y)
            {
                prefix[0] = 0.0f;
                for (int x = 0; x < width; ++x)
                    prefix[(size_t)x + 1u] = prefix[(size_t)x]
                        + source[(size_t)y * (size_t)width + (size_t)x];
                for (int x = 0; x < width; ++x)
                {
                    const int first = std::max(0, x - radius);
                    const int last = std::min(width - 1, x + radius);
                    const int samples = last - first + 1;
                    const float sum = prefix[(size_t)last + 1u] - prefix[(size_t)first];
                    horizontal[(size_t)y * (size_t)width + (size_t)x] =
                        sum / (float)samples;
                }
            }

            out.assign(count, 0.0f);
            for (int x = 0; x < width; ++x)
            {
                prefix[0] = 0.0f;
                for (int y = 0; y < height; ++y)
                    prefix[(size_t)y + 1u] = prefix[(size_t)y]
                        + horizontal[(size_t)y * (size_t)width + (size_t)x];
                for (int y = 0; y < height; ++y)
                {
                    const int first = std::max(0, y - radius);
                    const int last = std::min(height - 1, y + radius);
                    const int samples = last - first + 1;
                    const float sum = prefix[(size_t)last + 1u] - prefix[(size_t)first];
                    out[(size_t)y * (size_t)width + (size_t)x] =
                        sum / (float)samples;
                }
            }
        }

        static void gaussian_like_blur_scalar(
            std::vector<float>& values,
            int width,
            int height,
            float radius)
        {
            // Three box passes give a smooth Gaussian-like brush profile in
            // linear time, even for a large liquify radius.
            const int box_radius = std::max(1, (int)std::lround(radius / 3.0f));
            std::vector<float> blurred;
            for (int pass = 0; pass < 3; ++pass)
            {
                box_blur_scalar(values, width, height, box_radius, blurred);
                values.swap(blurred);
            }
        }

        struct DeformationWarpSeedCounts
        {
            int red = 0;
            int purple = 0;

            int total() const { return red + purple; }
        };

        static DeformationWarpSeedCounts build_deformation_warp(
            const std::vector<unsigned char>& deformation_rgba,
            const std::vector<unsigned char>& body_mask,
            const std::vector<unsigned char>& cloth_mask,
            int width,
            int height,
            float smooth_radius,
            float force,
            std::vector<float>& out_dx,
            std::vector<float>& out_dy)
        {
            const size_t count = (size_t)width * (size_t)height;
            out_dx.assign(count, 0.0f);
            out_dy.assign(count, 0.0f);
            DeformationWarpSeedCounts seed_counts;
            if (deformation_rgba.size() < count * 4u
                || body_mask.size() < count
                || cloth_mask.size() < count
                || force <= 0.0f)
                return seed_counts;

            std::vector<unsigned char> red_mask(count, 0u);
            std::vector<unsigned char> purple_mask(count, 0u);
            for (size_t i = 0; i < count; ++i)
            {
                const size_t p = i * 4u;
                // The deformation-map red is deliberately detected narrowly so
                // blue and purple cases cannot become red warp sources.
                if (deformation_rgba[p + 3u] != 0u
                    && deformation_rgba[p + 0u] >= 240u
                    && deformation_rgba[p + 1u] <= 96u
                    && deformation_rgba[p + 2u] <= 80u)
                {
                    red_mask[i] = 1u;
                    ++seed_counts.red;
                }
                else if (deformation_rgba[p + 3u] != 0u
                    && deformation_rgba[p + 0u] >= 180u
                    && deformation_rgba[p + 1u] <= 96u
                    && deformation_rgba[p + 2u] >= 200u)
                {
                    purple_mask[i] = 1u;
                    ++seed_counts.purple;
                }
            }
            if (seed_counts.total() == 0)
                return seed_counts;

            std::vector<int> body_x;
            std::vector<int> body_y;
            const bool have_body_targets =
                seed_counts.red > 0 && nearest_mask_pixels(body_mask, width, height, body_x, body_y);
            if (seed_counts.red > 0 && !have_body_targets)
                seed_counts.red = 0;

            std::vector<int> cloth_x;
            std::vector<int> cloth_y;
            const bool have_cloth_targets =
                seed_counts.purple > 0 && nearest_mask_pixels(cloth_mask, width, height, cloth_x, cloth_y);
            if (seed_counts.purple > 0 && !have_cloth_targets)
                seed_counts.purple = 0;
            if (seed_counts.total() == 0)
                return seed_counts;

            std::vector<unsigned char> seed_mask(count, 0u);
            std::vector<float> seed_weight(count, 0.0f);
            std::vector<float> seed_dx(count, 0.0f);
            std::vector<float> seed_dy(count, 0.0f);
            for (size_t i = 0; i < count; ++i)
            {
                const int x = (int)(i % (size_t)width);
                const int y = (int)(i / (size_t)width);
                if (red_mask[i] != 0u && have_body_targets)
                {
                    seed_mask[i] = 1u;
                    seed_weight[i] = 1.0f;
                    seed_dx[i] = (float)(body_x[i] - x);
                    seed_dy[i] = (float)(body_y[i] - y);
                }
                else if (purple_mask[i] != 0u && have_cloth_targets)
                {
                    seed_mask[i] = 1u;
                    seed_weight[i] = 1.0f;
                    seed_dx[i] = (float)(x - cloth_x[i]);
                    seed_dy[i] = (float)(y - cloth_y[i]);
                }
            }

            std::vector<int> seed_x;
            std::vector<int> seed_y;
            if (!nearest_mask_pixels(seed_mask, width, height, seed_x, seed_y))
                return DeformationWarpSeedCounts{};

            const float radius = std::max(0.0f, smooth_radius);
            if (radius > 0.0f)
            {
                gaussian_like_blur_scalar(seed_weight, width, height, radius);
                gaussian_like_blur_scalar(seed_dx, width, height, radius);
                gaussian_like_blur_scalar(seed_dy, width, height, radius);
            }
            const float radius_squared = radius * radius;
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t i = (size_t)y * (size_t)width + (size_t)x;
                    const int sx = seed_x[i];
                    const int sy = seed_y[i];
                    if (sx < 0 || sy < 0)
                        continue;
                    const float dx = (float)(x - sx);
                    const float dy = (float)(y - sy);
                    const float distance_squared = dx * dx + dy * dy;
                    if ((radius <= 0.0f && distance_squared > 0.0f)
                        || (radius > 0.0f && distance_squared > radius_squared))
                        continue;

                    float influence = 1.0f;
                    if (radius > 0.0f)
                    {
                        const float t = std::clamp(std::sqrt(distance_squared) / radius, 0.0f, 1.0f);
                        const float smoothstep = t * t * (3.0f - 2.0f * t);
                        influence = 1.0f - smoothstep;
                    }
                    const float density = seed_weight[i];
                    if (!(density > 1e-6f))
                        continue;

                    // Normalize the blurred vectors by blurred seed density.
                    // Adjacent deformation pixels therefore act as one continuous
                    // liquify stroke instead of competing nearest-pixel sites.
                    out_dx[i] = (seed_dx[i] / density) * force * influence;
                    out_dy[i] = (seed_dy[i] / density) * force * influence;
                }
            }
            return seed_counts;
        }

        static float sample_bilinear_scalar(
            const std::vector<float>& values,
            int width,
            int height,
            float x,
            float y)
        {
            x = std::clamp(x, 0.0f, (float)(width - 1));
            y = std::clamp(y, 0.0f, (float)(height - 1));
            const int x0 = (int)std::floor(x);
            const int y0 = (int)std::floor(y);
            const int x1 = std::min(x0 + 1, width - 1);
            const int y1 = std::min(y0 + 1, height - 1);
            const float fx = x - (float)x0;
            const float fy = y - (float)y0;
            const float v00 = values[(size_t)y0 * (size_t)width + (size_t)x0];
            const float v10 = values[(size_t)y0 * (size_t)width + (size_t)x1];
            const float v01 = values[(size_t)y1 * (size_t)width + (size_t)x0];
            const float v11 = values[(size_t)y1 * (size_t)width + (size_t)x1];
            return (v00 * (1.0f - fx) + v10 * fx) * (1.0f - fy)
                + (v01 * (1.0f - fx) + v11 * fx) * fy;
        }

        static void build_inverse_warp_map(
            int width,
            int height,
            const std::vector<float>& dx,
            const std::vector<float>& dy,
            std::vector<float>& out_source_x,
            std::vector<float>& out_source_y)
        {
            const size_t count = (size_t)width * (size_t)height;
            out_source_x.resize(count);
            out_source_y.resize(count);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    float source_x = (float)x;
                    float source_y = (float)y;
                    // Solve source + displacement(source) = destination. The
                    // damped fixed-point iterations stay stable for the broad,
                    // smooth field produced by the liquify brush.
                    for (int iteration = 0; iteration < 12; ++iteration)
                    {
                        const float displacement_x = sample_bilinear_scalar(dx, width, height, source_x, source_y);
                        const float displacement_y = sample_bilinear_scalar(dy, width, height, source_x, source_y);
                        const float next_x = (float)x - displacement_x;
                        const float next_y = (float)y - displacement_y;
                        source_x += (next_x - source_x) * 0.65f;
                        source_y += (next_y - source_y) * 0.65f;
                    }
                    const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                    out_source_x[pixel] = source_x;
                    out_source_y[pixel] = source_y;
                }
            }
        }

        static float catmull_rom_weight(float x)
        {
            x = std::abs(x);
            if (x <= 1.0f)
                return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
            if (x < 2.0f)
                return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
            return 0.0f;
        }

        static void warp_rgba_inverse_sharp(
            const std::vector<unsigned char>& source,
            int width,
            int height,
            const std::vector<float>& source_x,
            const std::vector<float>& source_y,
            std::vector<unsigned char>& out)
        {
            const size_t count = (size_t)width * (size_t)height;
            if (source.size() < count * 4u || source_x.size() < count || source_y.size() < count)
            {
                out = source;
                return;
            }

            out.assign(count * 4u, 0u);
            for (size_t target = 0; target < count; ++target)
            {
                const float x = source_x[target];
                const float y = source_y[target];
                if (x < 0.0f || y < 0.0f || x > (float)(width - 1) || y > (float)(height - 1))
                    continue;

                const int ix = (int)std::floor(x);
                const int iy = (int)std::floor(y);
                float alpha = 0.0f;
                std::array<float, 3> premultiplied{ 0.0f, 0.0f, 0.0f };
                float total_weight = 0.0f;
                for (int oy = -1; oy <= 2; ++oy)
                {
                    const int sy = std::clamp(iy + oy, 0, height - 1);
                    const float wy = catmull_rom_weight(y - (float)(iy + oy));
                    for (int ox = -1; ox <= 2; ++ox)
                    {
                        const int sx = std::clamp(ix + ox, 0, width - 1);
                        const float wx = catmull_rom_weight(x - (float)(ix + ox));
                        const float weight = wx * wy;
                        total_weight += weight;
                        const size_t source_pixel = (size_t)sy * (size_t)width + (size_t)sx;
                        const float sample_alpha = (float)source[source_pixel * 4u + 3u] / 255.0f;
                        alpha += weight * sample_alpha;
                        for (size_t c = 0; c < 3u; ++c)
                            premultiplied[c] += weight * sample_alpha * (float)source[source_pixel * 4u + c];
                    }
                }

                if (std::abs(total_weight) > 1e-6f)
                {
                    alpha /= total_weight;
                    for (float& value : premultiplied)
                        value /= total_weight;
                }
                alpha = std::clamp(alpha, 0.0f, 1.0f);
                out[target * 4u + 3u] = (unsigned char)std::clamp((int)std::lround(alpha * 255.0f), 0, 255);
                if (alpha > 1e-6f)
                {
                    for (size_t c = 0; c < 3u; ++c)
                        out[target * 4u + c] = (unsigned char)std::clamp(
                            (int)std::lround(premultiplied[c] / alpha), 0, 255);
                }
            }
        }

        static void filter_small_occlusion_components(
            const std::vector<unsigned char>& active,
            int width,
            int height,
            std::vector<unsigned char>& keep_flags,
            size_t min_component_size)
        {
            const size_t count = (size_t)width * (size_t)height;
            std::vector<unsigned char> visited(count, 0u);
            std::vector<size_t> stack;
            std::vector<size_t> component;
            stack.reserve(64);
            component.reserve(64);

            for (size_t seed = 0; seed < count; ++seed)
            {
                if (!active[seed] || visited[seed])
                    continue;

                const unsigned char initial_keep = keep_flags[seed];
                stack.clear();
                component.clear();
                stack.push_back(seed);
                visited[seed] = 1u;
                while (!stack.empty())
                {
                    const size_t pixel = stack.back();
                    stack.pop_back();
                    component.push_back(pixel);

                    const int x = (int)(pixel % (size_t)width);
                    const int y = (int)(pixel / (size_t)width);
                    for (int oy = -1; oy <= 1; ++oy)
                    {
                        const int ny = y + oy;
                        if (ny < 0 || ny >= height)
                            continue;
                        for (int ox = -1; ox <= 1; ++ox)
                        {
                            if (ox == 0 && oy == 0)
                                continue;
                            const int nx = x + ox;
                            if (nx < 0 || nx >= width)
                                continue;
                            const size_t neighbor = (size_t)ny * (size_t)width + (size_t)nx;
                            if (!active[neighbor]
                                || visited[neighbor]
                                || keep_flags[neighbor] != initial_keep)
                                continue;
                            visited[neighbor] = 1u;
                            stack.push_back(neighbor);
                        }
                    }
                }

                const unsigned char filtered_keep =
                    component.size() < min_component_size
                        ? (initial_keep ? 0u : 1u)
                        : initial_keep;
                for (size_t pixel : component)
                    keep_flags[pixel] = filtered_keep;
            }
        }

        static void build_mask_integral(
            const std::vector<unsigned char>& values,
            int width,
            int height,
            std::vector<int>& integral)
        {
            const int stride = width + 1;
            integral.assign((size_t)stride * (size_t)(height + 1), 0);
            for (int y = 0; y < height; ++y)
            {
                int row_sum = 0;
                for (int x = 0; x < width; ++x)
                {
                    row_sum += values[(size_t)y * (size_t)width + (size_t)x] != 0u ? 1 : 0;
                    integral[(size_t)(y + 1) * (size_t)stride + (size_t)(x + 1)] =
                        integral[(size_t)y * (size_t)stride + (size_t)(x + 1)] + row_sum;
                }
            }
        }

        static int mask_integral_sum(
            const std::vector<int>& integral,
            int width,
            int x0,
            int y0,
            int x1,
            int y1)
        {
            const int stride = width + 1;
            return integral[(size_t)y1 * (size_t)stride + (size_t)x1]
                - integral[(size_t)y0 * (size_t)stride + (size_t)x1]
                - integral[(size_t)y1 * (size_t)stride + (size_t)x0]
                + integral[(size_t)y0 * (size_t)stride + (size_t)x0];
        }

        static void simplify_chaotic_occlusion_regions(
            const std::vector<unsigned char>& active,
            int width,
            int height,
            int radius,
            int passes,
            float chaos_threshold,
            std::vector<unsigned char>& keep_flags,
            std::vector<unsigned char>& chaotic_flags)
        {
            const size_t count = (size_t)width * (size_t)height;
            if (active.size() < count || keep_flags.size() < count || width <= 0 || height <= 0)
                return;

            radius = std::clamp(radius, 1, 32);
            passes = std::clamp(passes, 0, 8);
            chaotic_flags.assign(count, 0u);
            chaos_threshold = std::clamp(chaos_threshold, 0.01f, 0.5f);
            std::vector<unsigned char> boundary(count, 0u);
            std::vector<unsigned char> active_keep(count, 0u);
            std::vector<unsigned char> next(count, 0u);
            std::vector<int> active_integral;
            std::vector<int> keep_integral;
            std::vector<int> boundary_integral;
            const std::array<std::array<int, 2>, 4> neighbors = {{
                {{ -1, 0 }},
                {{ 1, 0 }},
                {{ 0, -1 }},
                {{ 0, 1 }},
            }};

            for (int pass = 0; pass < passes; ++pass)
            {
                std::fill(boundary.begin(), boundary.end(), 0u);
                std::fill(active_keep.begin(), active_keep.end(), 0u);
                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                        if (!active[pixel])
                            continue;
                        active_keep[pixel] = keep_flags[pixel];
                        for (const auto& direction : neighbors)
                        {
                            const int nx = x + direction[0];
                            const int ny = y + direction[1];
                            if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                                continue;
                            const size_t neighbor = (size_t)ny * (size_t)width + (size_t)nx;
                            if (active[neighbor] && keep_flags[neighbor] != keep_flags[pixel])
                            {
                                boundary[pixel] = 1u;
                                break;
                            }
                        }
                    }
                }

                build_mask_integral(active, width, height, active_integral);
                build_mask_integral(active_keep, width, height, keep_integral);
                build_mask_integral(boundary, width, height, boundary_integral);
                next = keep_flags;
                int changed = 0;
                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                        if (!active[pixel])
                            continue;
                        const int x0 = std::max(0, x - radius);
                        const int y0 = std::max(0, y - radius);
                        const int x1 = std::min(width, x + radius + 1);
                        const int y1 = std::min(height, y + radius + 1);
                        const int active_count = mask_integral_sum(active_integral, width, x0, y0, x1, y1);
                        if (active_count < 9)
                            continue;
                        const int boundary_count = mask_integral_sum(boundary_integral, width, x0, y0, x1, y1);
                        const float chaos = (float)boundary_count / (float)active_count;
                        if (chaos < chaos_threshold)
                            continue;
                        chaotic_flags[pixel] = 1u;

                        const int keep_count = mask_integral_sum(keep_integral, width, x0, y0, x1, y1);
                        const int signed_majority = 2 * keep_count - active_count;
                        const float confidence = std::abs((float)signed_majority) / (float)active_count;
                        if (confidence < 0.08f)
                            continue;
                        const unsigned char filtered = signed_majority >= 0 ? 1u : 0u;
                        if (filtered != keep_flags[pixel])
                        {
                            next[pixel] = filtered;
                            ++changed;
                        }
                    }
                }
                keep_flags = std::move(next);
                if (changed == 0)
                    break;
            }
        }

        static void diagonalize_occlusion_edges(
            const std::vector<unsigned char>& active,
            const std::vector<unsigned char>& eligible,
            int width,
            int height,
            std::vector<unsigned char>& keep_flags)
        {
            const size_t count = (size_t)width * (size_t)height;
            if (active.size() < count || eligible.size() < count || keep_flags.size() < count
                || width <= 2 || height <= 2)
                return;

            auto sample = [&](int x, int y, unsigned char& out_keep) -> bool
            {
                if (x < 0 || y < 0 || x >= width || y >= height)
                    return false;
                const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                if (!active[pixel])
                    return false;
                out_keep = keep_flags[pixel];
                return true;
            };

            const std::array<int, 2> directions = { -1, 1 };
            std::vector<unsigned char> next = keep_flags;
            for (int y = 1; y + 1 < height; ++y)
            {
                for (int x = 1; x + 1 < width; ++x)
                {
                    const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                    if (!active[pixel] || !eligible[pixel])
                        continue;

                    const unsigned char current = keep_flags[pixel];
                    bool cut_corner = false;
                    for (int sy : directions)
                    {
                        for (int sx : directions)
                        {
                            unsigned char side_x = 0u;
                            unsigned char side_y = 0u;
                            unsigned char diagonal = 0u;
                            unsigned char continuation_x = 0u;
                            unsigned char continuation_y = 0u;
                            if (!sample(x - sx, y, side_x)
                                || !sample(x, y - sy, side_y)
                                || !sample(x - sx, y - sy, diagonal)
                                || !sample(x + sx, y, continuation_x)
                                || !sample(x, y + sy, continuation_y))
                                continue;

                            if (side_x != current
                                && side_y != current
                                && diagonal != current
                                && continuation_x == current
                                && continuation_y == current)
                            {
                                cut_corner = true;
                                break;
                            }
                        }
                        if (cut_corner)
                            break;
                    }
                    if (cut_corner)
                        next[pixel] = current ? 0u : 1u;
                }
            }
            keep_flags = std::move(next);
        }

        struct UvFace
        {
            int face = -1;
            double u0 = 0.0;
            double v0 = 0.0;
            double u1 = 0.0;
            double v1 = 0.0;
            double u2 = 0.0;
            double v2 = 0.0;
        };

        static std::string read_text_file_displacement(const std::filesystem::path& path)
        {
            std::ifstream f(path, std::ios::binary);
            if (!f)
                return {};
            std::ostringstream ss;
            ss << f.rdbuf();
            return ss.str();
        }

        static bool parse_json_number_at_fit(const std::string& text, size_t* pos, double* out)
        {
            if (!pos || !out)
                return false;
            const char* begin = text.c_str() + *pos;
            char* end = nullptr;
            const double value = std::strtod(begin, &end);
            if (end == begin || !std::isfinite(value))
                return false;
            *out = value;
            *pos = (size_t)(end - text.c_str());
            return true;
        }

        static bool parse_json_number_field_fit(const std::string& text, const char* key, double* out_value)
        {
            if (!key || !out_value)
                return false;
            const std::string needle = std::string("\"") + key + "\"";
            const size_t key_pos = text.find(needle);
            if (key_pos == std::string::npos)
                return false;
            size_t pos = text.find(':', key_pos + needle.size());
            if (pos == std::string::npos)
                return false;
            ++pos;
            while (pos < text.size() && std::isspace((unsigned char)text[pos]))
                ++pos;
            return parse_json_number_at_fit(text, &pos, out_value);
        }

        static bool parse_json_number_array_field_fit(
            const std::string& text,
            const char* key,
            std::vector<double>* out_values)
        {
            if (!key || !out_values)
                return false;
            out_values->clear();
            const std::string needle = std::string("\"") + key + "\"";
            const size_t key_pos = text.find(needle);
            if (key_pos == std::string::npos)
                return false;
            size_t pos = text.find('[', key_pos + needle.size());
            if (pos == std::string::npos)
                return false;
            ++pos;
            while (pos < text.size())
            {
                while (pos < text.size() && (std::isspace((unsigned char)text[pos]) || text[pos] == ','))
                    ++pos;
                if (pos >= text.size())
                    break;
                if (text[pos] == ']')
                    return true;
                double value = 0.0;
                if (!parse_json_number_at_fit(text, &pos, &value))
                    return false;
                out_values->push_back(value);
            }
            return false;
        }

        static bool parse_sketch2rig_joints_fit(
            const std::string& text,
            Eigen::MatrixXd* out_joints,
            int* out_image_size)
        {
            if (!out_joints)
                return false;
            constexpr int kSketchJointCount = 22;
            double image_size_d = 512.0;
            (void)parse_json_number_field_fit(text, "image_size", &image_size_d);
            if (out_image_size)
                *out_image_size = std::clamp((int)std::lround(image_size_d), 2, 8192);

            const size_t joints_key = text.find("\"joints\"");
            if (joints_key == std::string::npos)
                return false;
            size_t pos = text.find('[', joints_key);
            if (pos == std::string::npos)
                return false;
            ++pos;

            Eigen::MatrixXd joints = Eigen::MatrixXd::Zero(kSketchJointCount, 3);
            int parsed = 0;
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
                double index_d = -1.0;
                std::vector<double> values;
                if (parse_json_number_field_fit(object, "index", &index_d)
                    && parse_json_number_array_field_fit(object, "position", &values)
                    && values.size() >= 3)
                {
                    const int index = (int)std::lround(index_d);
                    if (index >= 0 && index < kSketchJointCount)
                    {
                        joints.row(index) = Eigen::RowVector3d(values[0], values[1], values[2]);
                        ++parsed;
                    }
                }
                pos = object_end + 1;
            }

            if (parsed < kSketchJointCount)
                return false;
            *out_joints = std::move(joints);
            return true;
        }

        static Eigen::MatrixXd sketch2rig_joints_to_camera_scene_axes_fit(
            const Eigen::MatrixXd& joints,
            int image_size)
        {
            const int size = std::max(2, image_size);
            const double cx = 0.5 * (double)(size - 1);
            const double cy = 0.5 * (double)(size - 1);
            const double fx = 0.5 * (double)(size - 1) / std::tan(0.5 * (double)kSketch2RigHorizontalFovRad);
            const double fy = 0.5 * (double)(size - 1) / std::tan(0.5 * (double)kSketch2RigVerticalFovRad);

            Eigen::MatrixXd out(joints.rows(), 3);
            out.setZero();
            for (int i = 0; i < joints.rows(); ++i)
            {
                const double px = joints(i, 0);
                const double py = joints(i, 1);
                const double z = joints(i, 2);
                const double u = (px >= -0.25 && px <= 1.25) ? px * (double)(size - 1) : px;
                const double v = (py >= -0.25 && py <= 1.25) ? py * (double)(size - 1) : py;
                out(i, 0) = (u - cx) / fx * z;
                out(i, 1) = -(v - cy) / fy * z;
                out(i, 2) = -z;
            }
            return out;
        }

        static bool rig_height_if_valid_fit(const Eigen::MatrixXd& joints, double* out_height)
        {
            if (!out_height || joints.rows() <= 0)
                return false;
            const double h = joints.col(1).maxCoeff() - joints.col(1).minCoeff();
            if (!(h > 1e-6) || !std::isfinite(h))
                return false;
            *out_height = h;
            return true;
        }

        static double smpl_native_height_from_betas_fit(
            const SmplhModel& model,
            const Eigen::VectorXd& betas)
        {
            const int V = model.vertex_count();
            const int B = model.beta_count();
            if (V <= 0 || model.J_regressor.cols() != V)
                return 1.6;

            Eigen::VectorXd b = Eigen::VectorXd::Zero(std::max(0, B));
            const int n = std::min<int>((int)betas.size(), B);
            for (int i = 0; i < n; ++i)
                b[i] = betas[i];

            Eigen::Map<const MatrixX3dRM> v_template(model.v_template.data(), V, 3);
            Eigen::MatrixXd v_shaped = v_template;
            if (B > 0)
            {
                Eigen::Map<const Eigen::VectorXd> v_template_flat(model.v_template.data(), (size_t)V * 3);
                Eigen::VectorXd v_shaped_flat = v_template_flat;
                v_shaped_flat.noalias() += model.shapedirs * b;
                v_shaped = Eigen::Map<const MatrixX3dRM>(v_shaped_flat.data(), V, 3);
            }

            const Eigen::MatrixXd joints = model.J_regressor * v_shaped;
            double h = 0.0;
            return rig_height_if_valid_fit(joints, &h) ? h : 1.6;
        }

        static bool make_sketch2rig_camera_matrices_fit(
            const std::filesystem::path& rig_path,
            const SmplhModel& model,
            const Eigen::VectorXd& betas,
            float dnear,
            float dfar,
            Eigen::Matrix4f& view,
            Eigen::Matrix4f& proj)
        {
            const std::string text = read_text_file_displacement(rig_path);
            if (text.empty())
                return false;

            // NLF stores a metric camera-space body and its exact 2D
            // projection. Reusing the flat-depth Sketch2Rig joints here would
            // silently replace that camera with the custom pipeline's fixed
            // FOV, causing the fitting render to drift from the sketch.
            std::vector<double> nlf_joints3d;
            std::vector<double> nlf_vertices3d;
            std::vector<double> nlf_vertices2d;
            double nlf_vertex_count_d = 0.0;
            double nlf_image_width_d = 0.0;
            double nlf_image_height_d = 0.0;
            double nlf_image_size_d = 0.0;
            (void)parse_json_number_field_fit(text, "image_size", &nlf_image_size_d);
            if (!parse_json_number_field_fit(text, "image_width", &nlf_image_width_d))
                nlf_image_width_d = nlf_image_size_d;
            if (!parse_json_number_field_fit(text, "image_height", &nlf_image_height_d))
                nlf_image_height_d = nlf_image_size_d;
            if (parse_json_number_array_field_fit(text, "joints3d_flat", &nlf_joints3d)
                && nlf_joints3d.size() >= 3u
                && parse_json_number_array_field_fit(text, "vertices3d_flat", &nlf_vertices3d)
                && parse_json_number_array_field_fit(text, "vertices2d_flat", &nlf_vertices2d)
                && parse_json_number_field_fit(text, "vertices3d_count", &nlf_vertex_count_d))
            {
                const int vertex_count = std::max(0, (int)std::lround(nlf_vertex_count_d));
                if (vertex_count > 0
                    && nlf_image_width_d > 1.0
                    && nlf_image_height_d > 1.0
                    && nlf_vertices3d.size() >= (size_t)vertex_count * 3u
                    && nlf_vertices2d.size() >= (size_t)vertex_count * 2u)
                {
                    const auto fit_axis = [&](int coordinate, double& focal, double& principal)
                    {
                        double sum_p = 0.0;
                        double sum_q = 0.0;
                        double sum_pp = 0.0;
                        double sum_pq = 0.0;
                        int sample_count = 0;
                        for (int i = 0; i < vertex_count; ++i)
                        {
                            const size_t k3 = (size_t)i * 3u;
                            const size_t k2 = (size_t)i * 2u;
                            const double z = nlf_vertices3d[k3 + 2u];
                            if (!(z > 1e-9) || !std::isfinite(z))
                                continue;
                            const double p = nlf_vertices3d[k3 + (size_t)coordinate] / z;
                            const double q = nlf_vertices2d[k2 + (size_t)coordinate];
                            if (!std::isfinite(p) || !std::isfinite(q))
                                continue;
                            sum_p += p;
                            sum_q += q;
                            sum_pp += p * p;
                            sum_pq += p * q;
                            ++sample_count;
                        }
                        const double denom = sum_pp - (sum_p * sum_p) / std::max(1, sample_count);
                        if (sample_count < 3 || std::abs(denom) <= 1e-12 || !std::isfinite(denom))
                            return false;
                        focal = (sum_pq - (sum_p * sum_q) / (double)sample_count) / denom;
                        principal = (sum_q - focal * sum_p) / (double)sample_count;
                        return focal > 1e-6 && std::isfinite(focal) && std::isfinite(principal);
                    };

                    double focal_x = 0.0;
                    double principal_x = 0.0;
                    double focal_y = 0.0;
                    double principal_y = 0.0;
                    if (fit_axis(0, focal_x, principal_x)
                        && fit_axis(1, focal_y, principal_y))
                    {
                        const Eigen::Vector3f eye(
                            (float)(-nlf_joints3d[0] / 1000.0),
                            (float)(nlf_joints3d[1] / 1000.0),
                            (float)(nlf_joints3d[2] / 1000.0));
                        if (eye.allFinite())
                        {
                            igl::look_at(
                                eye,
                                eye + Eigen::Vector3f(0.0f, 0.0f, -1.0f),
                                Eigen::Vector3f(0.0f, 1.0f, 0.0f),
                                view);
                            const float left = (float)(-principal_x / focal_x) * dnear;
                            const float right = (float)((nlf_image_width_d - principal_x) / focal_x) * dnear;
                            const float bottom = (float)((principal_y - nlf_image_height_d) / focal_y) * dnear;
                            const float top = (float)(principal_y / focal_y) * dnear;
                            igl::frustum(left, right, bottom, top, dnear, dfar, proj);
                            return true;
                        }
                    }
                }
            }

            Eigen::MatrixXd joints;
            int image_size = 512;
            if (!parse_sketch2rig_joints_fit(text, &joints, &image_size))
                return false;
            Eigen::MatrixXd camera_joints = sketch2rig_joints_to_camera_scene_axes_fit(joints, image_size);
            if (camera_joints.rows() <= 0)
                return false;

            const Eigen::Vector3d root_offset = camera_joints.row(0).transpose();
            camera_joints.rowwise() -= root_offset.transpose();
            double predicted_height = 0.0;
            double scene_scale = 1.0;
            const double target_height = smpl_native_height_from_betas_fit(model, betas);
            if (rig_height_if_valid_fit(camera_joints, &predicted_height)
                && predicted_height > 1e-6
                && std::isfinite(predicted_height)
                && target_height > 1e-6
                && std::isfinite(target_height))
            {
                scene_scale = std::clamp(target_height / predicted_height, 0.05, 20.0);
            }

            const Eigen::Vector3f eye(
                (float)(-scene_scale * root_offset.x()),
                (float)(-scene_scale * root_offset.y()),
                (float)(-scene_scale * root_offset.z()));
            if (!eye.allFinite())
                return false;

            igl::look_at(
                eye,
                eye + Eigen::Vector3f(0.0f, 0.0f, -1.0f),
                Eigen::Vector3f(0.0f, 1.0f, 0.0f),
                view);

            constexpr float kPi = 3.14159265358979323846f;
            const float f_h = std::tan(kSketch2RigVerticalFovDeg / 360.0f * kPi) * dnear;
            igl::frustum(-f_h, f_h, -f_h, f_h, dnear, dfar, proj);
            return true;
        }

        static std::string parse_json_string_field_displacement(const std::string& text, const char* key)
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

        static bool barycentric_uv(
            double u,
            double v,
            const UvFace& tri,
            Eigen::Vector3d& bc)
        {
            const double v0x = tri.u1 - tri.u0;
            const double v0y = tri.v1 - tri.v0;
            const double v1x = tri.u2 - tri.u0;
            const double v1y = tri.v2 - tri.v0;
            const double v2x = u - tri.u0;
            const double v2y = v - tri.v0;

            const double d00 = v0x * v0x + v0y * v0y;
            const double d01 = v0x * v1x + v0y * v1y;
            const double d11 = v1x * v1x + v1y * v1y;
            const double d20 = v2x * v0x + v2y * v0y;
            const double d21 = v2x * v1x + v2y * v1y;
            const double denom = d00 * d11 - d01 * d01;
            if (!(std::abs(denom) > 1e-20) || !std::isfinite(denom))
                return false;

            const double b = (d11 * d20 - d01 * d21) / denom;
            const double c = (d00 * d21 - d01 * d20) / denom;
            const double a = 1.0 - b - c;
            constexpr double eps = -1e-5;
            if (a < eps || b < eps || c < eps)
                return false;

            bc = Eigen::Vector3d(a, b, c);
            return bc.allFinite();
        }

        template <typename T>
        static float npy_value_as_float(const cnpy::NpyArray& a, size_t idx)
        {
            return (float)a.data<T>()[idx];
        }

        static bool npy_float_at(const cnpy::NpyArray& a, size_t idx, float* out)
        {
            if (!out)
                return false;
            if (a.word_size == sizeof(float))
            {
                *out = npy_value_as_float<float>(a, idx);
                return true;
            }
            if (a.word_size == sizeof(double))
            {
                *out = npy_value_as_float<double>(a, idx);
                return true;
            }
            return false;
        }

        struct UvSourceData
        {
            int width = 0;
            int height = 0;
            std::vector<float> uv;   // [2, height, width]
            std::vector<float> mask; // [height, width]
        };

        struct IsoSourceData
        {
            int width = 0;
            int height = 0;
            std::vector<float> descriptors; // [height, width, 64]
            std::vector<float> mask;        // [height, width]
        };

        struct DepthSourceData
        {
            int width = 0;
            int height = 0;
            float depth_min = 0.0f;
            float depth_max = 1.0f;
            std::vector<float> depth;
            std::vector<float> mask;
        };

        static std::string lowercase_extension(const std::filesystem::path& path)
        {
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                [](unsigned char c) { return (char)std::tolower(c); });
            return ext;
        }

        static bool load_uv_source_png(
            const std::filesystem::path& path,
            UvSourceData& out,
            std::string* out_error)
        {
            int iw = 0, ih = 0, comp = 0;
            stbi_uc* data = stbi_load(path.string().c_str(), &iw, &ih, &comp, 4);
            if (!data)
            {
                if (out_error)
                    *out_error = "could not load UV PNG: " + path.string();
                return false;
            }

            if (iw <= 0 || ih <= 0)
            {
                stbi_image_free(data);
                if (out_error)
                    *out_error = "invalid UV PNG dimensions: " + path.string();
                return false;
            }

            const size_t count = (size_t)iw * (size_t)ih;
            out.width = iw;
            out.height = ih;
            out.uv.assign(count * 2u, 0.0f);
            out.mask.assign(count, 0.0f);

            const bool has_alpha = comp >= 4;
            for (size_t i = 0; i < count; ++i)
            {
                const unsigned char r = data[i * 4u + 0u];
                const unsigned char g = data[i * 4u + 1u];
                const unsigned char b = data[i * 4u + 2u];
                const unsigned char a = data[i * 4u + 3u];
                out.uv[i] = (float)r / 255.0f;
                out.uv[count + i] = (float)g / 255.0f;
                out.mask[i] = has_alpha
                    ? (float)a / 255.0f
                    : ((r != 0u || g != 0u || b != 0u) ? 1.0f : 0.0f);
            }

            stbi_image_free(data);
            return true;
        }

        static bool load_uv_source_json(
            const std::filesystem::path& path,
            UvSourceData& out,
            std::string* out_error)
        {
            const std::string uv_json = read_text_file_displacement(path);
            if (uv_json.empty())
            {
                if (out_error)
                    *out_error = "could not read " + path.string();
                return false;
            }

            std::filesystem::path uv_npz_path = parse_json_string_field_displacement(uv_json, "uv_npz_path");
            if (uv_npz_path.empty())
                uv_npz_path = path.parent_path() / (path.stem().string() + "_uv.npz");
            if (uv_npz_path.is_relative())
                uv_npz_path = path.parent_path() / uv_npz_path;

            cnpy::NpyArray uv_arr;
            cnpy::NpyArray mask_arr;
            try
            {
                uv_arr = cnpy::npz_load(uv_npz_path.string(), "uv");
                mask_arr = cnpy::npz_load(uv_npz_path.string(), "mask");
            }
            catch (const std::exception& e)
            {
                if (out_error)
                    *out_error = std::string("could not load UV NPZ: ") + e.what();
                return false;
            }

            if (uv_arr.shape.size() != 3 || uv_arr.shape[0] < 2 || mask_arr.shape.size() != 2)
            {
                if (out_error)
                    *out_error = "unexpected UV NPZ shape";
                return false;
            }

            const int uv_h = (int)uv_arr.shape[1];
            const int uv_w = (int)uv_arr.shape[2];
            if (uv_w <= 0 || uv_h <= 0 || (int)mask_arr.shape[0] != uv_h || (int)mask_arr.shape[1] != uv_w)
            {
                if (out_error)
                    *out_error = "UV and mask dimensions do not match";
                return false;
            }
            if (!((uv_arr.word_size == sizeof(float) || uv_arr.word_size == sizeof(double))
                    && (mask_arr.word_size == sizeof(float) || mask_arr.word_size == sizeof(double))))
            {
                if (out_error)
                    *out_error = "UV NPZ arrays are not float arrays";
                return false;
            }

            const size_t count = (size_t)uv_w * (size_t)uv_h;
            out.width = uv_w;
            out.height = uv_h;
            out.uv.assign(count * 2u, 0.0f);
            out.mask.assign(count, 0.0f);
            for (size_t i = 0; i < count; ++i)
            {
                if (!npy_float_at(mask_arr, i, &out.mask[i])
                    || !npy_float_at(uv_arr, i, &out.uv[i])
                    || !npy_float_at(uv_arr, count + i, &out.uv[count + i]))
                {
                    if (out_error)
                        *out_error = "failed to read UV NPZ arrays";
                    return false;
                }
            }
            return true;
        }

        static bool load_uv_source(
            const std::filesystem::path& path,
            UvSourceData& out,
            std::string* out_error)
        {
            if (path.empty())
            {
                if (out_error)
                    *out_error = "missing UV source path";
                return false;
            }

            const std::string ext = lowercase_extension(path);
            if (ext == ".png")
                return load_uv_source_png(path, out, out_error);
            return load_uv_source_json(path, out, out_error);
        }

        static float npy_numeric_value(const cnpy::NpyArray& a, size_t idx)
        {
            if (a.word_size == sizeof(float))
                return a.data<float>()[idx];
            if (a.word_size == sizeof(double))
                return (float)a.data<double>()[idx];
            if (a.word_size == sizeof(std::uint8_t))
                return a.data<std::uint8_t>()[idx] != 0u ? 1.0f : 0.0f;
            return 0.0f;
        }

        static bool load_iso_source_npz(
            const std::filesystem::path& path,
            IsoSourceData& out,
            std::string* out_error)
        {
            if (path.empty())
            {
                if (out_error)
                    *out_error = "missing ISO source path";
                return false;
            }

            cnpy::NpyArray isomap;
            cnpy::NpyArray mask;
            try
            {
                isomap = cnpy::npz_load(path.string(), "isomap");
                mask = cnpy::npz_load(path.string(), "mask");
            }
            catch (const std::exception& e)
            {
                if (out_error)
                    *out_error = std::string("could not load ISO NPZ: ") + e.what();
                return false;
            }

            if (isomap.shape.size() != 3
                || (isomap.word_size != sizeof(float) && isomap.word_size != sizeof(double)))
            {
                if (out_error)
                    *out_error = "ISO NPZ must contain a float isomap array shaped [H,W,64] or [64,H,W]";
                return false;
            }

            const bool hwc = isomap.shape[2] == kIsoDimensions;
            const bool chw = !hwc && isomap.shape[0] == kIsoDimensions;
            if (!hwc && !chw)
            {
                if (out_error)
                    *out_error = "ISO NPZ isomap must have 64 descriptor channels";
                return false;
            }

            const int iso_h = (int)(hwc ? isomap.shape[0] : isomap.shape[1]);
            const int iso_w = (int)(hwc ? isomap.shape[1] : isomap.shape[2]);
            if (iso_w <= 0 || iso_h <= 0
                || mask.shape.size() != 2
                || (int)mask.shape[0] != iso_h
                || (int)mask.shape[1] != iso_w)
            {
                if (out_error)
                    *out_error = "ISO NPZ mask dimensions do not match isomap";
                return false;
            }

            const size_t pixel_count = (size_t)iso_w * (size_t)iso_h;
            out.width = iso_w;
            out.height = iso_h;
            out.descriptors.assign(pixel_count * (size_t)kIsoDimensions, 0.0f);
            out.mask.assign(pixel_count, 0.0f);
            for (size_t pixel = 0; pixel < pixel_count; ++pixel)
            {
                out.mask[pixel] = npy_numeric_value(mask, pixel);
                for (int channel = 0; channel < kIsoDimensions; ++channel)
                {
                    const size_t source = hwc
                        ? pixel * (size_t)kIsoDimensions + (size_t)channel
                        : (size_t)channel * pixel_count + pixel;
                    out.descriptors[pixel * (size_t)kIsoDimensions + (size_t)channel] =
                        npy_numeric_value(isomap, source);
                }
            }
            return true;
        }

        static bool load_depth_source(
            const std::filesystem::path& path,
            DepthSourceData& out,
            std::string* out_error)
        {
            if (path.empty())
            {
                if (out_error)
                    *out_error = "missing depth prediction path";
                return false;
            }

            std::filesystem::path npz_path = path;
            if (lowercase_extension(path) == ".json")
            {
                const std::string json = read_text_file_displacement(path);
                if (json.empty())
                {
                    if (out_error)
                        *out_error = "could not read depth prediction: " + path.string();
                    return false;
                }
                npz_path = parse_json_string_field_displacement(json, "depth_npz_path");
                if (npz_path.empty())
                    npz_path = path.parent_path() / (path.stem().string() + "_depth.npz");
                else if (npz_path.is_relative())
                    npz_path = path.parent_path() / npz_path;
            }

            cnpy::NpyArray depth;
            cnpy::NpyArray mask;
            cnpy::NpyArray depth_min;
            cnpy::NpyArray depth_max;
            try
            {
                depth = cnpy::npz_load(npz_path.string(), "depth");
                mask = cnpy::npz_load(npz_path.string(), "mask");
                depth_min = cnpy::npz_load(npz_path.string(), "depth_min");
                depth_max = cnpy::npz_load(npz_path.string(), "depth_max");
            }
            catch (const std::exception& e)
            {
                if (out_error)
                    *out_error = std::string("could not load depth NPZ: ") + e.what();
                return false;
            }

            if (depth.shape.size() != 2 || mask.shape.size() != 2
                || depth.shape != mask.shape
                || depth.shape[0] == 0 || depth.shape[1] == 0
                || (depth.word_size != sizeof(float) && depth.word_size != sizeof(double))
                || (mask.word_size != sizeof(float) && mask.word_size != sizeof(double)))
            {
                if (out_error)
                    *out_error = "depth NPZ must contain matching float depth and mask arrays shaped [H,W]";
                return false;
            }
            if (depth_min.num_vals == 0 || depth_max.num_vals == 0)
            {
                if (out_error)
                    *out_error = "depth NPZ is missing its camera-depth range";
                return false;
            }

            out.height = (int)depth.shape[0];
            out.width = (int)depth.shape[1];
            out.depth_min = npy_numeric_value(depth_min, 0);
            out.depth_max = npy_numeric_value(depth_max, 0);
            if (!std::isfinite(out.depth_min) || !std::isfinite(out.depth_max)
                || !(out.depth_max > out.depth_min))
            {
                if (out_error)
                    *out_error = "depth NPZ has an invalid camera-depth range";
                return false;
            }

            const size_t count = (size_t)out.width * (size_t)out.height;
            out.depth.resize(count);
            out.mask.resize(count);
            for (size_t i = 0; i < count; ++i)
            {
                out.depth[i] = npy_numeric_value(depth, i);
                out.mask[i] = npy_numeric_value(mask, i);
            }
            return true;
        }

        static bool make_target_camera_matrices(
            const AppState& state,
            float dnear,
            float dfar,
            Eigen::Matrix4f& view,
            Eigen::Matrix4f& proj)
        {
            if (!state.scene_view_target_valid)
                return false;

            Eigen::Vector3f eye(
                state.scene_view_target_eye[0],
                state.scene_view_target_eye[1],
                state.scene_view_target_eye[2]);
            Eigen::Vector3f dir(
                state.scene_view_target_direction[0],
                state.scene_view_target_direction[1],
                state.scene_view_target_direction[2]);
            Eigen::Vector3f up(
                state.scene_view_target_up[0],
                state.scene_view_target_up[1],
                state.scene_view_target_up[2]);

            if (!eye.allFinite() || !dir.allFinite() || !up.allFinite())
                return false;
            if (!(dir.squaredNorm() > 1e-12f))
                return false;
            dir.normalize();
            up -= dir * up.dot(dir);
            if (!(up.squaredNorm() > 1e-12f))
                up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
            up.normalize();

            igl::look_at(eye, eye + dir, up, view);

            constexpr float kPi = 3.14159265358979323846f;
            const float fov_degrees =
                (state.scene_view_target_fov_degrees > 1.0f
                    && std::isfinite(state.scene_view_target_fov_degrees))
                ? state.scene_view_target_fov_degrees
                : kSketch2RigVerticalFovDeg;
            const float f_h = std::tan(fov_degrees / 360.0f * kPi) * dnear;
            igl::frustum(-f_h, f_h, -f_h, f_h, dnear, dfar, proj);
            return true;
        }

        static bool project_to_image(
            const Eigen::Matrix4f& view,
            const Eigen::Matrix4f& proj,
            const Eigen::Vector3d& p,
            int w,
            int h,
            float& x,
            float& y)
        {
            const Eigen::Vector4f hp((float)p.x(), (float)p.y(), (float)p.z(), 1.0f);
            const Eigen::Vector4f clip = (proj * view) * hp;
            if (!(clip.w() > 1e-8f) || !std::isfinite(clip.w()))
                return false;

            const Eigen::Vector3f ndc = clip.head<3>() / clip.w();
            if (!ndc.allFinite())
                return false;
            x = (ndc.x() * 0.5f + 0.5f) * (float)w;
            y = (-ndc.y() * 0.5f + 0.5f) * (float)h;
            return true;
        }

        static bool project_to_image_depth(
            const Eigen::Matrix4f& view,
            const Eigen::Matrix4f& proj,
            const Eigen::Vector3d& p,
            int w,
            int h,
            float& x,
            float& y,
            float& depth01)
        {
            const Eigen::Vector4f hp((float)p.x(), (float)p.y(), (float)p.z(), 1.0f);
            const Eigen::Vector4f clip = (proj * view) * hp;
            if (!(std::abs(clip.w()) > 1e-8f) || !std::isfinite(clip.w()))
                return false;

            const Eigen::Vector3f ndc = clip.head<3>() / clip.w();
            if (!ndc.allFinite() || ndc.z() < -1.0f || ndc.z() > 1.0f)
                return false;

            x = (ndc.x() * 0.5f + 0.5f) * (float)w;
            y = (-ndc.y() * 0.5f + 0.5f) * (float)h;
            depth01 = ndc.z() * 0.5f + 0.5f;
            return std::isfinite(x) && std::isfinite(y) && std::isfinite(depth01);
        }

        static float camera_distance_from_depth01(const Eigen::Matrix4f& proj, float depth01)
        {
            const float ndc_z = depth01 * 2.0f - 1.0f;
            const float denominator = ndc_z * proj(3, 2) - proj(2, 2);
            if (!(std::abs(denominator) > 1e-8f))
                return std::numeric_limits<float>::infinity();
            const float view_z = (proj(2, 3) - ndc_z * proj(3, 3)) / denominator;
            const float distance = -view_z;
            return distance > 0.0f && std::isfinite(distance)
                ? distance
                : std::numeric_limits<float>::infinity();
        }

        static void set_uniform_matrix4_if_present(GLuint program, const char* name, const Eigen::Matrix4f& value)
        {
            const GLint loc = glGetUniformLocation(program, name);
            if (loc >= 0)
                glUniformMatrix4fv(loc, 1, GL_FALSE, value.data());
        }

        static void set_uniform_1f_if_present(GLuint program, const char* name, float value)
        {
            const GLint loc = glGetUniformLocation(program, name);
            if (loc >= 0)
                glUniform1f(loc, value);
        }

        static void set_uniform_1i_if_present(GLuint program, const char* name, int value)
        {
            const GLint loc = glGetUniformLocation(program, name);
            if (loc >= 0)
                glUniform1i(loc, value);
        }

        static void insert_depth_layer(float depth, float& first, float& second)
        {
            if (!std::isfinite(depth) || depth < 0.0f || depth > 1.0f)
                return;

            constexpr float kSameLayerEps = 1e-5f;
            if (std::abs(depth - first) <= kSameLayerEps || std::abs(depth - second) <= kSameLayerEps)
                return;

            if (depth < first)
            {
                second = first;
                first = depth;
                return;
            }
            if (depth < second)
                second = depth;
        }

        static void rasterize_depth_layers(
            const Eigen::MatrixXd& vertices,
            const Eigen::MatrixXi& faces,
            const Eigen::Matrix4f& view,
            const Eigen::Matrix4f& proj,
            int width,
            int height,
            std::vector<float>& first_depth,
            std::vector<float>& second_depth)
        {
            first_depth.assign((size_t)width * (size_t)height, 1.0f);
            second_depth.assign((size_t)width * (size_t)height, 1.0f);

            if (width <= 0 || height <= 0 || vertices.rows() <= 0 || faces.rows() <= 0)
                return;

            for (int fi = 0; fi < faces.rows(); ++fi)
            {
                const int i0 = faces(fi, 0);
                const int i1 = faces(fi, 1);
                const int i2 = faces(fi, 2);
                if (i0 < 0 || i1 < 0 || i2 < 0
                    || i0 >= vertices.rows()
                    || i1 >= vertices.rows()
                    || i2 >= vertices.rows())
                    continue;

                float x0 = 0.0f;
                float y0 = 0.0f;
                float z0 = 1.0f;
                float x1 = 0.0f;
                float y1 = 0.0f;
                float z1 = 1.0f;
                float x2 = 0.0f;
                float y2 = 0.0f;
                float z2 = 1.0f;
                if (!project_to_image_depth(view, proj, vertices.row(i0).transpose(), width, height, x0, y0, z0)
                    || !project_to_image_depth(view, proj, vertices.row(i1).transpose(), width, height, x1, y1, z1)
                    || !project_to_image_depth(view, proj, vertices.row(i2).transpose(), width, height, x2, y2, z2))
                    continue;

                const float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
                if (!(std::abs(area) > 1e-8f) || !std::isfinite(area))
                    continue;

                const int min_x = std::clamp((int)std::floor(std::min({ x0, x1, x2 })), 0, width - 1);
                const int max_x = std::clamp((int)std::ceil(std::max({ x0, x1, x2 })), 0, width - 1);
                const int min_y = std::clamp((int)std::floor(std::min({ y0, y1, y2 })), 0, height - 1);
                const int max_y = std::clamp((int)std::ceil(std::max({ y0, y1, y2 })), 0, height - 1);

                for (int py = min_y; py <= max_y; ++py)
                {
                    const float sy = (float)py + 0.5f;
                    for (int px = min_x; px <= max_x; ++px)
                    {
                        const float sx = (float)px + 0.5f;
                        const float w0 = ((x1 - sx) * (y2 - sy) - (y1 - sy) * (x2 - sx)) / area;
                        const float w1 = ((x2 - sx) * (y0 - sy) - (y2 - sy) * (x0 - sx)) / area;
                        const float w2 = 1.0f - w0 - w1;
                        constexpr float kEdgeEps = -1e-5f;
                        if (w0 < kEdgeEps || w1 < kEdgeEps || w2 < kEdgeEps)
                            continue;

                        const float depth = w0 * z0 + w1 * z1 + w2 * z2;
                        const size_t idx = (size_t)(height - 1 - py) * (size_t)width + (size_t)px;
                        insert_depth_layer(depth, first_depth[idx], second_depth[idx]);
                    }
                }
            }
        }

        static void nearest_vertex_distances(
            const Eigen::MatrixXd& cloth_vertices,
            const Eigen::MatrixXd& body_vertices,
            std::vector<float>& cloth_to_body,
            std::vector<float>& body_to_cloth,
            std::vector<int>* body_to_cloth_nearest_vertex = nullptr)
        {
            std::vector<double> cloth_squared(
                (size_t)cloth_vertices.rows(),
                std::numeric_limits<double>::infinity());
            std::vector<double> body_squared(
                (size_t)body_vertices.rows(),
                std::numeric_limits<double>::infinity());
            if (body_to_cloth_nearest_vertex)
                body_to_cloth_nearest_vertex->assign((size_t)body_vertices.rows(), -1);
            for (int cloth_vertex = 0; cloth_vertex < cloth_vertices.rows(); ++cloth_vertex)
            {
                for (int body_vertex = 0; body_vertex < body_vertices.rows(); ++body_vertex)
                {
                    const double dx = cloth_vertices(cloth_vertex, 0) - body_vertices(body_vertex, 0);
                    const double dy = cloth_vertices(cloth_vertex, 1) - body_vertices(body_vertex, 1);
                    const double dz = cloth_vertices(cloth_vertex, 2) - body_vertices(body_vertex, 2);
                    const double squared = dx * dx + dy * dy + dz * dz;
                    cloth_squared[(size_t)cloth_vertex] =
                        std::min(cloth_squared[(size_t)cloth_vertex], squared);
                    if (squared < body_squared[(size_t)body_vertex])
                    {
                        body_squared[(size_t)body_vertex] = squared;
                        if (body_to_cloth_nearest_vertex)
                            (*body_to_cloth_nearest_vertex)[(size_t)body_vertex] = cloth_vertex;
                    }
                }
            }

            cloth_to_body.resize(cloth_squared.size());
            for (size_t i = 0; i < cloth_squared.size(); ++i)
                cloth_to_body[i] = (float)std::sqrt(cloth_squared[i]);
            body_to_cloth.resize(body_squared.size());
            for (size_t i = 0; i < body_squared.size(); ++i)
                body_to_cloth[i] = (float)std::sqrt(body_squared[i]);
        }

        static std::vector<Eigen::Vector3d> averaged_vertex_normals(
            const Eigen::MatrixXd& vertices,
            const Eigen::MatrixXi& faces)
        {
            std::vector<Eigen::Vector3d> normals(
                (size_t)std::max(0, (int)vertices.rows()),
                Eigen::Vector3d::Zero());
            if (vertices.rows() <= 0 || faces.rows() <= 0)
                return normals;

            for (int fi = 0; fi < faces.rows(); ++fi)
            {
                const int i0 = faces(fi, 0);
                const int i1 = faces(fi, 1);
                const int i2 = faces(fi, 2);
                if (i0 < 0 || i1 < 0 || i2 < 0
                    || i0 >= vertices.rows() || i1 >= vertices.rows() || i2 >= vertices.rows())
                    continue;

                const Eigen::Vector3d v0 = vertices.row(i0).transpose();
                const Eigen::Vector3d v1 = vertices.row(i1).transpose();
                const Eigen::Vector3d v2 = vertices.row(i2).transpose();
                Eigen::Vector3d face_normal = (v1 - v0).cross(v2 - v0);
                const double len = face_normal.norm();
                if (!(len > 1e-12) || !std::isfinite(len))
                    continue;
                face_normal /= len;
                normals[(size_t)i0] += face_normal;
                normals[(size_t)i1] += face_normal;
                normals[(size_t)i2] += face_normal;
            }

            for (Eigen::Vector3d& normal : normals)
            {
                const double len = normal.norm();
                if (len > 1e-12 && std::isfinite(len))
                    normal /= len;
                else
                    normal.setZero();
            }
            return normals;
        }

    }

    void Renderer::process_canva_displacement_map(AppState& state)
    {
        const std::uint64_t request_id = state.canva_displacement_request_id;
        if (request_id == 0 || request_id == m_last_canva_displacement_request_id)
            return;

        m_last_canva_displacement_request_id = request_id;
        state.canva_displacement_completed_id = request_id;
        state.canva_displacement_path.clear();

        auto fail = [&](const std::string& msg)
        {
            console_add_line(state, "Canva displacement failed: " + msg);
        };

        if (!state.scene_view_target_valid)
        {
            fail("predicted camera view is not available");
            return;
        }
        if (!m_smplh_loaded || m_smplh_vertices.rows() <= 0 || m_smplh_model.F.rows() <= 0)
        {
            fail("SMPL mesh is not ready");
            return;
        }
        if (!m_smpl_uv_layout_loaded && !try_load_smpl_uv_layout_once())
        {
            fail("SMPL UV layout is not available");
            return;
        }
        if (m_smpl_F_uv.rows() != m_smplh_model.F.rows() || m_smpl_uv_coords.rows() <= 0)
        {
            fail("SMPL UV layout does not match the loaded SMPL mesh");
            return;
        }

        std::filesystem::path uv_source_path = state.canva_displacement_uv_json_path;
        if (uv_source_path.empty())
        {
            fail("missing UV source path");
            return;
        }

        UvSourceData uv_source;
        std::string uv_source_error;
        if (!load_uv_source(uv_source_path, uv_source, &uv_source_error))
        {
            fail(uv_source_error.empty() ? "could not load UV source" : uv_source_error);
            return;
        }

        const int uv_h = uv_source.height;
        const int uv_w = uv_source.width;

        Eigen::Matrix4f view;
        Eigen::Matrix4f proj;
        const auto& core = m_viewer.core();
        if (!make_target_camera_matrices(state, core.camera_dnear, core.camera_dfar, view, proj))
        {
            fail("could not build predicted camera matrices");
            return;
        }

        std::vector<UvFace> faces;
        faces.reserve((size_t)m_smplh_model.F.rows());
        constexpr int kGrid = 64;
        std::vector<std::vector<int>> grid((size_t)kGrid * (size_t)kGrid);

        for (int fi = 0; fi < m_smpl_F_uv.rows(); ++fi)
        {
            const int t0 = m_smpl_F_uv(fi, 0);
            const int t1 = m_smpl_F_uv(fi, 1);
            const int t2 = m_smpl_F_uv(fi, 2);
            if (t0 < 0 || t1 < 0 || t2 < 0)
                continue;
            if (t0 >= m_smpl_uv_coords.rows() || t1 >= m_smpl_uv_coords.rows() || t2 >= m_smpl_uv_coords.rows())
                continue;

            UvFace f;
            f.face = fi;
            f.u0 = m_smpl_uv_coords(t0, 0);
            f.v0 = m_smpl_uv_coords(t0, 1);
            f.u1 = m_smpl_uv_coords(t1, 0);
            f.v1 = m_smpl_uv_coords(t1, 1);
            f.u2 = m_smpl_uv_coords(t2, 0);
            f.v2 = m_smpl_uv_coords(t2, 1);
            const int idx = (int)faces.size();
            faces.push_back(f);

            const double min_u = std::min({ f.u0, f.u1, f.u2 });
            const double max_u = std::max({ f.u0, f.u1, f.u2 });
            const double min_v = std::min({ f.v0, f.v1, f.v2 });
            const double max_v = std::max({ f.v0, f.v1, f.v2 });
            const int gx0 = std::clamp((int)std::floor(min_u * kGrid), 0, kGrid - 1);
            const int gx1 = std::clamp((int)std::floor(max_u * kGrid), 0, kGrid - 1);
            const int gy0 = std::clamp((int)std::floor(min_v * kGrid), 0, kGrid - 1);
            const int gy1 = std::clamp((int)std::floor(max_v * kGrid), 0, kGrid - 1);
            for (int gy = gy0; gy <= gy1; ++gy)
            {
                for (int gx = gx0; gx <= gx1; ++gx)
                    grid[(size_t)gy * (size_t)kGrid + (size_t)gx].push_back(idx);
            }
        }

        if (faces.empty())
        {
            fail("SMPL UV layout has no usable triangles");
            return;
        }

        std::vector<unsigned char> out((size_t)uv_w * (size_t)uv_h * 4u, 0u);
        int projected_count = 0;
        int foreground_count = 0;
        int outside_body_uv_count = 0;

        for (int y = 0; y < uv_h; ++y)
        {
            for (int x = 0; x < uv_w; ++x)
            {
                const size_t pix = (size_t)y * (size_t)uv_w + (size_t)x;
                float mask = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                mask = uv_source.mask[pix];
                u = uv_source.uv[pix];
                v = uv_source.uv[(size_t)uv_h * (size_t)uv_w + pix];
                if (!(mask > 0.5f) || !std::isfinite(u) || !std::isfinite(v))
                    continue;

                ++foreground_count;
                u = std::clamp(u, 0.0f, 1.0f);
                v = std::clamp(v, 0.0f, 1.0f);

                const int gx = std::clamp((int)std::floor((double)u * kGrid), 0, kGrid - 1);
                const int gy = std::clamp((int)std::floor((double)v * kGrid), 0, kGrid - 1);
                const std::vector<int>& candidates = grid[(size_t)gy * (size_t)kGrid + (size_t)gx];

                Eigen::Vector3d bc;
                int face_idx = -1;
                for (const int ci : candidates)
                {
                    if (ci < 0 || ci >= (int)faces.size())
                        continue;
                    if (barycentric_uv((double)u, (double)v, faces[(size_t)ci], bc))
                    {
                        face_idx = faces[(size_t)ci].face;
                        break;
                    }
                }
                if (face_idx < 0)
                {
                    unsigned char* dst = out.data() + (pix * 4u);
                    dst[0] = 0u;
                    dst[1] = 0u;
                    dst[2] = 255u;
                    dst[3] = 255u;
                    ++outside_body_uv_count;
                    continue;
                }

                const int ia = m_smplh_model.F(face_idx, 0);
                const int ib = m_smplh_model.F(face_idx, 1);
                const int ic = m_smplh_model.F(face_idx, 2);
                if (ia < 0 || ib < 0 || ic < 0
                    || ia >= m_smplh_vertices.rows()
                    || ib >= m_smplh_vertices.rows()
                    || ic >= m_smplh_vertices.rows())
                    continue;

                const Eigen::Vector3d p =
                    bc.x() * m_smplh_vertices.row(ia).transpose()
                    + bc.y() * m_smplh_vertices.row(ib).transpose()
                    + bc.z() * m_smplh_vertices.row(ic).transpose();

                float px = 0.0f;
                float py = 0.0f;
                if (!project_to_image(view, proj, p, uv_w, uv_h, px, py))
                    continue;

                const float dx = px - ((float)x + 0.5f);
                const float dy = py - ((float)y + 0.5f);
                unsigned char* dst = out.data() + (pix * 4u);
                dst[0] = (unsigned char)std::clamp<int>((int)std::lround(std::abs(dx)), 0, 255);
                dst[1] = (unsigned char)std::clamp<int>((int)std::lround(std::abs(dy)), 0, 255);
                dst[2] = 0u;
                dst[3] = 255u;
                ++projected_count;
            }
        }

        std::error_code ec;
        const std::filesystem::path renders_dir = std::filesystem::path("renders");
        std::filesystem::create_directories(renders_dir, ec);
        if (ec)
        {
            fail("could not create renders directory: " + ec.message());
            return;
        }

        const std::filesystem::path out_path = renders_dir / "canva_displacement.png";
        const int ok = stbi_write_png(out_path.string().c_str(), uv_w, uv_h, 4, out.data(), uv_w * 4);
        if (ok == 0)
        {
            fail("could not write " + out_path.string());
            return;
        }

        state.canva_displacement_path = out_path.string();
        console_add_line(
            state,
            "Canva displacement saved: " + out_path.string()
                + " (projected=" + std::to_string(projected_count)
                + ", outside_body_uv=" + std::to_string(outside_body_uv_count)
                + "/" + std::to_string(foreground_count) + ")");
    }

    bool Renderer::fit_cloth_render_to_smpl_visibility(
        AppState& state,
        const std::vector<unsigned char>& cloth_rgba,
        int width,
        int height,
        const std::string& iso_npz_path_string,
        const std::string& depth_prediction_path_string,
        const std::string& rig_json_path_string,
        float prediction_from_cloth_scale,
        float prediction_from_cloth_offset_x,
        float prediction_from_cloth_offset_y,
        ClothFittingLayers& out_layers,
        std::string* out_error)
    {
        auto& out_dumb_render_rgba = out_layers.dumb_render;
        auto& out_rgba = out_layers.occluded_cloth;
        auto& out_warped_rgba = out_layers.warped_occluded_cloth;
        auto& out_unfiltered_debug_rgba = out_layers.raw_iso_matching;
        auto& out_debug_rgba = out_layers.filtered_iso_matching;
        auto& out_scene_occlusion_rgba = out_layers.true_depth_visibility;
        auto& out_corrected_scene_occlusion_rgba =
            out_layers.corrected_true_depth_visibility;
        auto& out_aligned_predicted_depth_occlusion_rgba =
            out_layers.predicted_depth_visibility;
        auto& out_iso_occlusion_rgba = out_layers.iso_visibility;
        auto& out_unfiltered_occlusion_rgba = out_layers.raw_final_occlusion;
        auto& out_occlusion_rgba = out_layers.filtered_final_occlusion;
        auto& out_deformation_steps_rgba = out_layers.deformation_steps;
        auto& out_tested_count = out_layers.tested_count;
        auto& out_removed_count = out_layers.removed_count;
        std::vector<unsigned char> out_scene_debug_rgba;
        std::vector<unsigned char> out_deformation_rgba;

        out_layers = ClothFittingLayers{};
        out_dumb_render_rgba.clear();
        out_rgba.clear();
        out_layers.warped_cloth.clear();
        out_warped_rgba.clear();
        out_debug_rgba.clear();
        out_unfiltered_debug_rgba.clear();
        out_unfiltered_occlusion_rgba.clear();
        out_occlusion_rgba.clear();
        out_scene_occlusion_rgba.clear();
        out_corrected_scene_occlusion_rgba.clear();
        out_aligned_predicted_depth_occlusion_rgba.clear();
        out_iso_occlusion_rgba.clear();
        out_scene_debug_rgba.clear();
        out_deformation_rgba.clear();
        out_deformation_steps_rgba.clear();
        out_tested_count = 0;
        out_removed_count = 0;
        const auto timing_start = std::chrono::steady_clock::now();
        auto stage_start = timing_start;
        const auto finish_stage = [&](const char* name)
        {
            out_layers.timings_seconds.emplace_back(
                name,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count());
            stage_start = std::chrono::steady_clock::now();
        };

        auto fail = [&](const std::string& msg)
        {
            if (out_error)
                *out_error = msg;
            return false;
        };

        if (width <= 0 || height <= 0)
            return fail("invalid cloth layer size");
        if (cloth_rgba.size() < (size_t)width * (size_t)height * 4u)
            return fail("cloth layer pixel buffer is incomplete");
        if (!(prediction_from_cloth_scale > 1e-6f)
            || !std::isfinite(prediction_from_cloth_scale)
            || !std::isfinite(prediction_from_cloth_offset_x)
            || !std::isfinite(prediction_from_cloth_offset_y))
            return fail("invalid prediction-to-cloth layer transform");
        if (rig_json_path_string.empty() && !state.scene_view_target_valid)
            return fail("predicted camera view is not available");

        ensure_smplh_mesh_loaded(state);
        ensure_garment_mesh_loaded(state);
        handle_garment_posing(state, false);
        apply_cloth_simulation_frame(state);

        if (!m_smplh_loaded || m_smplh_vertices.rows() <= 0 || m_smplh_model.F.rows() <= 0)
            return fail("SMPL mesh is not ready");
        if (!ensure_iso_vertex_descriptors(state))
            return fail("SMPL ISO mapping is not available");
        if (m_iso_vertex_descriptor_count != m_smplh_vertices.rows())
            return fail("SMPL ISO mapping does not match the loaded SMPL mesh");

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() <= 0 || garment.F.rows() <= 0)
            return fail("cloth mesh is not ready");
        auto& smpl = m_viewer.data(m_smplh_mesh_index);
        const Eigen::MatrixXd& smpl_vertices_for_iso =
            (smpl.V.rows() == m_smplh_vertices.rows()) ? smpl.V : m_smplh_vertices;

        std::filesystem::path iso_source_path = iso_npz_path_string;
        if (iso_source_path.empty())
            return fail("missing ISO source path");

        IsoSourceData iso_source;
        std::string iso_source_error;
        if (!load_iso_source_npz(iso_source_path, iso_source, &iso_source_error))
            return fail(iso_source_error.empty() ? "could not load ISO source" : iso_source_error);

        const int iso_h = iso_source.height;
        const int iso_w = iso_source.width;

        DepthSourceData depth_source;
        std::string depth_source_error;
        if (!load_depth_source(depth_prediction_path_string, depth_source, &depth_source_error))
            return fail(depth_source_error.empty() ? "could not load depth prediction" : depth_source_error);

        Eigen::Matrix4f view;
        Eigen::Matrix4f proj;
        const auto& core_ref = m_viewer.core();
        const std::filesystem::path rig_json_path = rig_json_path_string;
        const bool used_rig_camera =
            !rig_json_path.empty()
            && make_sketch2rig_camera_matrices_fit(
                rig_json_path,
                m_smplh_model,
                m_smplh_params.betas,
                core_ref.camera_dnear,
                core_ref.camera_dfar,
                view,
                proj);
        if (!used_rig_camera
            && !make_target_camera_matrices(state, core_ref.camera_dnear, core_ref.camera_dfar, view, proj))
            return fail("could not build predicted camera matrices");
        finish_stage("Input loading and setup");

        GLint old_framebuffer = 0;
        GLint old_viewport[4] = { 0, 0, 0, 0 };
        GLint old_read_buffer = GL_BACK;
        GLboolean old_scissor = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean old_blend = glIsEnabled(GL_BLEND);
        GLboolean old_depth = glIsEnabled(GL_DEPTH_TEST);
        GLboolean old_cull = glIsEnabled(GL_CULL_FACE);
        GLboolean old_depth_mask = GL_TRUE;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &old_framebuffer);
        glGetIntegerv(GL_VIEWPORT, old_viewport);
        glGetIntegerv(GL_READ_BUFFER, &old_read_buffer);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &old_depth_mask);

        GLuint fbo = 0;
        GLuint color = 0;
        GLuint depth = 0;
        glGenFramebuffers(1, &fbo);
        glGenTextures(1, &color);
        glGenRenderbuffers(1, &depth);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glBindTexture(GL_TEXTURE_2D, color);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
            glReadBuffer((GLenum)old_read_buffer);
            glDeleteRenderbuffers(1, &depth);
            glDeleteTextures(1, &color);
            glDeleteFramebuffers(1, &fbo);
            return fail("cloth fitting framebuffer is incomplete");
        }

        auto& core = m_viewer.core();
        const Eigen::Vector4f old_core_viewport = core.viewport;
        const Eigen::Matrix4f old_view = core.view;
        const Eigen::Matrix4f old_proj = core.proj;
        const Eigen::Matrix4f old_norm = core.norm;
        const Eigen::Vector4f old_background = core.background_color;
        const bool old_garment_visible = garment.is_visible;
        const bool old_smpl_visible = smpl.is_visible;

        core.viewport = Eigen::Vector4f(0.0f, 0.0f, (float)width, (float)height);
        core.view = view;
        core.proj = proj;
        core.norm = core.view.inverse().transpose();
        core.background_color = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);

        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (garment.dirty)
        {
            garment.updateGL(garment, garment.invert_normals, garment.meshgl);
            garment.dirty = igl::opengl::MeshGL::DIRTY_NONE;
        }

        garment.set_visible(true);
        if (garment.meshgl.is_initialized)
        {
            garment.meshgl.bind_mesh();
            const GLuint prog = garment.meshgl.shader_mesh;
            set_uniform_matrix4_if_present(prog, "view", core.view);
            set_uniform_matrix4_if_present(prog, "proj", core.proj);
            set_uniform_matrix4_if_present(prog, "normal_matrix", core.norm);
            set_uniform_1f_if_present(prog, "texture_factor", 0.0f);
            set_uniform_1f_if_present(prog, "matcap_factor", 0.0f);
            set_uniform_1f_if_present(prog, "double_sided", 1.0f);
            set_uniform_1i_if_present(prog, "toon_outline_pass", 0);
            set_uniform_1f_if_present(prog, "toon_outline_width", 0.0f);
            garment.meshgl.draw_mesh(true);
        }

        std::vector<float> cloth_depth_gl((size_t)width * (size_t)height, 1.0f);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, cloth_depth_gl.data());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (smpl.dirty)
        {
            smpl.updateGL(smpl, smpl.invert_normals, smpl.meshgl);
            smpl.dirty = igl::opengl::MeshGL::DIRTY_NONE;
        }

        smpl.set_visible(true);
        if (smpl.meshgl.is_initialized)
        {
            smpl.meshgl.bind_mesh();
            const GLuint prog = smpl.meshgl.shader_mesh;
            set_uniform_matrix4_if_present(prog, "view", core.view);
            set_uniform_matrix4_if_present(prog, "proj", core.proj);
            set_uniform_matrix4_if_present(prog, "normal_matrix", core.norm);
            set_uniform_1f_if_present(prog, "texture_factor", 0.0f);
            set_uniform_1f_if_present(prog, "matcap_factor", 0.0f);
            set_uniform_1f_if_present(prog, "double_sided", 1.0f);
            set_uniform_1i_if_present(prog, "toon_outline_pass", 0);
            set_uniform_1f_if_present(prog, "toon_outline_width", 0.0f);
            smpl.meshgl.draw_mesh(true);
        }

        std::vector<float> smpl_depth_gl((size_t)width * (size_t)height, 1.0f);
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, smpl_depth_gl.data());

        garment.is_visible = old_garment_visible;
        smpl.is_visible = old_smpl_visible;
        core.viewport = old_core_viewport;
        core.view = old_view;
        core.proj = old_proj;
        core.norm = old_norm;
        core.background_color = old_background;

        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
        glReadBuffer((GLenum)old_read_buffer);
        glViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
        if (old_scissor)
            glEnable(GL_SCISSOR_TEST);
        else
            glDisable(GL_SCISSOR_TEST);
        if (old_blend)
            glEnable(GL_BLEND);
        else
            glDisable(GL_BLEND);
        if (old_depth)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);
        if (old_cull)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);
        glDepthMask(old_depth_mask);
        glDeleteRenderbuffers(1, &depth);
        glDeleteTextures(1, &color);
        glDeleteFramebuffers(1, &fbo);
        finish_stage("True-depth scene render");

        out_rgba = cloth_rgba;
        out_debug_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_occlusion_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_scene_occlusion_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_corrected_scene_occlusion_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_aligned_predicted_depth_occlusion_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_iso_occlusion_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_scene_debug_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_deformation_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        std::vector<unsigned char> iso_body_mask((size_t)width * (size_t)height, 0u);
        const float depth_margin = std::clamp(state.cloth_fitting_depth_margin, 0.0f, 0.05f);
        const float depth_margin_pred = std::clamp(state.cloth_fitting_depth_margin_pred, 0.0f, 1.0f);

        auto predicted_body_distance_at = [&](int x, int y, float& out_distance)
        {
            const float prediction_x =
                prediction_from_cloth_scale * ((float)x + 0.5f) + prediction_from_cloth_offset_x;
            const float prediction_y =
                prediction_from_cloth_scale * ((float)y + 0.5f) + prediction_from_cloth_offset_y;
            if (prediction_x < 0.0f || prediction_x >= (float)width
                || prediction_y < 0.0f || prediction_y >= (float)height)
                return false;

            const float depth_x = prediction_x / (float)width * (float)depth_source.width - 0.5f;
            const float depth_y = prediction_y / (float)height * (float)depth_source.height - 0.5f;
            const float body_mask = sample_bilinear_scalar(
                depth_source.mask, depth_source.width, depth_source.height, depth_x, depth_y);
            const float normalized_inverse_depth = sample_bilinear_scalar(
                depth_source.depth, depth_source.width, depth_source.height, depth_x, depth_y);
            if (!(body_mask > 0.5f) || !std::isfinite(normalized_inverse_depth))
                return false;

            out_distance = depth_source.depth_max
                - std::clamp(normalized_inverse_depth, 0.0f, 1.0f)
                    * (depth_source.depth_max - depth_source.depth_min);
            return std::isfinite(out_distance);
        };

        struct RunningDepthStats
        {
            double sum = 0.0;
            double sum_squared = 0.0;
            size_t count = 0;

            void add(float value)
            {
                if (!std::isfinite(value))
                    return;
                sum += (double)value;
                sum_squared += (double)value * (double)value;
                ++count;
            }

            float mean() const
            {
                return count > 0 ? (float)(sum / (double)count) : 0.0f;
            }

            float standard_deviation() const
            {
                if (count == 0)
                    return 0.0f;
                const double m = sum / (double)count;
                return (float)std::sqrt(std::max(0.0, sum_squared / (double)count - m * m));
            }
        };

        RunningDepthStats predicted_stats;
        RunningDepthStats body_model_stats;
        struct PairedDepthSample
        {
            float predicted = 0.0f;
            float body_model = 0.0f;
        };
        std::vector<PairedDepthSample> paired_depth_samples;
        const int alignment_sample_stride = std::max(1, std::min(width, height) / 512);
        paired_depth_samples.reserve(
            (size_t)((width + alignment_sample_stride - 1) / alignment_sample_stride)
                * (size_t)((height + alignment_sample_stride - 1) / alignment_sample_stride));
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float predicted_distance = 0.0f;
                const bool have_predicted_depth =
                    predicted_body_distance_at(x, y, predicted_distance);
                if (have_predicted_depth)
                    predicted_stats.add(predicted_distance);

                const size_t depth_idx =
                    (size_t)(height - 1 - y) * (size_t)width + (size_t)x;
                if (smpl_depth_gl[depth_idx] < 0.999999f)
                {
                    const float body_model_distance =
                        camera_distance_from_depth01(proj, smpl_depth_gl[depth_idx]);
                    body_model_stats.add(body_model_distance);
                    if (have_predicted_depth
                        && x % alignment_sample_stride == 0
                        && y % alignment_sample_stride == 0
                        && std::isfinite(body_model_distance))
                    {
                        paired_depth_samples.push_back(
                            { predicted_distance, body_model_distance });
                    }
                }
            }
        }

        const float predicted_mean = predicted_stats.mean();
        const float predicted_stddev = predicted_stats.standard_deviation();
        const float body_model_mean = body_model_stats.mean();
        const float body_model_stddev = body_model_stats.standard_deviation();
        const bool have_alignment_stats = predicted_stats.count > 0 && body_model_stats.count > 0;
        float aligned_depth_scale = have_alignment_stats && predicted_stddev > 1e-6f
            ? body_model_stddev / predicted_stddev
            : 1.0f;
        float aligned_depth_translation = have_alignment_stats
            ? body_model_mean - aligned_depth_scale * predicted_mean
            : 0.0f;
        std::string depth_alignment_method = "global mean/std fallback";
        float paired_depth_correlation = 0.0f;

        if (paired_depth_samples.size() >= 64u)
        {
            std::vector<float> predicted_overlap;
            std::vector<float> body_model_overlap;
            predicted_overlap.reserve(paired_depth_samples.size());
            body_model_overlap.reserve(paired_depth_samples.size());

            double predicted_sum = 0.0;
            double body_model_sum = 0.0;
            for (const PairedDepthSample& sample : paired_depth_samples)
            {
                predicted_overlap.push_back(sample.predicted);
                body_model_overlap.push_back(sample.body_model);
                predicted_sum += sample.predicted;
                body_model_sum += sample.body_model;
            }

            const double predicted_overlap_mean =
                predicted_sum / (double)paired_depth_samples.size();
            const double body_model_overlap_mean =
                body_model_sum / (double)paired_depth_samples.size();
            double covariance = 0.0;
            double predicted_variance = 0.0;
            double body_model_variance = 0.0;
            for (const PairedDepthSample& sample : paired_depth_samples)
            {
                const double predicted_delta =
                    (double)sample.predicted - predicted_overlap_mean;
                const double body_model_delta =
                    (double)sample.body_model - body_model_overlap_mean;
                covariance += predicted_delta * body_model_delta;
                predicted_variance += predicted_delta * predicted_delta;
                body_model_variance += body_model_delta * body_model_delta;
            }
            const double correlation_denominator =
                std::sqrt(predicted_variance * body_model_variance);
            if (correlation_denominator > 1e-12)
            {
                paired_depth_correlation = (float)std::clamp(
                    covariance / correlation_denominator,
                    -1.0,
                    1.0);
            }

            std::sort(predicted_overlap.begin(), predicted_overlap.end());
            std::sort(body_model_overlap.begin(), body_model_overlap.end());
            const auto sorted_percentile = [](const std::vector<float>& values, float q)
            {
                if (values.empty())
                    return 0.0f;
                const float position = std::clamp(q, 0.0f, 1.0f)
                    * (float)(values.size() - 1u);
                const size_t lower = (size_t)std::floor(position);
                const size_t upper = std::min(lower + 1u, values.size() - 1u);
                const float t = position - (float)lower;
                return values[lower] * (1.0f - t) + values[upper] * t;
            };

            const float predicted_low = sorted_percentile(predicted_overlap, 0.1f);
            const float predicted_high = sorted_percentile(predicted_overlap, 0.9f);
            const float body_model_low = sorted_percentile(body_model_overlap, 0.1f);
            const float body_model_high = sorted_percentile(body_model_overlap, 0.9f);
            const float predicted_span = predicted_high - predicted_low;
            const float body_model_span = body_model_high - body_model_low;
            const float paired_scale = body_model_span / std::max(1e-6f, predicted_span);

            if (paired_depth_correlation > 0.0f
                && predicted_span > 1e-5f
                && body_model_span > 1e-5f
                && paired_scale > 0.01f
                && paired_scale < 100.0f
                && std::isfinite(paired_scale))
            {
                std::vector<float> paired_residuals;
                paired_residuals.reserve(paired_depth_samples.size());
                for (const PairedDepthSample& sample : paired_depth_samples)
                {
                    paired_residuals.push_back(
                        sample.body_model - paired_scale * sample.predicted);
                }
                const size_t median_index = paired_residuals.size() / 2u;
                std::nth_element(
                    paired_residuals.begin(),
                    paired_residuals.begin() + (std::ptrdiff_t)median_index,
                    paired_residuals.end());
                const float paired_translation = paired_residuals[median_index];
                if (std::isfinite(paired_translation))
                {
                    aligned_depth_scale = paired_scale;
                    aligned_depth_translation = paired_translation;
                    depth_alignment_method = "overlap p10/p90 + paired median";
                }
            }
        }
        finish_stage("Predicted-depth alignment");

        auto set_visibility_pixel = [](std::vector<unsigned char>& rgba, size_t pixel, bool occluded)
        {
            unsigned char* dst = rgba.data() + pixel * 4u;
            dst[0] = occluded ? 255u : 32u;
            dst[1] = occluded ? 48u : 220u;
            dst[2] = occluded ? 48u : 80u;
            dst[3] = 255u;
        };

        // Keep the depth tests separate here. They are combined with the ISO
        // result only when the final visibility map is built below.
        const size_t visibility_pixel_count = (size_t)width * (size_t)height;
        std::vector<unsigned char> rendered_cloth_coverage(
            visibility_pixel_count,
            0u);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t out_pix = (size_t)y * (size_t)width + (size_t)x;
                const size_t depth_idx =
                    (size_t)(height - 1 - y) * (size_t)width + (size_t)x;
                if (cloth_rgba[out_pix * 4u + 3u] == 0u)
                    continue;
                const float cloth_depth = cloth_depth_gl[depth_idx];
                if (!(cloth_depth < 0.999999f))
                    continue;

                rendered_cloth_coverage[out_pix] = 1u;

                const float cloth_distance = camera_distance_from_depth01(proj, cloth_depth);
                const bool true_depth_occluded =
                    smpl_depth_gl[depth_idx] < 0.999999f
                    && camera_distance_from_depth01(proj, smpl_depth_gl[depth_idx])
                        + depth_margin < cloth_distance;
                set_visibility_pixel(out_scene_occlusion_rgba, out_pix, true_depth_occluded);

                float predicted_distance = 0.0f;
                const bool have_predicted_depth = predicted_body_distance_at(x, y, predicted_distance);
                const float aligned_predicted_distance =
                    aligned_depth_scale * predicted_distance + aligned_depth_translation;
                const bool aligned_predicted_depth_occluded =
                    have_predicted_depth && aligned_predicted_distance + depth_margin_pred < cloth_distance;
                set_visibility_pixel(
                    out_aligned_predicted_depth_occlusion_rgba,
                    out_pix,
                    aligned_predicted_depth_occluded);
            }
        }

        // The inked cloth silhouette extends beyond the rasterized garment by a
        // few pixels. Copy the nearest measured visibility into that outline;
        // defaulting those pixels to visible creates a green halo around red
        // occlusion regions. Keep this completion limited to unmeasured pixels
        // so genuine visible cloth next to the body is not swallowed.
        const int cloth_outline_coverage_radius = std::clamp(
            (std::min(width, height) + 255) / 256,
            2,
            8);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                if (cloth_rgba[pixel * 4u + 3u] == 0u
                    || rendered_cloth_coverage[pixel] != 0u)
                    continue;

                size_t nearest_pixel = visibility_pixel_count;
                int nearest_distance_squared = std::numeric_limits<int>::max();
                for (int offset_y = -cloth_outline_coverage_radius;
                     offset_y <= cloth_outline_coverage_radius;
                     ++offset_y)
                {
                    const int neighbor_y = y + offset_y;
                    if (neighbor_y < 0 || neighbor_y >= height)
                        continue;
                    for (int offset_x = -cloth_outline_coverage_radius;
                         offset_x <= cloth_outline_coverage_radius;
                         ++offset_x)
                    {
                        const int distance_squared =
                            offset_x * offset_x + offset_y * offset_y;
                        if (distance_squared >= nearest_distance_squared)
                            continue;
                        const int neighbor_x = x + offset_x;
                        if (neighbor_x < 0 || neighbor_x >= width)
                            continue;
                        const size_t neighbor =
                            (size_t)neighbor_y * (size_t)width + (size_t)neighbor_x;
                        if (rendered_cloth_coverage[neighbor] == 0u)
                            continue;
                        nearest_pixel = neighbor;
                        nearest_distance_squared = distance_squared;
                    }
                }

                if (nearest_pixel < visibility_pixel_count)
                {
                    std::copy_n(
                        out_scene_occlusion_rgba.data() + nearest_pixel * 4u,
                        4u,
                        out_scene_occlusion_rgba.data() + pixel * 4u);
                    std::copy_n(
                        out_aligned_predicted_depth_occlusion_rgba.data()
                            + nearest_pixel * 4u,
                        4u,
                        out_aligned_predicted_depth_occlusion_rgba.data()
                            + pixel * 4u);
                }
                else
                {
                    set_visibility_pixel(out_scene_occlusion_rgba, pixel, false);
                    set_visibility_pixel(
                        out_aligned_predicted_depth_occlusion_rgba,
                        pixel,
                        false);
                }
            }
        }

        // Close the remaining one- or two-pixel SMPL raster coverage gap at a
        // true body boundary. This is conservative only where body occlusion
        // has already been measured and remains clipped to the raw cloth mask.
        const int true_depth_coverage_radius = std::clamp(
            (std::min(width, height) + 1023) / 1024,
            1,
            2);
        std::vector<unsigned char> expanded_true_depth_occlusion(
            visibility_pixel_count,
            0u);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                if (!rgba_is_color(
                        out_scene_occlusion_rgba,
                        pixel,
                        255u,
                        48u,
                        48u))
                    continue;

                for (int offset_y = -true_depth_coverage_radius;
                     offset_y <= true_depth_coverage_radius;
                     ++offset_y)
                {
                    const int neighbor_y = y + offset_y;
                    if (neighbor_y < 0 || neighbor_y >= height)
                        continue;
                    for (int offset_x = -true_depth_coverage_radius;
                         offset_x <= true_depth_coverage_radius;
                         ++offset_x)
                    {
                        if (offset_x * offset_x + offset_y * offset_y
                            > true_depth_coverage_radius * true_depth_coverage_radius)
                            continue;
                        const int neighbor_x = x + offset_x;
                        if (neighbor_x < 0 || neighbor_x >= width)
                            continue;
                        const size_t neighbor =
                            (size_t)neighbor_y * (size_t)width + (size_t)neighbor_x;
                        if (cloth_rgba[neighbor * 4u + 3u] != 0u)
                            expanded_true_depth_occlusion[neighbor] = 1u;
                    }
                }
            }
        }
        for (size_t pixel = 0; pixel < visibility_pixel_count; ++pixel)
        {
            if (expanded_true_depth_occlusion[pixel] != 0u)
                set_visibility_pixel(out_scene_occlusion_rgba, pixel, true);
        }

        out_dumb_render_rgba = cloth_rgba;
        for (size_t pixel = 0; pixel < visibility_pixel_count; ++pixel)
        {
            if (rgba_is_color(out_scene_occlusion_rgba, pixel, 255u, 48u, 48u))
                out_dumb_render_rgba[pixel * 4u + 3u] = 0u;
        }

        console_add_line(
            state,
            "Cloth fitting predicted-depth alignment: method=" + depth_alignment_method
                + ", scale=" + std::to_string(aligned_depth_scale)
                + ", translation=" + std::to_string(aligned_depth_translation)
                + ", overlap pairs=" + std::to_string(paired_depth_samples.size())
                + ", correlation=" + std::to_string(paired_depth_correlation)
                + ", predicted(mean/std)=" + std::to_string(predicted_mean)
                + "/" + std::to_string(predicted_stddev)
                + ", body(mean/std)=" + std::to_string(body_model_mean)
                + "/" + std::to_string(body_model_stddev));
        finish_stage("True and predicted depth visibility");

        std::vector<float> cloth_depth_cpu;
        std::vector<float> cloth_second_depth_cpu;
        rasterize_depth_layers(
            garment.V,
            garment.F,
            view,
            proj,
            width,
            height,
            cloth_depth_cpu,
            cloth_second_depth_cpu);
        std::vector<float> body_depth_cpu;
        std::vector<float> body_second_depth_cpu;
        rasterize_depth_layers(
            smpl.V,
            smpl.F,
            view,
            proj,
            width,
            height,
            body_depth_cpu,
            body_second_depth_cpu);

        const float body_distance_threshold =
            std::clamp(state.cloth_fitting_body_distance_threshold, 0.0f, 1.0f);
        std::vector<float> cloth_body_vertex_distances;
        std::vector<float> body_cloth_vertex_distances;
        std::vector<int> body_cloth_nearest_vertices;
        nearest_vertex_distances(
            garment.V,
            smpl_vertices_for_iso,
            cloth_body_vertex_distances,
            body_cloth_vertex_distances,
            &body_cloth_nearest_vertices);
        const std::vector<Eigen::Vector3d> smpl_vertex_normals =
            averaged_vertex_normals(smpl_vertices_for_iso, smpl.F);
        std::vector<float> body_cloth_normal_scaled_thresholds(
            body_cloth_vertex_distances.size(),
            0.0f);
        for (int body_vertex = 0; body_vertex < (int)body_cloth_vertex_distances.size(); ++body_vertex)
        {
            const int cloth_vertex =
                body_vertex < (int)body_cloth_nearest_vertices.size()
                    ? body_cloth_nearest_vertices[(size_t)body_vertex]
                    : -1;
            if (cloth_vertex < 0 || cloth_vertex >= garment.V.rows())
                continue;

            const float distance = body_cloth_vertex_distances[(size_t)body_vertex];
            if (!std::isfinite(distance))
                continue;

            float normal_alignment = 1.0f;
            if (body_vertex < (int)smpl_vertex_normals.size()
                && smpl_vertex_normals[(size_t)body_vertex].squaredNorm() > 1e-12)
            {
                const Eigen::Vector3d body_pos =
                    smpl_vertices_for_iso.row(body_vertex).transpose();
                Eigen::Vector3d direction =
                    garment.V.row(cloth_vertex).transpose() - body_pos;
                const double direction_len = direction.norm();
                if (direction_len > 1e-12 && std::isfinite(direction_len))
                {
                    direction /= direction_len;
                    normal_alignment = (float)std::clamp(
                        smpl_vertex_normals[(size_t)body_vertex].dot(direction),
                        0.0,
                        1.0);
                }
            }

            body_cloth_normal_scaled_thresholds[(size_t)body_vertex] =
                body_distance_threshold * normal_alignment;
        }
        std::vector<unsigned char> iso_body_close_to_cloth_mask((size_t)width * (size_t)height, 0u);

        struct SceneHit
        {
            float depth = 1.0f;
            bool body = false;
        };
        auto set_scene_debug_pixel = [&](size_t pix, unsigned char r, unsigned char g, unsigned char b)
        {
            unsigned char* dst = out_scene_debug_rgba.data() + pix * 4u;
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
            dst[3] = 220u;
        };

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t out_pix = (size_t)y * (size_t)width + (size_t)x;
                const size_t depth_idx =
                    (size_t)(height - 1 - y) * (size_t)width + (size_t)x;
                std::array<SceneHit, 4> hits;
                int hit_count = 0;
                auto add_hit = [&](float depth, bool body)
                {
                    if (depth < 0.999999f)
                        hits[(size_t)hit_count++] = SceneHit{ depth, body };
                };
                add_hit(cloth_depth_cpu[depth_idx], false);
                add_hit(cloth_second_depth_cpu[depth_idx], false);
                add_hit(body_depth_cpu[depth_idx], true);
                add_hit(body_second_depth_cpu[depth_idx], true);
                if (hit_count == 0)
                    continue;

                for (int i = 1; i < hit_count; ++i)
                {
                    const SceneHit hit = hits[(size_t)i];
                    int insert_at = i;
                    while (insert_at > 0 && hit.depth < hits[(size_t)insert_at - 1u].depth)
                    {
                        hits[(size_t)insert_at] = hits[(size_t)insert_at - 1u];
                        --insert_at;
                    }
                    hits[(size_t)insert_at] = hit;
                }

                bool coincident = false;
                for (int i = 1; i < hit_count; ++i)
                {
                    if (hits[(size_t)i - 1u].body != hits[(size_t)i].body
                        && hits[(size_t)i].depth - hits[(size_t)i - 1u].depth <= depth_margin)
                    {
                        coincident = true;
                        break;
                    }
                }
                if (coincident)
                {
                    set_scene_debug_pixel(out_pix, 255u, 224u, 0u);
                    continue;
                }

                unsigned int signature = 0u;
                for (int i = 0; i < hit_count; ++i)
                    signature = (signature << 1u) | (hits[(size_t)i].body ? 1u : 0u);
                const unsigned int key = ((unsigned int)hit_count << 4u) | signature;
                switch (key)
                {
                case 0x10u: set_scene_debug_pixel(out_pix, 0u, 188u, 212u); break;      // C
                case 0x11u: set_scene_debug_pixel(out_pix, 96u, 125u, 139u); break;     // B
                case 0x20u: set_scene_debug_pixel(out_pix, 224u, 64u, 251u); break;     // CC
                case 0x21u: set_scene_debug_pixel(out_pix, 32u, 220u, 80u); break;      // CB
                case 0x22u: set_scene_debug_pixel(out_pix, 255u, 48u, 48u); break;      // BC
                case 0x23u: set_scene_debug_pixel(out_pix, 255u, 152u, 0u); break;      // BB
                case 0x31u: set_scene_debug_pixel(out_pix, 102u, 255u, 102u); break;    // CCB
                case 0x32u: set_scene_debug_pixel(out_pix, 0u, 160u, 64u); break;       // CBC
                case 0x33u: set_scene_debug_pixel(out_pix, 178u, 255u, 89u); break;     // CBB
                case 0x34u: set_scene_debug_pixel(out_pix, 255u, 64u, 129u); break;     // BCC
                case 0x35u: set_scene_debug_pixel(out_pix, 213u, 0u, 0u); break;        // BCB
                case 0x36u: set_scene_debug_pixel(out_pix, 255u, 109u, 0u); break;      // BBC
                case 0x43u: set_scene_debug_pixel(out_pix, 0u, 230u, 118u); break;      // CCBB
                case 0x45u: set_scene_debug_pixel(out_pix, 0u, 191u, 165u); break;      // CBCB
                case 0x46u: set_scene_debug_pixel(out_pix, 100u, 221u, 23u); break;     // CBBC
                case 0x49u: set_scene_debug_pixel(out_pix, 255u, 23u, 68u); break;      // BCCB
                case 0x4au: set_scene_debug_pixel(out_pix, 197u, 17u, 98u); break;      // BCBC
                case 0x4cu: set_scene_debug_pixel(out_pix, 170u, 0u, 255u); break;      // BBCC
                default: break;
                }
            }
        }

        auto set_debug_pixel = [&](size_t pix, unsigned char r, unsigned char g, unsigned char b)
        {
            unsigned char* dst = out_debug_rgba.data() + pix * 4u;
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
            dst[3] = 255u;
        };
        for (size_t pixel = 0; pixel < (size_t)width * (size_t)height; ++pixel)
        {
            if (cloth_rgba[pixel * 4u + 3u] != 0u)
                set_debug_pixel(pixel, 32u, 220u, 80u);
        }
        // ISO matching is a binary cloth decision: orange where the matched,
        // projected body surface occludes the cloth and green everywhere else.
        const float iso_distance_threshold = std::max(0.0f, state.iso_distance_threshold);
        const size_t body_vertex_count = (size_t)smpl_vertices_for_iso.rows();
        std::vector<float> body_vertex_screen_x(
            body_vertex_count, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> body_vertex_screen_y(
            body_vertex_count, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> body_vertex_screen_depth(
            body_vertex_count, std::numeric_limits<float>::quiet_NaN());

        // Build a sparse, visibility-aware screen-space body lookup once. The
        // network descriptor remains the primary correspondence, but a match is
        // no longer allowed to jump from torso cloth to an arm elsewhere in the
        // image merely because that arm descriptor has a slightly lower 64-D
        // distance. The nearest projected body vertex provides a local candidate
        // that is scored jointly by descriptor and reprojection distance.
        const size_t output_pixel_count = (size_t)width * (size_t)height;
        std::vector<unsigned char> projected_body_seed_mask(output_pixel_count, 0u);
        std::vector<int> projected_body_seed_vertex(output_pixel_count, -1);
        std::vector<float> projected_body_seed_score(
            output_pixel_count, std::numeric_limits<float>::infinity());
        for (int candidate_vertex = 0;
             candidate_vertex < smpl_vertices_for_iso.rows();
             ++candidate_vertex)
        {
            float projected_x = 0.0f;
            float projected_y = 0.0f;
            float projected_depth = 1.0f;
            const Eigen::Vector3d point =
                smpl_vertices_for_iso.row(candidate_vertex).transpose();
            if (!project_to_image_depth(
                    view,
                    proj,
                    point,
                    width,
                    height,
                    projected_x,
                    projected_y,
                    projected_depth))
                continue;

            body_vertex_screen_x[(size_t)candidate_vertex] = projected_x;
            body_vertex_screen_y[(size_t)candidate_vertex] = projected_y;
            body_vertex_screen_depth[(size_t)candidate_vertex] = projected_depth;
            const int screen_x = (int)std::floor(projected_x);
            const int screen_y = (int)std::floor(projected_y);
            if (screen_x < 0 || screen_y < 0 || screen_x >= width || screen_y >= height)
                continue;

            const size_t screen_pixel =
                (size_t)screen_y * (size_t)width + (size_t)screen_x;
            const size_t depth_pixel =
                (size_t)(height - 1 - screen_y) * (size_t)width + (size_t)screen_x;
            const float visible_depth = smpl_depth_gl[depth_pixel];
            const float depth_error = visible_depth < 0.999999f
                ? std::abs(projected_depth - visible_depth)
                : 1.0f + projected_depth;
            const float dx = projected_x - ((float)screen_x + 0.5f);
            const float dy = projected_y - ((float)screen_y + 0.5f);
            const float seed_score = depth_error * 4096.0f + dx * dx + dy * dy;
            if (seed_score < projected_body_seed_score[screen_pixel])
            {
                projected_body_seed_score[screen_pixel] = seed_score;
                projected_body_seed_vertex[screen_pixel] = candidate_vertex;
                projected_body_seed_mask[screen_pixel] = 1u;
            }
        }
        std::vector<int> nearest_projected_body_x;
        std::vector<int> nearest_projected_body_y;
        const bool have_projected_body_lookup = nearest_mask_pixels(
            projected_body_seed_mask,
            width,
            height,
            nearest_projected_body_x,
            nearest_projected_body_y);

        double direct_projection_cost = 0.0;
        double mirrored_projection_cost = 0.0;
        size_t projection_sample_count = 0u;
        const int projection_sample_stride =
            std::max(1, std::min(width, height) / 64);
        const float maximum_projection_error_squared =
            0.25f * (float)std::min(width, height) * (float)std::min(width, height);
        for (int y = projection_sample_stride / 2;
             y < height;
             y += projection_sample_stride)
        {
            const float prediction_y =
                prediction_from_cloth_scale * ((float)y + 0.5f) + prediction_from_cloth_offset_y;
            if (prediction_y < 0.0f || prediction_y >= (float)height)
                continue;
            const int iso_y = std::clamp(
                (int)std::floor(prediction_y / (float)height * (float)iso_h),
                0,
                iso_h - 1);
            for (int x = projection_sample_stride / 2;
                 x < width;
                 x += projection_sample_stride)
            {
                const float prediction_x =
                    prediction_from_cloth_scale * ((float)x + 0.5f) + prediction_from_cloth_offset_x;
                if (prediction_x < 0.0f || prediction_x >= (float)width)
                    continue;
                const int iso_x = std::clamp(
                    (int)std::floor(prediction_x / (float)width * (float)iso_w),
                    0,
                    iso_w - 1);
                const size_t iso_pixel = (size_t)iso_y * (size_t)iso_w + (size_t)iso_x;
                const size_t descriptor_offset = iso_pixel * (size_t)kIsoDimensions;
                if (iso_pixel >= iso_source.mask.size()
                    || !(iso_source.mask[iso_pixel] > 0.5f)
                    || descriptor_offset + (size_t)kIsoDimensions > iso_source.descriptors.size())
                    continue;

                int direct_vertex = -1;
                float descriptor_distance_squared = 0.0f;
                if (!find_nearest_iso_vertex(
                        iso_source.descriptors.data() + descriptor_offset,
                        direct_vertex,
                        descriptor_distance_squared)
                    || direct_vertex < 0
                    || direct_vertex >= (int)body_vertex_count
                    || direct_vertex >= (int)m_iso_vertex_mirror_indices.size())
                    continue;
                const int mirrored_vertex = m_iso_vertex_mirror_indices[(size_t)direct_vertex];
                if (mirrored_vertex < 0 || mirrored_vertex >= (int)body_vertex_count)
                    continue;

                const float direct_x = body_vertex_screen_x[(size_t)direct_vertex];
                const float direct_y = body_vertex_screen_y[(size_t)direct_vertex];
                const float mirrored_x = body_vertex_screen_x[(size_t)mirrored_vertex];
                const float mirrored_y = body_vertex_screen_y[(size_t)mirrored_vertex];
                if (!std::isfinite(direct_x) || !std::isfinite(direct_y)
                    || !std::isfinite(mirrored_x) || !std::isfinite(mirrored_y))
                    continue;

                const float direct_dx = direct_x - prediction_x;
                const float direct_dy = direct_y - prediction_y;
                const float mirrored_dx = mirrored_x - prediction_x;
                const float mirrored_dy = mirrored_y - prediction_y;
                direct_projection_cost += std::min(
                    maximum_projection_error_squared,
                    direct_dx * direct_dx + direct_dy * direct_dy);
                mirrored_projection_cost += std::min(
                    maximum_projection_error_squared,
                    mirrored_dx * mirrored_dx + mirrored_dy * mirrored_dy);
                ++projection_sample_count;
            }
        }
        const bool mirror_iso_vertices = projection_sample_count >= 16u
            && mirrored_projection_cost < direct_projection_cost;
        m_iso_prediction_mirror_vertices = mirror_iso_vertices;
        m_iso_projection_orientation_npz_path =
            !state.sketch2iso.last_output_path.empty()
            ? state.sketch2iso.last_output_path
            : iso_source_path.string();

        const float spatial_match_radius =
            std::max(12.0f, 0.03f * (float)std::min(width, height));
        const float hard_spatial_match_radius = spatial_match_radius * 1.75f;
        const float local_lookup_radius = spatial_match_radius * 2.5f;
        const float descriptor_cost_scale =
            std::max(0.05f, iso_distance_threshold);
        const float spatial_cost_weight =
            (descriptor_cost_scale * descriptor_cost_scale)
            / (spatial_match_radius * spatial_match_radius);
        const auto descriptor_distance_squared_to_vertex = [&](const float* descriptor, int vertex)
        {
            if (!descriptor || vertex < 0 || vertex >= m_iso_vertex_descriptor_count)
                return std::numeric_limits<float>::infinity();
            const int descriptor_vertex =
                mirror_iso_vertices
                && vertex < (int)m_iso_vertex_mirror_indices.size()
                && m_iso_vertex_mirror_indices[(size_t)vertex] >= 0
                ? m_iso_vertex_mirror_indices[(size_t)vertex]
                : vertex;
            const float* vertex_descriptor = m_iso_vertex_descriptors.data()
                + (size_t)descriptor_vertex * (size_t)kIsoDimensions;
            float distance_squared = 0.0f;
            for (int channel = 0; channel < kIsoDimensions; ++channel)
            {
                const float delta = descriptor[channel] - vertex_descriptor[channel];
                distance_squared += delta * delta;
            }
            return distance_squared;
        };
        size_t retained_descriptor_matches = 0u;
        size_t spatially_corrected_matches = 0u;
        size_t rejected_distant_matches = 0u;
        for (int y = 0; y < height; ++y)
        {
            const float prediction_layer_y =
                prediction_from_cloth_scale * ((float)y + 0.5f) + prediction_from_cloth_offset_y;
            const bool prediction_y_valid =
                prediction_layer_y >= 0.0f && prediction_layer_y < (float)height;
            const int iso_y = prediction_y_valid
                ? std::clamp(
                    (int)std::floor((prediction_layer_y / (float)height) * (float)iso_h),
                    0,
                    iso_h - 1)
                : 0;
            for (int x = 0; x < width; ++x)
            {
                const size_t out_pix = (size_t)y * (size_t)width + (size_t)x;
                const bool source_cloth_pixel = cloth_rgba[out_pix * 4u + 3u] != 0u;

                const float prediction_layer_x =
                    prediction_from_cloth_scale * ((float)x + 0.5f) + prediction_from_cloth_offset_x;
                const bool prediction_x_valid =
                    prediction_layer_x >= 0.0f && prediction_layer_x < (float)width;
                const int iso_x = prediction_x_valid
                    ? std::clamp(
                        (int)std::floor((prediction_layer_x / (float)width) * (float)iso_w),
                        0,
                        iso_w - 1)
                    : 0;
                const size_t iso_pix = (size_t)iso_y * (size_t)iso_w + (size_t)iso_x;
                const bool iso_body_pixel =
                    prediction_x_valid
                    && prediction_y_valid
                    && iso_pix < iso_source.mask.size()
                    && iso_source.mask[iso_pix] > 0.5f;
                if (iso_body_pixel)
                    iso_body_mask[out_pix] = 1u;

                if (!prediction_x_valid
                    || !prediction_y_valid
                    || iso_pix >= iso_source.mask.size()
                    || !(iso_source.mask[iso_pix] > 0.5f))
                {
                    if (source_cloth_pixel)
                    {
                        if (prediction_x_valid && prediction_y_valid)
                        {
                            const int prediction_x = (int)std::floor(prediction_layer_x);
                            const int prediction_y = (int)std::floor(prediction_layer_y);
                            const size_t depth_idx =
                                (size_t)(height - 1 - prediction_y) * (size_t)width
                                + (size_t)prediction_x;
                            if (smpl_depth_gl[depth_idx] < 0.999999f)
                            {
                                unsigned char* dst = out_deformation_rgba.data() + out_pix * 4u;
                                dst[0] = 255u;
                                dst[1] = 64u;
                                dst[2] = 32u;
                                dst[3] = 220u;
                            }
                        }
                    }
                    continue;
                }

                const int prediction_x = (int)std::floor(prediction_layer_x);
                const int prediction_y = (int)std::floor(prediction_layer_y);
                const size_t prediction_depth_idx =
                    (size_t)(height - 1 - prediction_y) * (size_t)width
                    + (size_t)prediction_x;

                // Deformation-map colors:
                // orange = cloth + 3D body, but no sketch body;
                // blue = sketch body + cloth, but no 3D body at the sketch pixel;
                // magenta = sketch body + no source cloth, but a 3D cloth vertex is
                // close to the ISO-matched body vertex;
                // yellow = sketch body + source cloth, but the ISO-matched body
                // vertex is visible without cloth on top.
                if (source_cloth_pixel && !(smpl_depth_gl[prediction_depth_idx] < 0.999999f))
                {
                    unsigned char* dst = out_deformation_rgba.data() + out_pix * 4u;
                    dst[0] = 64u;
                    dst[1] = 128u;
                    dst[2] = 255u;
                    dst[3] = 220u;
                }

                const size_t descriptor_offset = iso_pix * (size_t)kIsoDimensions;
                if (descriptor_offset + (size_t)kIsoDimensions > iso_source.descriptors.size())
                    continue;

                const float* descriptor =
                    iso_source.descriptors.data() + descriptor_offset;
                int vertex_idx = -1;
                float iso_distance_squared = std::numeric_limits<float>::infinity();
                const bool descriptor_match_found = find_nearest_iso_vertex(
                    descriptor,
                    vertex_idx,
                    iso_distance_squared);
                if (descriptor_match_found
                    && mirror_iso_vertices
                    && vertex_idx >= 0
                    && vertex_idx < (int)m_iso_vertex_mirror_indices.size())
                {
                    vertex_idx = m_iso_vertex_mirror_indices[(size_t)vertex_idx];
                }
                bool descriptor_match_valid = descriptor_match_found
                    && vertex_idx >= 0
                    && vertex_idx < smpl_vertices_for_iso.rows()
                    && std::isfinite(iso_distance_squared)
                    && std::sqrt(std::max(0.0f, iso_distance_squared))
                        <= iso_distance_threshold;
                bool used_local_match = false;
                bool rejected_distant_match = false;

                float descriptor_screen_distance_squared =
                    std::numeric_limits<float>::infinity();
                if (descriptor_match_valid)
                {
                    const float match_dx = body_vertex_screen_x[(size_t)vertex_idx]
                        - prediction_layer_x;
                    const float match_dy = body_vertex_screen_y[(size_t)vertex_idx]
                        - prediction_layer_y;
                    descriptor_screen_distance_squared =
                        match_dx * match_dx + match_dy * match_dy;
                }
                const bool descriptor_jumps_across_image = descriptor_match_valid
                    && descriptor_screen_distance_squared
                        > hard_spatial_match_radius * hard_spatial_match_radius;

                int local_vertex_idx = -1;
                float local_screen_distance_squared =
                    std::numeric_limits<float>::infinity();
                if (have_projected_body_lookup)
                {
                    const size_t prediction_pixel =
                        (size_t)prediction_y * (size_t)width + (size_t)prediction_x;
                    const int seed_x = nearest_projected_body_x[prediction_pixel];
                    const int seed_y = nearest_projected_body_y[prediction_pixel];
                    if (seed_x >= 0 && seed_y >= 0)
                    {
                        const size_t seed_pixel =
                            (size_t)seed_y * (size_t)width + (size_t)seed_x;
                        local_vertex_idx = projected_body_seed_vertex[seed_pixel];
                        if (local_vertex_idx >= 0
                            && local_vertex_idx < smpl_vertices_for_iso.rows())
                        {
                            const float local_dx =
                                body_vertex_screen_x[(size_t)local_vertex_idx]
                                - prediction_layer_x;
                            const float local_dy =
                                body_vertex_screen_y[(size_t)local_vertex_idx]
                                - prediction_layer_y;
                            local_screen_distance_squared =
                                local_dx * local_dx + local_dy * local_dy;
                            if (local_screen_distance_squared
                                > local_lookup_radius * local_lookup_radius)
                                local_vertex_idx = -1;
                        }
                    }
                }

                if (local_vertex_idx >= 0)
                {
                    const float local_descriptor_distance_squared =
                        descriptor_distance_squared_to_vertex(descriptor, local_vertex_idx);
                    const bool local_descriptor_match_valid =
                        std::isfinite(local_descriptor_distance_squared)
                        && std::sqrt(std::max(0.0f, local_descriptor_distance_squared))
                            <= iso_distance_threshold;

                    const float descriptor_cost = iso_distance_squared
                        + spatial_cost_weight * descriptor_screen_distance_squared;
                    const float local_cost = local_descriptor_distance_squared
                        + spatial_cost_weight * local_screen_distance_squared;
                    if (!descriptor_match_valid
                        || (local_descriptor_match_valid
                            && (descriptor_jumps_across_image
                                || local_cost < descriptor_cost)))
                    {
                        vertex_idx = local_vertex_idx;
                        iso_distance_squared = local_descriptor_distance_squared;
                        descriptor_match_valid = true;
                        used_local_match = true;
                        rejected_distant_match = descriptor_jumps_across_image;
                    }
                }

                if (!descriptor_match_valid
                    || vertex_idx < 0
                    || vertex_idx >= smpl_vertices_for_iso.rows())
                    continue;

                if (source_cloth_pixel)
                {
                    if (used_local_match)
                        ++spatially_corrected_matches;
                    else
                        ++retained_descriptor_matches;
                    if (rejected_distant_match)
                        ++rejected_distant_matches;
                }

                const bool iso_matched_body_close_to_cloth =
                    vertex_idx < (int)body_cloth_vertex_distances.size()
                    && vertex_idx < (int)body_cloth_normal_scaled_thresholds.size()
                    && body_cloth_vertex_distances[(size_t)vertex_idx]
                        < body_cloth_normal_scaled_thresholds[(size_t)vertex_idx];
                if (iso_matched_body_close_to_cloth)
                    iso_body_close_to_cloth_mask[out_pix] = 1u;

                if (!source_cloth_pixel && iso_matched_body_close_to_cloth)
                {
                    unsigned char* dst = out_deformation_rgba.data() + out_pix * 4u;
                    dst[0] = 220u;
                    dst[1] = 64u;
                    dst[2] = 255u;
                    dst[3] = 220u;
                }

                const float projected_body_x = body_vertex_screen_x[(size_t)vertex_idx];
                const float projected_body_y = body_vertex_screen_y[(size_t)vertex_idx];
                const float projected_body_depth = body_vertex_screen_depth[(size_t)vertex_idx];
                if (!std::isfinite(projected_body_x)
                    || !std::isfinite(projected_body_y)
                    || !std::isfinite(projected_body_depth))
                    continue;

                // A spatial correction means the global descriptor pointed at a
                // different place in the image. Classify that pixel at its own
                // visible raster surface instead of at the discrete nearby
                // vertex. The latter is commonly a few millimetres behind the
                // triangle interpolation and used to create alternating yellow
                // bands over an otherwise continuous green torso.
                const float sample_x = used_local_match
                    ? prediction_layer_x
                    : projected_body_x;
                const float sample_y = used_local_match
                    ? prediction_layer_y
                    : projected_body_y;
                const int sx = (int)std::floor(sample_x);
                const int sy = (int)std::floor(sample_y);
                if (sx < 0 || sy < 0 || sx >= width || sy >= height)
                    continue;

                if (source_cloth_pixel)
                    ++out_tested_count;
                const size_t depth_idx = (size_t)(height - 1 - sy) * (size_t)width + (size_t)sx;
                const float cloth_depth = cloth_depth_gl[depth_idx];
                const bool cloth_present_at_body = cloth_depth < 0.999999f;
                const float visible_smpl_depth = smpl_depth_gl[depth_idx];
                const bool visible_smpl_present = visible_smpl_depth < 0.999999f;
                const float body_depth = used_local_match && visible_smpl_present
                    ? visible_smpl_depth
                    : projected_body_depth;
                const bool cloth_on_top_of_match =
                    cloth_present_at_body
                    && cloth_depth + depth_margin < body_depth;
                const float nearest_smpl_depth = visible_smpl_present
                    ? visible_smpl_depth
                    : body_depth;
                const bool other_smpl_layer_on_top = nearest_smpl_depth + depth_margin < body_depth;
                const float visibility_depth_tolerance = std::max(0.0001f, depth_margin * 4.0f);
                const bool matched_body_surface_visible = visible_smpl_present
                    && std::abs(projected_body_depth - visible_smpl_depth)
                        <= visibility_depth_tolerance;
                int dominant_body_joint = -1;
                if (vertex_idx >= 0 && vertex_idx < m_smplh_model.weights.rows())
                {
                    Eigen::Index dominant_joint_index = -1;
                    m_smplh_model.weights.row(vertex_idx).maxCoeff(&dominant_joint_index);
                    dominant_body_joint = (int)dominant_joint_index;
                }
                const bool matched_upper_body_occluder = dominant_body_joint >= 12;
                const bool body_in_front_of_cloth = matched_upper_body_occluder
                    && matched_body_surface_visible
                    && (!cloth_present_at_body
                        || visible_smpl_depth + depth_margin < cloth_depth);
                if (source_cloth_pixel && !cloth_present_at_body && !other_smpl_layer_on_top)
                {
                    unsigned char* dst = out_deformation_rgba.data() + out_pix * 4u;
                    dst[0] = 255u;
                    dst[1] = 224u;
                    dst[2] = 0u;
                    dst[3] = 220u;
                }
                if (!source_cloth_pixel)
                {
                    if (cloth_on_top_of_match)
                        set_debug_pixel(out_pix, 48u, 48u, 48u);
                    else
                        set_debug_pixel(out_pix, 0u, 0u, 0u);
                }
                else if (body_in_front_of_cloth)
                {
                    set_debug_pixel(out_pix, 255u, 144u, 0u);
                }
                else
                {
                    set_debug_pixel(out_pix, 32u, 220u, 80u);
                }
            }
        }

        console_add_line(
            state,
            "Cloth fitting ISO projection: "
                + std::string(mirror_iso_vertices ? "mirrored" : "direct")
                + " (samples=" + std::to_string(projection_sample_count)
                + ", direct cost=" + std::to_string(direct_projection_cost)
                + ", mirrored cost=" + std::to_string(mirrored_projection_cost)
                + ")");
        console_add_line(
            state,
            "Cloth fitting ISO correspondence: descriptor="
                + std::to_string(retained_descriptor_matches)
                + ", spatially corrected="
                + std::to_string(spatially_corrected_matches)
                + ", distant jumps rejected="
                + std::to_string(rejected_distant_matches));
        finish_stage("ISO matching");

        for (size_t pixel = 0; pixel < (size_t)width * (size_t)height; ++pixel)
        {
            if (cloth_rgba[pixel * 4u + 3u] == 0u)
            {
                out_debug_rgba[pixel * 4u + 0u] = 0u;
                out_debug_rgba[pixel * 4u + 1u] = 0u;
                out_debug_rgba[pixel * 4u + 2u] = 0u;
                out_debug_rgba[pixel * 4u + 3u] = 0u;
            }
        }
        out_unfiltered_debug_rgba = out_debug_rgba;
        filter_cloth_fitting_debug_colors(
            out_debug_rgba,
            width,
            height,
            state.cloth_fitting_debug_smoothing_radius,
            state.cloth_fitting_debug_smoothing_passes,
            state.cloth_fitting_debug_min_patch_area,
            state.cloth_fitting_debug_error_fill_radius);
        for (size_t pixel = 0; pixel < (size_t)width * (size_t)height; ++pixel)
        {
            if (cloth_rgba[pixel * 4u + 3u] == 0u)
                continue;

            const bool iso_occluded =
                rgba_is_color(out_debug_rgba, pixel, 255u, 32u, 32u)
                || rgba_is_color(out_debug_rgba, pixel, 255u, 144u, 0u);
            set_visibility_pixel(out_iso_occlusion_rgba, pixel, iso_occluded);
        }
        finish_stage("ISO matching filters");

        std::vector<unsigned char> true_depth_red(visibility_pixel_count, 0u);
        std::vector<unsigned char> iso_red(visibility_pixel_count, 0u);
        for (size_t pixel = 0; pixel < visibility_pixel_count; ++pixel)
        {
            if (cloth_rgba[pixel * 4u + 3u] == 0u)
                continue;
            true_depth_red[pixel] = rgba_is_color(
                out_scene_occlusion_rgba, pixel, 255u, 48u, 48u) ? 1u : 0u;
            iso_red[pixel] = rgba_is_color(
                out_iso_occlusion_rgba, pixel, 255u, 48u, 48u) ? 1u : 0u;
            set_visibility_pixel(out_corrected_scene_occlusion_rgba, pixel, false);
        }

        // ISO matching locates an occluder reliably but its boundary is noisy.
        // Translate each true-depth component toward the corresponding ISO
        // region for the corrected diagnostic layer while keeping the clean
        // rendered shape and area.
        struct VisibilityComponent
        {
            std::vector<size_t> pixels;
        };
        const auto visibility_components = [&](const std::vector<unsigned char>& mask)
        {
            std::vector<VisibilityComponent> components;
            std::vector<unsigned char> visited(visibility_pixel_count, 0u);
            std::vector<size_t> stack;
            for (size_t seed = 0; seed < visibility_pixel_count; ++seed)
            {
                if (mask[seed] == 0u || visited[seed] != 0u)
                    continue;

                VisibilityComponent component;
                stack.clear();
                stack.push_back(seed);
                visited[seed] = 1u;
                while (!stack.empty())
                {
                    const size_t pixel = stack.back();
                    stack.pop_back();
                    component.pixels.push_back(pixel);
                    const int x = (int)(pixel % (size_t)width);
                    const int y = (int)(pixel / (size_t)width);
                    for (int oy = -1; oy <= 1; ++oy)
                    {
                        const int ny = y + oy;
                        if (ny < 0 || ny >= height)
                            continue;
                        for (int ox = -1; ox <= 1; ++ox)
                        {
                            if (ox == 0 && oy == 0)
                                continue;
                            const int nx = x + ox;
                            if (nx < 0 || nx >= width)
                                continue;
                            const size_t neighbor =
                                (size_t)ny * (size_t)width + (size_t)nx;
                            if (mask[neighbor] == 0u || visited[neighbor] != 0u)
                                continue;
                            visited[neighbor] = 1u;
                            stack.push_back(neighbor);
                        }
                    }
                }
                components.push_back(std::move(component));
            }
            return components;
        };

        const int iso_support_radius = std::clamp(
            state.cloth_fitting_true_depth_correction_support_radius,
            0,
            64);
        std::vector<unsigned char> iso_support(visibility_pixel_count, 0u);
        std::vector<int> nearest_iso_x;
        std::vector<int> nearest_iso_y;
        if (nearest_mask_pixels(iso_red, width, height, nearest_iso_x, nearest_iso_y))
        {
            const int radius_squared = iso_support_radius * iso_support_radius;
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                    const int nx = nearest_iso_x[pixel];
                    const int ny = nearest_iso_y[pixel];
                    if (nx < 0 || ny < 0)
                        continue;
                    const int dx = nx - x;
                    const int dy = ny - y;
                    iso_support[pixel] =
                        dx * dx + dy * dy <= radius_squared ? 1u : 0u;
                }
            }
        }

        const std::vector<VisibilityComponent> source_components =
            visibility_components(true_depth_red);
        const int maximum_shift = std::clamp(
            state.cloth_fitting_true_depth_correction_max_shift,
            0,
            std::max(width, height));
        const float minimum_overlap = std::clamp(
            state.cloth_fitting_true_depth_correction_min_overlap,
            0.0f,
            1.0f);
        const float minimum_improvement = std::clamp(
            state.cloth_fitting_true_depth_correction_min_improvement,
            0.0f,
            1.0f);
        int shifted_component_count = 0;
        long long total_shift_x = 0;
        long long total_shift_y = 0;
        for (const VisibilityComponent& source : source_components)
        {
            const size_t sample_stride = std::max<size_t>(
                1u,
                (source.pixels.size() + 511u) / 512u);
            const auto alignment_score = [&](int shift_x, int shift_y, int* out_support_hits)
            {
                int exact_hits = 0;
                int support_hits = 0;
                int sample_count = 0;
                for (size_t i = 0; i < source.pixels.size(); i += sample_stride)
                {
                    const size_t pixel = source.pixels[i];
                    const int x = (int)(pixel % (size_t)width) + shift_x;
                    const int y = (int)(pixel / (size_t)width) + shift_y;
                    ++sample_count;
                    if (x < 0 || y < 0 || x >= width || y >= height)
                        continue;
                    const size_t shifted = (size_t)y * (size_t)width + (size_t)x;
                    if (cloth_rgba[shifted * 4u + 3u] == 0u)
                        continue;
                    exact_hits += iso_red[shifted] != 0u ? 1 : 0;
                    support_hits += iso_support[shifted] != 0u ? 1 : 0;
                }
                if (out_support_hits)
                    *out_support_hits = support_hits;
                if (sample_count == 0)
                    return -1.0f;
                return (2.0f * (float)exact_hits + (float)support_hits)
                    / (float)sample_count;
            };

            int best_shift_x = 0;
            int best_shift_y = 0;
            int baseline_support_hits = 0;
            int best_support_hits = 0;
            const float baseline_score = alignment_score(0, 0, &baseline_support_hits);
            float best_score = baseline_score;
            best_support_hits = baseline_support_hits;

            const auto consider_shift = [&](int shift_x, int shift_y)
            {
                int support_hits = 0;
                const float score = alignment_score(shift_x, shift_y, &support_hits);
                if (score > best_score)
                {
                    best_score = score;
                    best_shift_x = shift_x;
                    best_shift_y = shift_y;
                    best_support_hits = support_hits;
                }
            };

            constexpr int coarse_grid_radius = 24;
            if (maximum_shift > 0)
            {
                for (int grid_y = -coarse_grid_radius; grid_y <= coarse_grid_radius; ++grid_y)
                {
                    const int shift_y = (int)std::lround(
                        (float)grid_y * (float)maximum_shift / (float)coarse_grid_radius);
                    for (int grid_x = -coarse_grid_radius; grid_x <= coarse_grid_radius; ++grid_x)
                    {
                        const int shift_x = (int)std::lround(
                            (float)grid_x * (float)maximum_shift / (float)coarse_grid_radius);
                        consider_shift(shift_x, shift_y);
                    }
                }

                const int refinement_radius = std::max(
                    1,
                    (maximum_shift + coarse_grid_radius - 1) / coarse_grid_radius);
                const int coarse_best_x = best_shift_x;
                const int coarse_best_y = best_shift_y;
                for (int offset_y = -refinement_radius; offset_y <= refinement_radius; ++offset_y)
                {
                    for (int offset_x = -refinement_radius; offset_x <= refinement_radius; ++offset_x)
                    {
                        const int shift_x = coarse_best_x + offset_x;
                        const int shift_y = coarse_best_y + offset_y;
                        if (std::abs(shift_x) <= maximum_shift
                            && std::abs(shift_y) <= maximum_shift)
                            consider_shift(shift_x, shift_y);
                    }
                }
            }

            const int sampled_count = (int)((source.pixels.size() + sample_stride - 1u) / sample_stride);
            const float best_overlap = sampled_count > 0
                ? (float)best_support_hits / (float)sampled_count
                : 0.0f;
            if ((best_shift_x == 0 && best_shift_y == 0)
                || best_score < baseline_score + minimum_improvement
                || best_overlap < minimum_overlap)
            {
                best_shift_x = 0;
                best_shift_y = 0;
            }
            else
            {
                ++shifted_component_count;
                total_shift_x += best_shift_x;
                total_shift_y += best_shift_y;
            }

            for (size_t pixel : source.pixels)
            {
                const int x = (int)(pixel % (size_t)width) + best_shift_x;
                const int y = (int)(pixel / (size_t)width) + best_shift_y;
                if (x < 0 || y < 0 || x >= width || y >= height)
                    continue;
                const size_t shifted = (size_t)y * (size_t)width + (size_t)x;
                if (cloth_rgba[shifted * 4u + 3u] != 0u)
                    set_visibility_pixel(out_corrected_scene_occlusion_rgba, shifted, true);
            }
        }
        console_add_line(
            state,
            "Cloth fitting true-depth correction: shifted="
                + std::to_string(shifted_component_count)
                + "/" + std::to_string(source_components.size())
                + ", average offset=("
                + std::to_string(shifted_component_count > 0
                    ? (double)total_shift_x / (double)shifted_component_count
                    : 0.0)
                + ", "
                + std::to_string(shifted_component_count > 0
                    ? (double)total_shift_y / (double)shifted_component_count
                    : 0.0)
                + "), max shift=" + std::to_string(maximum_shift)
                + ", support radius=" + std::to_string(iso_support_radius)
                + ", min overlap/gain=" + std::to_string(minimum_overlap)
                + "/" + std::to_string(minimum_improvement));
        finish_stage("True-depth correction");

        const size_t pixel_count = (size_t)width * (size_t)height;
        const auto scene_debug_has_body = [](const std::vector<unsigned char>& rgba, size_t pixel)
        {
            const size_t i = pixel * 4u;
            if (i + 3u >= rgba.size() || rgba[i + 3u] == 0u)
                return false;

            // Scene debug encodes sorted depth hits. Only pure C and CC have no
            // body layer; coincident yellow and every B-containing signature do.
            const bool only_cloth =
                (rgba[i + 0u] == 0u && rgba[i + 1u] == 188u && rgba[i + 2u] == 212u)
                || (rgba[i + 0u] == 224u && rgba[i + 1u] == 64u && rgba[i + 2u] == 251u);
            return !only_cloth;
        };
        const auto scene_debug_has_cloth = [](const std::vector<unsigned char>& rgba, size_t pixel)
        {
            const size_t i = pixel * 4u;
            if (i + 3u >= rgba.size() || rgba[i + 3u] == 0u)
                return false;

            // Only B and BB have no cloth layer in the retained ray stack.
            const bool only_body =
                (rgba[i + 0u] == 96u && rgba[i + 1u] == 125u && rgba[i + 2u] == 139u)
                || (rgba[i + 0u] == 255u && rgba[i + 1u] == 152u && rgba[i + 2u] == 0u);
            return !only_body;
        };
        const auto rebuild_deformation_from_warped_layers = [&](
            const std::vector<unsigned char>& cloth,
            const std::vector<unsigned char>& scene_debug,
            std::vector<unsigned char>& deformation)
        {
            deformation.assign(pixel_count * 4u, 0u);
            for (size_t pixel = 0; pixel < pixel_count; ++pixel)
            {
                const bool source_cloth_pixel =
                    pixel * 4u + 3u < cloth.size() && cloth[pixel * 4u + 3u] != 0u;
                const bool sketch_body_pixel =
                    pixel < iso_body_mask.size() && iso_body_mask[pixel] != 0u;
                const bool scene_body_pixel = scene_debug_has_body(scene_debug, pixel);
                const bool iso_body_close_to_cloth =
                    pixel < iso_body_close_to_cloth_mask.size()
                    && iso_body_close_to_cloth_mask[pixel] != 0u;
                const bool scene_cloth_pixel = scene_debug_has_cloth(scene_debug, pixel);

                unsigned char* dst = deformation.data() + pixel * 4u;
                if (source_cloth_pixel && !sketch_body_pixel && scene_body_pixel)
                {
                    dst[0] = 255u;
                    dst[1] = 64u;
                    dst[2] = 32u;
                    dst[3] = 220u;
                }
                else if (source_cloth_pixel && sketch_body_pixel && scene_body_pixel && !scene_cloth_pixel)
                {
                    dst[0] = 255u;
                    dst[1] = 224u;
                    dst[2] = 0u;
                    dst[3] = 220u;
                }
                else if (source_cloth_pixel && sketch_body_pixel && !scene_body_pixel)
                {
                    dst[0] = 64u;
                    dst[1] = 128u;
                    dst[2] = 255u;
                    dst[3] = 220u;
                }
                else if (!source_cloth_pixel && sketch_body_pixel && iso_body_close_to_cloth)
                {
                    dst[0] = 220u;
                    dst[1] = 64u;
                    dst[2] = 255u;
                    dst[3] = 220u;
                }
            }
        };
        const auto count_final_deformation_colors = [](
            const std::vector<unsigned char>& deformation,
            int& red,
            int& blue,
            int& purple)
        {
            red = 0;
            blue = 0;
            purple = 0;
            const size_t count = deformation.size() / 4u;
            for (size_t pixel = 0; pixel < count; ++pixel)
            {
                const size_t p = pixel * 4u;
                if (deformation[p + 3u] == 0u)
                    continue;
                if (deformation[p + 0u] >= 240u
                    && deformation[p + 1u] <= 96u
                    && deformation[p + 2u] <= 80u)
                    ++red;
                else if (deformation[p + 0u] <= 96u
                    && deformation[p + 1u] >= 96u
                    && deformation[p + 2u] >= 220u)
                    ++blue;
                else if (deformation[p + 0u] >= 180u
                    && deformation[p + 1u] <= 96u
                    && deformation[p + 2u] >= 200u)
                    ++purple;
            }
        };

        struct OcclusionResult
        {
            std::vector<unsigned char> visible_cloth;
            std::vector<unsigned char> unfiltered_layer;
            std::vector<unsigned char> filtered_layer;
            int removed_count = 0;
        };

        auto build_occlusion = [&](const std::vector<unsigned char>& cloth,
                                   const std::vector<unsigned char>& debug,
                                   const std::vector<unsigned char>& scene_debug,
                                   const std::vector<unsigned char>& scene_occlusion,
                                   const std::vector<unsigned char>& aligned_predicted_depth_occlusion)
        {
            const size_t count = (size_t)width * (size_t)height;
            OcclusionResult result;
            result.visible_cloth = cloth;
            result.unfiltered_layer.assign(count * 4u, 0u);
            result.filtered_layer.assign(count * 4u, 0u);
            std::vector<unsigned char> active(count, 0u);
            std::vector<unsigned char> keep_flags(count, 0u);

            auto write_occlusion_pixel = [](std::vector<unsigned char>& rgba, size_t pixel, bool keep)
            {
                unsigned char* occlusion_pixel = rgba.data() + pixel * 4u;
                occlusion_pixel[0] = keep ? 32u : 255u;
                occlusion_pixel[1] = keep ? 220u : 48u;
                occlusion_pixel[2] = keep ? 80u : 48u;
                occlusion_pixel[3] = 255u;
            };

            for (size_t pixel = 0; pixel < count; ++pixel)
            {
                if (cloth[pixel * 4u + 3u] == 0u)
                    continue;
                active[pixel] = 1u;

                const bool depth_occluded =
                    rgba_is_color(scene_occlusion, pixel, 255u, 48u, 48u);
                bool keep = !depth_occluded;
                const bool iso_green = rgba_is_color(debug, pixel, 32u, 220u, 80u);
                const bool iso_orange = rgba_is_color(debug, pixel, 255u, 144u, 0u);
                const bool scene_dark_purple = rgba_is_color(scene_debug, pixel, 170u, 0u, 255u);
                const bool scene_deep_orange = rgba_is_color(scene_debug, pixel, 255u, 109u, 0u);
                const bool aligned_predicted_depth_occluded =
                    rgba_is_color(aligned_predicted_depth_occlusion, pixel, 255u, 48u, 48u);
                if (aligned_predicted_depth_occluded)
                    keep = false;
                else if (iso_green && scene_dark_purple)
                    keep = true;
                else if (iso_green && scene_deep_orange)
                    keep = false;
                else if (iso_orange)
                    keep = false;

                keep_flags[pixel] = keep ? 1u : 0u;
                write_occlusion_pixel(result.unfiltered_layer, pixel, keep);
            }

            const size_t min_patch_area = (size_t)std::clamp(
                state.cloth_fitting_occlusion_min_patch_area,
                1,
                4096);
            filter_small_occlusion_components(active, width, height, keep_flags, min_patch_area);
            std::vector<unsigned char> chaotic_regions;
            simplify_chaotic_occlusion_regions(
                active,
                width,
                height,
                state.cloth_fitting_occlusion_cleanup_radius,
                state.cloth_fitting_occlusion_cleanup_passes,
                state.cloth_fitting_occlusion_chaos_threshold,
                keep_flags,
                chaotic_regions);
            diagonalize_occlusion_edges(active, chaotic_regions, width, height, keep_flags);
            filter_small_occlusion_components(active, width, height, keep_flags, min_patch_area);

            for (size_t pixel = 0; pixel < count; ++pixel)
            {
                if (!active[pixel])
                    continue;

                const bool keep = keep_flags[pixel] != 0u;
                write_occlusion_pixel(result.filtered_layer, pixel, keep);
                if (!keep)
                {
                    result.visible_cloth[pixel * 4u + 3u] = 0u;
                    ++result.removed_count;
                }
            }
            return result;
        };

        OcclusionResult initial_occlusion = build_occlusion(
            cloth_rgba,
            out_debug_rgba,
            out_scene_debug_rgba,
            out_scene_occlusion_rgba,
            out_aligned_predicted_depth_occlusion_rgba);
        out_rgba = std::move(initial_occlusion.visible_cloth);
        out_unfiltered_occlusion_rgba = std::move(initial_occlusion.unfiltered_layer);
        out_occlusion_rgba = std::move(initial_occlusion.filtered_layer);
        out_removed_count = initial_occlusion.removed_count;
        rebuild_deformation_from_warped_layers(
            cloth_rgba,
            out_scene_debug_rgba,
            out_deformation_rgba);
        finish_stage("Visibility fusion and filters");

        // Deformation is computed from the raw cloth silhouette only. The same
        // accumulated field is applied independently to raw and occluded cloth;
        // visibility maps always remain in the original fitting coordinates.
        std::vector<unsigned char> current_unoccluded_cloth = cloth_rgba;
        std::vector<unsigned char> current_occluded_cloth = out_rgba;
        std::vector<unsigned char> current_deformation = out_deformation_rgba;
        std::vector<float> accumulated_source_x;
        std::vector<float> accumulated_source_y;
        out_deformation_steps_rgba.push_back(current_deformation);

        int total_red_seeds = 0;
        int total_purple_seeds = 0;
        int applied_warp_steps = 0;
        const int warp_steps = std::clamp(state.cloth_fitting_warp_steps, 1, 32);
        const float warp_radius = std::clamp(state.cloth_fitting_warp_radius, 0.0f, 512.0f);
        const float warp_force = std::clamp(state.cloth_fitting_warp_force, 0.0f, 6.0f);
        const float warp_force_per_step = warp_force / (float)std::max(1, warp_steps);
        for (int step = 0; step < warp_steps; ++step)
        {
            const auto warp_step_start = std::chrono::steady_clock::now();
            int step_red = 0;
            int step_blue = 0;
            int step_purple = 0;
            count_final_deformation_colors(current_deformation, step_red, step_blue, step_purple);
            if (step_red == 0 && step_blue == 0 && step_purple == 0)
                break;

            std::vector<unsigned char> source_cloth_mask(pixel_count, 0u);
            for (size_t pixel = 0; pixel < source_cloth_mask.size(); ++pixel)
                source_cloth_mask[pixel] = current_unoccluded_cloth[pixel * 4u + 3u] != 0u ? 1u : 0u;

            std::vector<float> warp_dx;
            std::vector<float> warp_dy;
            const DeformationWarpSeedCounts warp_seed_counts = build_deformation_warp(
                current_deformation,
                iso_body_mask,
                source_cloth_mask,
                width,
                height,
                warp_radius,
                warp_force_per_step,
                warp_dx,
                warp_dy);
            total_red_seeds += warp_seed_counts.red;
            total_purple_seeds += warp_seed_counts.purple;
            if (warp_seed_counts.total() == 0)
            {
                // Blue is diagnostic-only and does not currently produce a
                // warp field. Keep the final blue-only residual map, then stop.
                break;
            }

            std::vector<float> inverse_source_x;
            std::vector<float> inverse_source_y;
            build_inverse_warp_map(
                width,
                height,
                warp_dx,
                warp_dy,
                inverse_source_x,
                inverse_source_y);

            std::vector<unsigned char> next_unoccluded_cloth;
            std::vector<unsigned char> next_occluded_cloth;
            warp_rgba_inverse_sharp(
                current_unoccluded_cloth,
                width,
                height,
                inverse_source_x,
                inverse_source_y,
                next_unoccluded_cloth);
            warp_rgba_inverse_sharp(
                current_occluded_cloth,
                width,
                height,
                inverse_source_x,
                inverse_source_y,
                next_occluded_cloth);

            if (accumulated_source_x.empty())
            {
                accumulated_source_x = std::move(inverse_source_x);
                accumulated_source_y = std::move(inverse_source_y);
            }
            else
            {
                warp_dx.resize(pixel_count);
                warp_dy.resize(pixel_count);
                for (size_t pixel = 0; pixel < pixel_count; ++pixel)
                {
                    const float source_x = inverse_source_x[pixel];
                    const float source_y = inverse_source_y[pixel];
                    if (source_x < 0.0f || source_y < 0.0f
                        || source_x > (float)(width - 1) || source_y > (float)(height - 1))
                    {
                        warp_dx[pixel] = -1.0f;
                        warp_dy[pixel] = -1.0f;
                        continue;
                    }
                    warp_dx[pixel] = sample_bilinear_scalar(
                        accumulated_source_x, width, height, source_x, source_y);
                    warp_dy[pixel] = sample_bilinear_scalar(
                        accumulated_source_y, width, height, source_x, source_y);
                }
                accumulated_source_x = std::move(warp_dx);
                accumulated_source_y = std::move(warp_dy);
            }

            current_unoccluded_cloth = std::move(next_unoccluded_cloth);
            current_occluded_cloth = std::move(next_occluded_cloth);
            rebuild_deformation_from_warped_layers(
                current_unoccluded_cloth,
                out_scene_debug_rgba,
                current_deformation);
            out_deformation_steps_rgba.push_back(current_deformation);
            ++applied_warp_steps;
            out_layers.timings_seconds.emplace_back(
                "Warp step " + std::to_string(step + 1),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - warp_step_start).count());
        }

        out_deformation_rgba = current_deformation;
        out_layers.warped_cloth = std::move(current_unoccluded_cloth);
        out_warped_rgba = std::move(current_occluded_cloth);
        if (accumulated_source_x.empty())
        {
            accumulated_source_x.resize(pixel_count);
            accumulated_source_y.resize(pixel_count);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                    accumulated_source_x[pixel] = (float)x;
                    accumulated_source_y[pixel] = (float)y;
                }
            }
        }
        out_layers.deformation_source_x = std::move(accumulated_source_x);
        out_layers.deformation_source_y = std::move(accumulated_source_y);

        int final_red = 0;
        int final_blue = 0;
        int final_purple = 0;
        count_final_deformation_colors(out_deformation_rgba, final_red, final_blue, final_purple);
        console_add_line(
            state,
            "Cloth fitting deformation warp: steps=" + std::to_string(applied_warp_steps)
                + "/" + std::to_string(warp_steps)
                + ", seed_red=" + std::to_string(total_red_seeds)
                + ", seed_purple=" + std::to_string(total_purple_seeds)
                + ", residual(red/blue/purple)=" + std::to_string(final_red)
                + "/" + std::to_string(final_blue)
                + "/" + std::to_string(final_purple)
                + ", radius=" + std::to_string(warp_radius)
                + ", force(total/step)=" + std::to_string(warp_force)
                + "/" + std::to_string(warp_force_per_step)
                + ", removed=" + std::to_string(out_removed_count));
        finish_stage("Warping");

        const std::array<const std::vector<unsigned char>*, 8> raw_mask_layers = {{
            &out_unfiltered_debug_rgba,
            &out_debug_rgba,
            &out_aligned_predicted_depth_occlusion_rgba,
            &out_scene_occlusion_rgba,
            &out_corrected_scene_occlusion_rgba,
            &out_iso_occlusion_rgba,
            &out_unfiltered_occlusion_rgba,
            &out_occlusion_rgba,
        }};
        for (const auto* layer : raw_mask_layers)
        {
            if (layer->size() < pixel_count * 4u)
                return fail("cloth fitting produced an incomplete visibility layer");
            for (size_t pixel = 0; pixel < pixel_count; ++pixel)
            {
                const bool raw_active = cloth_rgba[pixel * 4u + 3u] != 0u;
                const bool layer_active = (*layer)[pixel * 4u + 3u] != 0u;
                if (raw_active != layer_active)
                    return fail("cloth fitting visibility layers do not share the raw cloth mask");
            }
        }

        const std::filesystem::path renders_dir = state.cloth_fitting_output_dir.empty()
            ? std::filesystem::path("renders")
            : std::filesystem::path(state.cloth_fitting_output_dir);
        std::error_code ec;
        std::filesystem::create_directories(renders_dir, ec);
        if (ec)
            return fail("could not create renders directory: " + ec.message());

        constexpr std::array<const char*, 18> obsolete_outputs = {{
            "cloth_fitting_3d_scene_debug.png",
            "cloth_fitting_aligned_predicted_depth_visibility.png",
            "cloth_fitting_close_to_body.png",
            "cloth_fitting_debug_no_warp.png",
            "cloth_fitting_debug_no_warp_unfiltered.png",
            "cloth_fitting_debug_warped.png",
            "cloth_fitting_deformation.png",
            "cloth_fitting_iso_mapping_colors.png",
            "cloth_fitting_no_warp.png",
            "cloth_fitting_occlusion_filtered.png",
            "cloth_fitting_occlusion_unfiltered.png",
            "cloth_fitting_occlusion_warped.png",
            "cloth_fitting_occlusion_warped_unfiltered.png",
            "cloth_fitting_pixelated_occluded.png",
            "cloth_fitting_unwarped_occlusion_smoothed.png",
            "cloth_fitting_warped.png",
            "cloth_fitting_final_occlusion.png",
            "cloth_fitting_final_occlusion_unfiltered.png",
        }};
        for (const char* filename : obsolete_outputs)
        {
            ec.clear();
            std::filesystem::remove(renders_dir / filename, ec);
            if (ec)
                return fail("could not remove obsolete fitting output " + std::string(filename));
        }
        for (const auto& entry : std::filesystem::directory_iterator(renders_dir, ec))
        {
            if (ec)
                break;
            const std::string filename = entry.path().filename().string();
            if (filename.rfind("cloth_fitting_deformation_step_", 0) == 0
                && entry.path().extension() == ".png")
            {
                std::error_code remove_error;
                std::filesystem::remove(entry.path(), remove_error);
                if (remove_error)
                    return fail("could not replace " + entry.path().string());
            }
        }
        if (ec)
            return fail("could not scan fitting output directory: " + ec.message());

        struct LayerImageOutput
        {
            const char* filename = nullptr;
            const std::vector<unsigned char>* pixels = nullptr;
        };
        const std::array<LayerImageOutput, 13> layer_outputs = {{
            { "cloth_fitting_raw_cloth.png", &cloth_rgba },
            { "cloth_fitting_dumb_render.png", &out_dumb_render_rgba },
            { "cloth_fitting_raw_iso_matching.png", &out_unfiltered_debug_rgba },
            { "cloth_fitting_filtered_iso_matching.png", &out_debug_rgba },
            { "cloth_fitting_predicted_depth_visibility.png", &out_aligned_predicted_depth_occlusion_rgba },
            { "cloth_fitting_true_depth_visibility.png", &out_scene_occlusion_rgba },
            { "cloth_fitting_true_depth_corrected_visibility.png", &out_corrected_scene_occlusion_rgba },
            { "cloth_fitting_iso_visibility.png", &out_iso_occlusion_rgba },
            { "cloth_fitting_raw_final_occlusion.png", &out_unfiltered_occlusion_rgba },
            { "cloth_fitting_filtered_final_occlusion.png", &out_occlusion_rgba },
            { "cloth_fitting_occluded_cloth.png", &out_rgba },
            { "cloth_fitting_warped_cloth.png", &out_layers.warped_cloth },
            { "cloth_fitting_warped_occluded_cloth.png", &out_warped_rgba },
        }};
        for (const LayerImageOutput& output : layer_outputs)
        {
            const std::filesystem::path path = renders_dir / output.filename;
            if (stbi_write_png(
                    path.string().c_str(),
                    width,
                    height,
                    4,
                    output.pixels->data(),
                    width * 4) == 0)
                return fail("could not write " + path.string());
        }
        for (int i = 0; i < (int)out_deformation_steps_rgba.size(); ++i)
        {
            const std::filesystem::path path =
                renders_dir / ("cloth_fitting_deformation_step_" + std::to_string(i) + ".png");
            const std::vector<unsigned char>& pixels = out_deformation_steps_rgba[(size_t)i];
            if (stbi_write_png(
                    path.string().c_str(),
                    width,
                    height,
                    4,
                    pixels.data(),
                    width * 4) == 0)
                return fail("could not write " + path.string());
        }
        console_add_line(state, "Cloth fitting layers saved to " + renders_dir.string());
        finish_stage("Output validation and saving");

        return true;
    }

    bool Renderer::refit_cloth_render_with_maps(
        const std::vector<unsigned char>& cloth_rgba,
        int width,
        int height,
        const std::vector<unsigned char>& visibility_rgba,
        const std::vector<float>& deformation_source_x,
        const std::vector<float>& deformation_source_y,
        std::vector<unsigned char>& out_warped_cloth,
        std::vector<unsigned char>& out_warped_occluded_cloth,
        std::string* out_error) const
    {
        const size_t pixel_count = (size_t)width * (size_t)height;
        if (width <= 0 || height <= 0
            || cloth_rgba.size() < pixel_count * 4u
            || visibility_rgba.size() < pixel_count * 4u
            || deformation_source_x.size() < pixel_count
            || deformation_source_y.size() < pixel_count)
        {
            if (out_error)
                *out_error = "saved fitting maps do not match the cloth render size";
            return false;
        }

        std::vector<unsigned char> occluded_cloth = cloth_rgba;
        for (size_t pixel = 0; pixel < pixel_count; ++pixel)
        {
            const size_t offset = pixel * 4u;
            const bool hidden = visibility_rgba[offset + 3u] != 0u
                && visibility_rgba[offset + 0u] >= 200u
                && visibility_rgba[offset + 1u] <= 100u
                && visibility_rgba[offset + 2u] <= 100u;
            if (hidden)
                occluded_cloth[offset + 3u] = 0u;
        }

        warp_rgba_inverse_sharp(
            cloth_rgba,
            width,
            height,
            deformation_source_x,
            deformation_source_y,
            out_warped_cloth);
        warp_rgba_inverse_sharp(
            occluded_cloth,
            width,
            height,
            deformation_source_x,
            deformation_source_y,
            out_warped_occluded_cloth);
        return true;
    }
}
