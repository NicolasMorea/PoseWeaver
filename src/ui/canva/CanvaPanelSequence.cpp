#include "ui/canva/CanvaPanelInternal.h"

#include "core/Paths.h"
#include "render/Renderer.h"
#include "smpl/Smplh.h"
#include "ui/canva/CanvaPreprocess.h"
#include "ui/canva/SketchPose.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <imgui.h>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <stb_image.h>
#include <stb_image_write.h>

namespace poseweaver::ui_canva
{
    namespace
    {
        static CanvaLayer sequence_copy_layer(const CanvaLayer& source, int id, const std::string& name)
        {
            CanvaLayer layer = source;
            layer.id = id;
            layer.name = name;
            layer.texture_id = 0;
            layer.dirty = true;
            layer.bounds_dirty = true;
            return layer;
        }

        static CanvaLayer sequence_make_layer_from_pixels(
            CanvaState& canva,
            std::vector<unsigned char> pixels,
            const std::string& name,
            CanvaLayerKind kind,
            bool enabled)
        {
            CanvaLayer layer = canva_make_layer(canva.width, canva.height, canva.next_layer_id++, name);
            layer.kind = kind;
            layer.enabled = enabled;
            layer.pixels = std::move(pixels);
            layer.texture_id = 0;
            layer.dirty = true;
            layer.bounds_dirty = true;
            return layer;
        }

        static std::vector<unsigned char> sequence_resample_bilinear_rgba(
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

        static bool sequence_load_image_as_canvas_pixels(
            const std::filesystem::path& path,
            int canvas_w,
            int canvas_h,
            std::vector<unsigned char>* out_pixels,
            std::string* out_error)
        {
            if (!out_pixels)
                return false;
            out_pixels->clear();

            int iw = 0, ih = 0, comp = 0;
            stbi_uc* data = stbi_load(path.string().c_str(), &iw, &ih, &comp, 4);
            if (!data)
            {
                if (out_error)
                    *out_error = "failed to load extracted frame: " + path.string();
                return false;
            }

            const float s = std::min(
                1.0f,
                std::min(
                    (float)canvas_w / (float)std::max(1, iw),
                    (float)canvas_h / (float)std::max(1, ih)));
            const int dw = std::clamp((int)std::floor((float)iw * s + 0.5f), 1, canvas_w);
            const int dh = std::clamp((int)std::floor((float)ih * s + 0.5f), 1, canvas_h);
            std::vector<unsigned char> resized =
                sequence_resample_bilinear_rgba((const unsigned char*)data, iw, ih, dw, dh);
            stbi_image_free(data);

            out_pixels->assign((size_t)canvas_w * (size_t)canvas_h * 4u, 255u);
            const int off_x = std::max(0, (canvas_w - dw) / 2);
            const int off_y = std::max(0, (canvas_h - dh) / 2);
            for (int y = 0; y < dh; ++y)
            {
                for (int x = 0; x < dw; ++x)
                {
                    const size_t src = ((size_t)y * (size_t)dw + (size_t)x) * 4u;
                    const size_t dst = ((size_t)(y + off_y) * (size_t)canvas_w + (size_t)(x + off_x)) * 4u;
                    (*out_pixels)[dst + 0u] = resized[src + 0u];
                    (*out_pixels)[dst + 1u] = resized[src + 1u];
                    (*out_pixels)[dst + 2u] = resized[src + 2u];
                    (*out_pixels)[dst + 3u] = resized[src + 3u];
                }
            }
            return true;
        }

        static bool sequence_pixels_identical_for_merge(
            const std::vector<unsigned char>& a,
            const std::vector<unsigned char>& b)
        {
            if (a.size() != b.size())
                return false;
            if (a.empty())
                return true;

            unsigned long long abs_sum = 0;
            unsigned long long sq_sum = 0;
            int changed_samples = 0;
            int samples = 0;

            // Compare on a small grid so static video compression noise merges,
            // while actual sketch edits split into a new sequence image.
            constexpr int kStridePixels = 4;
            constexpr int kChannels = 4;
            const size_t pixel_count = a.size() / kChannels;
            for (size_t pixel = 0; pixel < pixel_count; pixel += kStridePixels)
            {
                const size_t i = pixel * kChannels;
                const int dr = std::abs((int)a[i + 0u] - (int)b[i + 0u]);
                const int dg = std::abs((int)a[i + 1u] - (int)b[i + 1u]);
                const int db = std::abs((int)a[i + 2u] - (int)b[i + 2u]);
                const int d = std::max(dr, std::max(dg, db));
                abs_sum += (unsigned long long)d;
                sq_sum += (unsigned long long)d * (unsigned long long)d;
                if (d > 18)
                    ++changed_samples;
                ++samples;
            }

            if (samples <= 0)
                return true;

            const double mean_abs = (double)abs_sum / (double)samples;
            const double rms = std::sqrt((double)sq_sum / (double)samples);
            const double changed_ratio = (double)changed_samples / (double)samples;
            return mean_abs <= 1.25
                || (mean_abs <= 3.0 && rms <= 7.0 && changed_ratio <= 0.0035);
        }

        static bool sequence_put_sketch_frame_on_canvas(
            CanvaState& canva,
            int sketch_index,
            const std::string& name)
        {
            if (sketch_index < 0 || sketch_index >= (int)canva.sequence_sketch_frames.size())
                return false;

            const CanvaLayer& source = canva.sequence_sketch_frames[(size_t)sketch_index].layer;
            if (source.pixels.empty())
                return false;

            int live_idx = canva.sequence_live_sketch_layer_id != 0
                ? canva_find_layer_index_by_id(canva, canva.sequence_live_sketch_layer_id)
                : -1;
            if (live_idx >= 0)
            {
                CanvaLayer& layer = canva.layers[(size_t)live_idx];
                const GLuint texture_id = layer.texture_id;
                layer = source;
                layer.id = canva.sequence_live_sketch_layer_id;
                layer.name = name;
                layer.kind = CanvaLayerKind::Generic;
                layer.enabled = true;
                layer.texture_id = texture_id;
                layer.dirty = true;
                layer.bounds_dirty = true;
                canva.active_layer = live_idx;
            }
            else
            {
                CanvaLayer layer = sequence_copy_layer(
                    source,
                    canva.next_layer_id++,
                    name);
                layer.kind = CanvaLayerKind::Generic;
                layer.enabled = true;
                canva.sequence_live_sketch_layer_id = layer.id;
                canva.layers.insert(canva.layers.begin(), std::move(layer));
                canva.active_layer = 0;
            }

            canva.sequence_preview_active = false;
            canva.sequence_preview_frame_id = 0;
            canva_cancel_interactions(canva);
            return true;
        }

        static int sequence_video_target_cloth_frames(const CanvaState& canva)
        {
            if (!canva.sequence_has_imported_video || canva.sequence_sketch_frames.empty())
                return 0;
            return std::max(
                1,
                (int)canva.sequence_sketch_frames.size()
                    * std::max(1, canva.sequence_sub_frames));
        }

        static int sequence_sim_frame_at_cloth_index(
            const CanvaState& canva,
            int cloth_index,
            int simulation_frame_count)
        {
            if (simulation_frame_count <= 0)
                return -1;
            if (canva.sequence_cloth_frames.empty())
                return std::clamp(cloth_index, 0, simulation_frame_count - 1);

            const int clamped_index = std::clamp(
                cloth_index,
                0,
                (int)canva.sequence_cloth_frames.size() - 1);
            const int mapped = canva.sequence_cloth_frames[(size_t)clamped_index].sim_frame_index;
            if (mapped >= 0)
                return std::clamp(mapped, 0, simulation_frame_count - 1);
            return std::clamp(clamped_index, 0, simulation_frame_count - 1);
        }

        static int sequence_sim_frame_at_time(
            const CanvaState& canva,
            float time_seconds,
            int simulation_frame_count)
        {
            if (simulation_frame_count <= 0 || canva.sequence_cloth_frames.empty())
                return -1;
            const int cloth_index = (int)std::llround(
                (double)std::max(0.0f, time_seconds)
                * (double)std::max(1, canva.sequence_frame_rate));
            return sequence_sim_frame_at_cloth_index(canva, cloth_index, simulation_frame_count);
        }

        static float sequence_time_for_sim_frame(const CanvaState& canva, int sim_frame)
        {
            if (canva.sequence_cloth_frames.empty())
                return 0.0f;

            int best_index = 0;
            int best_distance = std::numeric_limits<int>::max();
            for (int i = 0; i < (int)canva.sequence_cloth_frames.size(); ++i)
            {
                int mapped = canva.sequence_cloth_frames[(size_t)i].sim_frame_index;
                if (mapped < 0)
                    mapped = i;
                const int distance = std::abs(mapped - sim_frame);
                if (distance < best_distance)
                {
                    best_distance = distance;
                    best_index = i;
                }
            }

            return (float)best_index / (float)std::max(1, canva.sequence_frame_rate);
        }

        static void sequence_update_rig_overlay_for_sim_frame(AppState& state, const CanvaState& canva)
        {
            state.scene_view_sequence_rig_overlay_active = false;
            state.scene_view_sequence_rig_overlay_visible = false;
            state.scene_view_sequence_rig_overlay_path.clear();

            if (canva.sequence_cloth_frames.empty()
                || canva.sequence_sketch_frames.empty()
                || state.controls.frame_count <= 0)
                return;

            const int sim_frame = std::clamp(
                state.controls.current_frame,
                0,
                std::max(0, state.controls.frame_count - 1));

            int best_index = -1;
            int best_distance = std::numeric_limits<int>::max();
            for (int i = 0; i < (int)canva.sequence_cloth_frames.size(); ++i)
            {
                const CanvaSequenceTrackFrame& frame = canva.sequence_cloth_frames[(size_t)i];
                int mapped = frame.sim_frame_index;
                if (mapped < 0)
                    mapped = i;
                const int distance = std::abs(mapped - sim_frame);
                if (distance < best_distance)
                {
                    best_distance = distance;
                    best_index = i;
                }
            }

            if (best_index < 0)
                return;

            const CanvaSequenceTrackFrame& frame = canva.sequence_cloth_frames[(size_t)best_index];
            const int sketch_index = frame.sketch_frame_index;
            if (sketch_index < 0 || sketch_index >= (int)canva.sequence_sketch_frames.size())
                return;

            const std::filesystem::path& rig_path = canva.sequence_sketch_frames[(size_t)sketch_index].rig_json_path;
            if (rig_path.empty())
                return;

            int mapped = frame.sim_frame_index;
            if (mapped < 0)
                mapped = best_index;
            const int min_visible_radius = 2;
            int visible_min = mapped - min_visible_radius;
            int visible_max = mapped + min_visible_radius;

            if (best_index > 0)
            {
                const CanvaSequenceTrackFrame& prev = canva.sequence_cloth_frames[(size_t)best_index - 1u];
                int prev_mapped = prev.sim_frame_index;
                if (prev_mapped < 0)
                    prev_mapped = best_index - 1;
                if (prev.sketch_frame_index == sketch_index)
                    visible_min = std::min(visible_min, (prev_mapped + mapped) / 2);
            }
            else
            {
                visible_min = 0;
            }

            if (best_index + 1 < (int)canva.sequence_cloth_frames.size())
            {
                const CanvaSequenceTrackFrame& next = canva.sequence_cloth_frames[(size_t)best_index + 1u];
                int next_mapped = next.sim_frame_index;
                if (next_mapped < 0)
                    next_mapped = best_index + 1;
                if (next.sketch_frame_index == sketch_index)
                    visible_max = std::max(visible_max, (mapped + next_mapped + 1) / 2);
            }
            else
            {
                visible_max = std::max(visible_max, state.controls.frame_count - 1);
            }

            visible_min = std::clamp(visible_min, 0, std::max(0, state.controls.frame_count - 1));
            visible_max = std::clamp(visible_max, 0, std::max(0, state.controls.frame_count - 1));

            state.scene_view_sequence_rig_overlay_active = true;
            state.scene_view_sequence_rig_overlay_visible =
                state.show_sketch2rig_rig
                && sim_frame >= visible_min
                && sim_frame <= visible_max;
            state.scene_view_sequence_rig_overlay_path = rig_path.string();
        }

        static const CanvaLayer* sequence_find_latest_warped_layer(const CanvaState& canva)
        {
            for (const CanvaLayer& layer : canva.layers)
            {
                if (layer.kind == CanvaLayerKind::ClothFittingWarped)
                    return &layer;
            }
            return nullptr;
        }

        static void sequence_store_cached_fitting_inputs(CanvaState& canva)
        {
            if (!canva.cloth_fitting_model_input_path.empty())
                canva.sequence_model_input_path = canva.cloth_fitting_model_input_path;
            if (!canva.cloth_fitting_iso_path.empty())
                canva.sequence_iso_path = canva.cloth_fitting_iso_path;
            if (!canva.cloth_fitting_depth_path.empty())
                canva.sequence_depth_path = canva.cloth_fitting_depth_path;
        }

        static bool sequence_have_cached_predictions(const CanvaState& canva)
        {
            return !canva.sequence_model_input_path.empty()
                && !canva.sequence_iso_path.empty()
                && !canva.sequence_depth_path.empty();
        }

        static float sequence_frame_duration(int frame_rate)
        {
            return 1.0f / (float)std::max(1, frame_rate);
        }

        static float sequence_duration_seconds(const CanvaState& canva)
        {
            const float cloth_duration =
                (float)canva.sequence_cloth_frames.size()
                * sequence_frame_duration(canva.sequence_frame_rate);
            const float sketch_duration =
                (float)canva.sequence_sketch_frames.size()
                    * (float)std::max(1, canva.sequence_sub_frames)
                    * sequence_frame_duration(canva.sequence_frame_rate);
            return std::max(cloth_duration, sketch_duration);
        }

        static int sequence_track_index_at_time(
            int frame_count,
            int frame_rate,
            float time_seconds,
            bool single_frame_spans_duration)
        {
            if (frame_count <= 0)
                return -1;
            if (single_frame_spans_duration && frame_count == 1)
                return 0;

            const int index = (int)std::floor(std::max(0.0f, time_seconds) * (float)std::max(1, frame_rate));
            return std::clamp(index, 0, frame_count - 1);
        }

        static int sequence_cloth_index_at_time(const CanvaState& canva, float time_seconds)
        {
            return sequence_track_index_at_time(
                (int)canva.sequence_cloth_frames.size(),
                canva.sequence_frame_rate,
                time_seconds,
                false);
        }

        static int sequence_sketch_index_at_time(const CanvaState& canva, float time_seconds)
        {
            const int frame_count = (int)canva.sequence_sketch_frames.size();
            if (frame_count <= 0)
                return -1;
            if (frame_count == 1)
                return 0;

            const int cloth_slot = (int)std::floor(
                std::max(0.0f, time_seconds)
                * (float)std::max(1, canva.sequence_frame_rate));
            const int sketch_index = cloth_slot / std::max(1, canva.sequence_sub_frames);
            return std::clamp(sketch_index, 0, frame_count - 1);
        }

        static std::string sequence_shell_quote(const std::string& value)
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

        static std::filesystem::path sequence_unique_export_dir(const char* stem)
        {
            const std::filesystem::path root = std::filesystem::path("renders") / "sequence_exports";
            const std::string base = stem && stem[0] != '\0' ? stem : "sequence";
            for (int i = 0; i < 10000; ++i)
            {
                char suffix[32];
                if (i == 0)
                    suffix[0] = '\0';
                else
                    std::snprintf(suffix, sizeof(suffix), "_%03d", i);
                const std::filesystem::path candidate = root / (base + suffix);
                std::error_code ec;
                if (!std::filesystem::exists(candidate, ec))
                    return candidate;
            }
            return root / (base + "_latest");
        }

        static void sequence_sample_layer_bilinear_rgba(
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

        static void sequence_blend_layer_over(CanvaState& canva, const CanvaLayer& layer, std::vector<unsigned char>& rgba)
        {
            const int w = canva.width;
            const int h = canva.height;
            if (!layer.enabled || layer.pixels.empty() || rgba.size() < (size_t)w * (size_t)h * 4u)
                return;

            const float s = std::max(1e-6f, layer.scale);
            const float min_x = layer.offset.x;
            const float min_y = layer.offset.y;
            const float max_x = layer.offset.x + (float)w * s;
            const float max_y = layer.offset.y + (float)h * s;

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
                    sequence_sample_layer_bilinear_rgba(layer, w, h, u, v, sr, sg, sb, sa);
                    sa = std::clamp(sa, 0.0f, 1.0f);
                    if (sa <= 0.0f)
                        continue;

                    const size_t idx = (((size_t)y * (size_t)w) + (size_t)x) * 4u;
                    const float dr = (float)rgba[idx + 0u] / 255.0f;
                    const float dg = (float)rgba[idx + 1u] / 255.0f;
                    const float db = (float)rgba[idx + 2u] / 255.0f;
                    const float da = (float)rgba[idx + 3u] / 255.0f;
                    const float out_a = sa + da * (1.0f - sa);
                    if (out_a <= 1e-6f)
                        continue;

                    const float out_r = (sr * sa + dr * da * (1.0f - sa)) / out_a;
                    const float out_g = (sg * sa + dg * da * (1.0f - sa)) / out_a;
                    const float out_b = (sb * sa + db * da * (1.0f - sa)) / out_a;
                    rgba[idx + 0u] = (unsigned char)std::clamp<int>((int)(out_r * 255.0f + 0.5f), 0, 255);
                    rgba[idx + 1u] = (unsigned char)std::clamp<int>((int)(out_g * 255.0f + 0.5f), 0, 255);
                    rgba[idx + 2u] = (unsigned char)std::clamp<int>((int)(out_b * 255.0f + 0.5f), 0, 255);
                    rgba[idx + 3u] = (unsigned char)std::clamp<int>((int)(out_a * 255.0f + 0.5f), 0, 255);
                }
            }
        }

        static bool sequence_export_frame_rgba(
            CanvaState& canva,
            int cloth_index,
            bool include_body,
            std::vector<unsigned char>& rgba,
            std::string* out_error)
        {
            if (cloth_index < 0 || cloth_index >= (int)canva.sequence_cloth_frames.size())
            {
                if (out_error)
                    *out_error = "invalid cloth frame index";
                return false;
            }

            const int w = canva.width;
            const int h = canva.height;
            rgba.assign((size_t)w * (size_t)h * 4u, include_body ? 255u : 0u);

            const float time_seconds = (float)cloth_index * sequence_frame_duration(canva.sequence_frame_rate);
            if (include_body)
            {
                const int sketch_index = sequence_sketch_index_at_time(canva, time_seconds);
                if (sketch_index >= 0)
                    sequence_blend_layer_over(canva, canva.sequence_sketch_frames[(size_t)sketch_index].layer, rgba);
            }

            sequence_blend_layer_over(canva, canva.sequence_cloth_frames[(size_t)cloth_index].layer, rgba);
            return true;
        }

        static bool sequence_ffmpeg_available()
        {
            return std::system("ffmpeg -version > /dev/null 2>&1") == 0;
        }

        static bool sequence_encode_mp4(
            const std::filesystem::path& export_dir,
            int frame_rate,
            bool include_body,
            int width,
            int height,
            std::filesystem::path* out_mp4_path)
        {
            if (!sequence_ffmpeg_available())
                return false;

            const std::filesystem::path input_pattern = export_dir / "frame_%04d.png";
            const std::filesystem::path output_path = export_dir / "sequence.mp4";
            std::ostringstream cmd;
            cmd << "ffmpeg -y -loglevel error"
                << " -framerate " << std::max(1, frame_rate)
                << " -i " << sequence_shell_quote(input_pattern.string());
            if (include_body)
            {
                cmd << " -vf " << sequence_shell_quote("format=yuv420p");
            }
            else
            {
                std::ostringstream filter;
                filter << "[0:v]format=rgba[fg];color=white:s="
                    << width << "x" << height
                    << ":r=" << std::max(1, frame_rate)
                    << "[bg];[bg][fg]overlay=shortest=1,format=yuv420p";
                cmd << " -filter_complex " << sequence_shell_quote(filter.str());
            }
            cmd << " " << sequence_shell_quote(output_path.string());

            const int rc = std::system(cmd.str().c_str());
            if (rc != 0 || !std::filesystem::exists(output_path))
                return false;
            if (out_mp4_path)
                *out_mp4_path = output_path;
            return true;
        }

        static bool sequence_export_pngs_and_mp4(
            AppState& state,
            CanvaState& canva,
            bool include_body,
            std::string* out_error)
        {
            if (canva.sequence_cloth_frames.empty())
            {
                if (out_error)
                    *out_error = "no cloth frames to export";
                return false;
            }

            const std::filesystem::path export_dir = sequence_unique_export_dir(
                include_body ? "body_garment" : "garment");
            std::error_code ec;
            std::filesystem::create_directories(export_dir, ec);
            if (ec)
            {
                if (out_error)
                    *out_error = "failed to create export directory: " + ec.message();
                return false;
            }

            std::vector<unsigned char> rgba;
            for (int i = 0; i < (int)canva.sequence_cloth_frames.size(); ++i)
            {
                if (!sequence_export_frame_rgba(canva, i, include_body, rgba, out_error))
                    return false;

                char file_name[32];
                std::snprintf(file_name, sizeof(file_name), "frame_%04d.png", i);
                const std::filesystem::path frame_path = export_dir / file_name;
                const int ok = stbi_write_png(
                    frame_path.string().c_str(),
                    canva.width,
                    canva.height,
                    4,
                    rgba.data(),
                    canva.width * 4);
                if (ok == 0)
                {
                    if (out_error)
                        *out_error = "failed to write PNG frame: " + frame_path.string();
                    return false;
                }
            }

            std::filesystem::path mp4_path;
            const bool mp4_ok = sequence_encode_mp4(
                export_dir,
                canva.sequence_frame_rate,
                include_body,
                canva.width,
                canva.height,
                &mp4_path);

            std::string msg = std::string("Sequence: exported ")
                + std::to_string((int)canva.sequence_cloth_frames.size())
                + " PNG frames to " + export_dir.string();
            if (mp4_ok)
                msg += " and MP4 " + mp4_path.string();
            else
                msg += " (MP4 skipped: ffmpeg unavailable or failed)";

            canva.sequence_export_status = msg;
            console_add_line(state, msg);
            return true;
        }

        static void sequence_assign_preview_layer(
            CanvaState& canva,
            int preview_index,
            const CanvaLayer* source)
        {
            if (preview_index < 0)
                return;
            if ((int)canva.sequence_preview_layers.size() <= preview_index)
                canva.sequence_preview_layers.resize((size_t)preview_index + 1u);

            const GLuint texture_id = canva.sequence_preview_layers[(size_t)preview_index].texture_id;
            if (source)
            {
                canva.sequence_preview_layers[(size_t)preview_index] = *source;
                canva.sequence_preview_layers[(size_t)preview_index].texture_id = texture_id;
                canva.sequence_preview_layers[(size_t)preview_index].enabled = true;
                canva.sequence_preview_layers[(size_t)preview_index].dirty = true;
                canva.sequence_preview_layers[(size_t)preview_index].bounds_dirty = true;
            }
            else
            {
                canva.sequence_preview_layers[(size_t)preview_index].texture_id = texture_id;
                canva.sequence_preview_layers[(size_t)preview_index].enabled = false;
                canva.sequence_preview_layers[(size_t)preview_index].dirty = false;
            }
        }

        static bool sequence_layer_is_frame_output(const CanvaLayer& layer)
        {
            switch (layer.kind)
            {
            case CanvaLayerKind::GeneratedCloth:
            case CanvaLayerKind::ClothFitting:
            case CanvaLayerKind::ClothFittingWarped:
            case CanvaLayerKind::ClothFittingDebug:
            case CanvaLayerKind::ClothFittingWarpedDebug:
            case CanvaLayerKind::ClothFittingOcclusionDebug:
            case CanvaLayerKind::ClothFittingSceneOcclusionDebug:
            case CanvaLayerKind::ClothFittingAlignedPredictedDepthOcclusionDebug:
            case CanvaLayerKind::ClothFittingIsoOcclusionDebug:
            case CanvaLayerKind::ClothFittingSceneDebug:
            case CanvaLayerKind::ClothFittingDeformation:
            case CanvaLayerKind::ClothFittingCloseToBodyDebug:
                return true;
            default:
                return false;
            }
        }

        static void sequence_remove_live_frame_output_layers(CanvaState& canva)
        {
            bool removed = false;
            for (int i = (int)canva.layers.size() - 1; i >= 0; --i)
            {
                CanvaLayer& layer = canva.layers[(size_t)i];
                if (!sequence_layer_is_frame_output(layer))
                    continue;

                const int removed_id = layer.id;
                canva_delete_layer_texture(layer);
                canva_prune_undo_for_layer(canva.undo, removed_id);
                canva.layers.erase(canva.layers.begin() + i);
                removed = true;
            }

            if (!removed)
                return;

            canva.active_layer = canva.layers.empty()
                ? -1
                : std::clamp(canva.active_layer, 0, (int)canva.layers.size() - 1);
            canva_cancel_interactions(canva);
        }

        static std::filesystem::path sequence_current_pose_path(const AppState& state)
        {
            if (state.selection.pose_index < 0
                || state.selection.pose_index >= (int)state.selection.poses.size())
                return {};
            const std::string& pose_name = state.selection.poses[(size_t)state.selection.pose_index];
            if (pose_name.empty())
                return {};
            return assets_dir() / "poses" / pose_name;
        }

        static bool sequence_gender_valid(const std::string& gender)
        {
            return gender == "male" || gender == "female" || gender == "neutral";
        }

        static int sequence_gender_index(const std::string& gender)
        {
            if (gender == "male")
                return 0;
            if (gender == "female")
                return 1;
            if (gender == "neutral")
                return 2;
            return -1;
        }

        static std::string sequence_gender_from_index(int index)
        {
            switch (index)
            {
            case 0: return "male";
            case 1: return "female";
            case 2: return "neutral";
            default: return "male";
            }
        }

        static std::string sequence_gender_from_smpl_type(SmplType gender)
        {
            if (gender == SmplType::Female)
                return "female";
            if (gender == SmplType::Neutral)
                return "neutral";
            return "male";
        }

        static void sequence_apply_gender_to_state(AppState& state, const std::string& gender)
        {
            if (gender == "female")
                state.selection.gender = SmplType::Female;
            else if (gender == "neutral")
                state.selection.gender = SmplType::Neutral;
            else
                state.selection.gender = SmplType::Male;
        }

        static std::optional<std::string> sequence_pose_gender(const std::filesystem::path& pose_path)
        {
            if (pose_path.empty())
                return std::nullopt;
            std::error_code ec;
            if (!std::filesystem::exists(pose_path, ec))
                return std::nullopt;

            std::optional<std::string> gender;
            std::string metadata_err;
            if (!smplh_load_pose_npz_metadata(pose_path, nullptr, &gender, &metadata_err))
                return std::nullopt;
            if (!gender.has_value() || !sequence_gender_valid(*gender))
                return std::nullopt;
            return gender;
        }

        static std::optional<std::string> sequence_frame_gender(const CanvaSequenceTrackFrame& frame)
        {
            if (sequence_gender_valid(frame.gender))
                return frame.gender;
            return sequence_pose_gender(frame.pose_npz_path);
        }

        static std::optional<std::string> sequence_previous_cloth_gender(const CanvaState& canva)
        {
            if (sequence_gender_valid(canva.sequence_locked_gender))
                return canva.sequence_locked_gender;

            for (int i = (int)canva.sequence_cloth_frames.size() - 1; i >= 0; --i)
            {
                const CanvaSequenceTrackFrame& cloth = canva.sequence_cloth_frames[(size_t)i];
                if (sequence_gender_valid(cloth.gender))
                    return cloth.gender;
                const int sketch_index = cloth.sketch_frame_index;
                if (sketch_index >= 0 && sketch_index < (int)canva.sequence_sketch_frames.size())
                {
                    const std::optional<std::string> sketch_gender =
                        sequence_frame_gender(canva.sequence_sketch_frames[(size_t)sketch_index]);
                    if (sketch_gender.has_value())
                        return sketch_gender;
                }
            }

            return std::nullopt;
        }

        static void sequence_count_gender(
            const CanvaSequenceTrackFrame& frame,
            int (&counts)[3])
        {
            const std::optional<std::string> gender = sequence_frame_gender(frame);
            if (!gender.has_value())
                return;
            const int index = sequence_gender_index(*gender);
            if (index >= 0)
                counts[index]++;
        }

        static std::string sequence_majority_sketch_gender(
            const AppState& state,
            const CanvaState& canva)
        {
            int counts[3] = { 0, 0, 0 };
            std::vector<int> counted_indices;
            counted_indices.reserve(canva.sequence_pending_render_sketch_indices.size());

            for (int sketch_index : canva.sequence_pending_render_sketch_indices)
            {
                if (sketch_index < 0 || sketch_index >= (int)canva.sequence_sketch_frames.size())
                    continue;
                if (std::find(counted_indices.begin(), counted_indices.end(), sketch_index)
                    != counted_indices.end())
                    continue;
                counted_indices.push_back(sketch_index);
                sequence_count_gender(canva.sequence_sketch_frames[(size_t)sketch_index], counts);
            }

            if (counts[0] + counts[1] + counts[2] == 0)
            {
                for (const CanvaSequenceTrackFrame& frame : canva.sequence_sketch_frames)
                    sequence_count_gender(frame, counts);
            }

            const int fallback = sequence_gender_index(sequence_gender_from_smpl_type(state.selection.gender));
            const int best_count = std::max(counts[0], std::max(counts[1], counts[2]));
            if (best_count <= 0)
                return sequence_gender_from_index(fallback >= 0 ? fallback : 0);
            if (fallback >= 0 && counts[fallback] == best_count)
                return sequence_gender_from_index(fallback);
            for (int i = 0; i < 3; ++i)
                if (counts[i] == best_count)
                    return sequence_gender_from_index(i);
            return "male";
        }

        static std::filesystem::path sequence_metadata_pose_for_gender(
            const CanvaState& canva,
            const std::vector<std::string>& pose_paths,
            const std::string& gender)
        {
            for (int i = (int)canva.sequence_cloth_frames.size() - 1; i >= 0; --i)
            {
                const CanvaSequenceTrackFrame& cloth = canva.sequence_cloth_frames[(size_t)i];
                const int sketch_index = cloth.sketch_frame_index;
                if (sketch_index < 0 || sketch_index >= (int)canva.sequence_sketch_frames.size())
                    continue;
                const CanvaSequenceTrackFrame& sketch = canva.sequence_sketch_frames[(size_t)sketch_index];
                const std::optional<std::string> sketch_gender = sequence_frame_gender(sketch);
                if (sketch_gender.has_value() && *sketch_gender == gender && !sketch.pose_npz_path.empty())
                    return sketch.pose_npz_path;
            }

            for (const std::string& pose_path : pose_paths)
            {
                const std::optional<std::string> pose_gender =
                    sequence_pose_gender(std::filesystem::path(pose_path));
                if (pose_gender.has_value() && *pose_gender == gender)
                    return std::filesystem::path(pose_path);
            }

            return pose_paths.empty() ? std::filesystem::path() : std::filesystem::path(pose_paths.front());
        }

        static void sequence_apply_sketch_pose_result(AppState& state, const SketchPoseBuildResult& result)
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
            state.show_smpl = true;
            state.show_smpl_character = true;
        }

        static bool sequence_queue_init_first_frame(
            AppState& state,
            CanvaState& canva,
            std::string* out_error)
        {
            if (canva.sequence_sketch_frames.empty())
            {
                if (out_error)
                    *out_error = "no sketch frames are loaded";
                return false;
            }
            const int first_index = canva.sequence_has_imported_video
                ? std::clamp(
                    canva.sequence_video_first_sketch_index,
                    0,
                    (int)canva.sequence_sketch_frames.size() - 1)
                : 0;
            if (!sequence_put_sketch_frame_on_canvas(canva, first_index, "Video first frame"))
            {
                if (out_error)
                    *out_error = "failed to put the first sketch frame on the canvas";
                return false;
            }

            canva.sequence_active_sketch_index = first_index;
            canva.sequence_sketch_frames[(size_t)first_index].inference_status = CanvaFrameInferenceStatus::Busy;
            CanvaSequenceTrackFrame& first_frame = canva.sequence_sketch_frames[(size_t)first_index];
            first_frame.sketch_frame_index = first_index;
            first_frame.frame_dir = canva_render_frame_dir(first_index);

            std::string err;
            if (canva_queue_extract_pose(state, canva, true, true, &err, first_frame.frame_dir))
            {
                canva.generate_cloth_active = true;
                canva.generate_cloth_waiting_pose = false;
                canva.generate_cloth_waiting_simulation = false;
                canva.generate_cloth_waiting_render = false;
                canva.generate_cloth_status = "Inference";
                canva.generate_cloth_failed_step = -1;
                canva.generate_cloth_fit_after_render = true;
                canva.generate_cloth_include_fitting_steps = true;
                canva.sequence_status = "Initializing";
                console_add_line(state, "Sequence: init queued Generate + Fit for first frame");
                return true;
            }

            canva.sequence_sketch_frames[(size_t)first_index].inference_status = CanvaFrameInferenceStatus::NotStarted;
            if (out_error)
                *out_error = err.empty() ? "Generate + Fit failed to queue" : err;
            return false;
        }

        static bool sequence_queue_sketch_inference(
            AppState& state,
            CanvaState& canva,
            int sketch_index,
            std::string* out_error)
        {
            if (sketch_index < 0 || sketch_index >= (int)canva.sequence_sketch_frames.size())
            {
                if (out_error)
                    *out_error = "invalid sketch frame index";
                return false;
            }
            if (state.sketch2rig.service == nullptr || state.sketch2rig.status != ModelInferenceStatus::Ready
                || state.sketch2betas.service == nullptr || state.sketch2betas.status != ModelInferenceStatus::Ready
                || state.sketch2iso.service == nullptr || state.sketch2iso.status != ModelInferenceStatus::Ready
                || state.sketch2depth.service == nullptr || state.sketch2depth.status != ModelInferenceStatus::Ready)
            {
                if (out_error)
                    *out_error = "sketch inference workers are not ready";
                return false;
            }

            CanvaSequenceTrackFrame& frame = canva.sequence_sketch_frames[(size_t)sketch_index];
            std::filesystem::path sketch_path;
            frame.sketch_frame_index = sketch_index;
            frame.frame_dir = canva_render_frame_dir(sketch_index);
            if (!save_normalized_model_input_png_to_path(
                    canva.width,
                    canva.height,
                    frame.layer.pixels,
                    frame.frame_dir / ("sequence_sketch_" + std::to_string(frame.id) + "_model_input.png"),
                    &sketch_path,
                    out_error))
                return false;

            const std::string base = "sequence_sketch_" + std::to_string(frame.id);
            frame.model_input_path = sketch_path;
            frame.rig_json_path = sketch_path.parent_path() / (base + "_rig.json");
            frame.betas_json_path = sketch_path.parent_path() / (base + "_betas.json");
            frame.iso_path = sketch_path.parent_path() / (base + "_iso.npz");
            frame.depth_path = sketch_path.parent_path() / (base + "_depth.json");
            frame.pose_json_path = assets_dir() / "poses" / (base + "_pose.json");
            frame.pose_npz_path = assets_dir() / "poses" / (base + "_pose.npz");
            frame.rig_request_id = state.sketch2rig.next_request_id++;
            frame.betas_request_id = state.sketch2betas.next_request_id++;
            frame.iso_request_id = state.sketch2iso.next_request_id++;
            frame.depth_request_id = state.sketch2depth.next_request_id++;

            std::string err;
            const bool rig_ok = state.sketch2rig.service->request_inference(
                frame.rig_request_id, sketch_path, frame.rig_json_path, &err);
            const bool betas_ok = rig_ok && state.sketch2betas.service->request_inference(
                frame.betas_request_id, sketch_path, frame.betas_json_path, &err);
            const bool iso_ok = betas_ok && state.sketch2iso.service->request_inference(
                frame.iso_request_id, sketch_path, frame.iso_path, &err);
            const bool depth_ok = iso_ok && state.sketch2depth.service->request_inference(
                frame.depth_request_id, sketch_path, frame.depth_path, &err);
            if (!depth_ok)
            {
                frame.inference_status = CanvaFrameInferenceStatus::Failed;
                if (out_error)
                    *out_error = err.empty() ? "failed to queue sketch inference" : err;
                return false;
            }

            state.sketch2rig.last_sketch_path = sketch_path.string();
            state.sketch2rig.status = ModelInferenceStatus::Busy;
            state.sketch2rig.active_request_id = frame.rig_request_id;
            state.sketch2rig.message = "Processing sequence sketch with Sketch2Rig...";
            state.sketch2betas.last_sketch_path = sketch_path.string();
            state.sketch2betas.status = ModelInferenceStatus::Busy;
            state.sketch2betas.active_request_id = frame.betas_request_id;
            state.sketch2betas.message = "Processing sequence sketch with Sketch2Betas...";
            state.sketch2iso.last_sketch_path = sketch_path.string();
            state.sketch2iso.status = ModelInferenceStatus::Busy;
            state.sketch2iso.active_request_id = frame.iso_request_id;
            state.sketch2iso.message = "Processing sequence sketch ISO...";
            state.sketch2depth.last_sketch_path = sketch_path.string();
            state.sketch2depth.status = ModelInferenceStatus::Busy;
            state.sketch2depth.active_request_id = frame.depth_request_id;
            state.sketch2depth.message = "Processing sequence sketch depth...";

            frame.inference_status = CanvaFrameInferenceStatus::Busy;
            canva.sequence_waiting_sketch_inference = true;
            canva.sequence_pending_sketch_index = sketch_index;
            canva.sequence_status = "Sketch inference";
            console_add_line(state, "Sequence: queued sketch inference for frame "
                + std::to_string(sketch_index));
            return true;
        }

        static bool sequence_update_pending_sketch_inference(AppState& state, CanvaState& canva)
        {
            if (!canva.sequence_waiting_sketch_inference)
                return true;
            const int sketch_index = canva.sequence_pending_sketch_index;
            if (sketch_index < 0 || sketch_index >= (int)canva.sequence_sketch_frames.size())
            {
                canva.sequence_waiting_sketch_inference = false;
                canva.sequence_status = "Failed";
                return false;
            }
            CanvaSequenceTrackFrame& frame = canva.sequence_sketch_frames[(size_t)sketch_index];
            const bool rig_done = state.sketch2rig.completed_request_id >= frame.rig_request_id;
            const bool betas_done = state.sketch2betas.completed_request_id >= frame.betas_request_id;
            const bool iso_done = state.sketch2iso.completed_request_id >= frame.iso_request_id;
            const bool depth_done = state.sketch2depth.completed_request_id >= frame.depth_request_id;
            if (!rig_done || !betas_done || !iso_done || !depth_done)
                return true;

            const bool ok =
                state.sketch2rig.status == ModelInferenceStatus::Ready
                && state.sketch2betas.status == ModelInferenceStatus::Ready
                && state.sketch2iso.status == ModelInferenceStatus::Ready
                && state.sketch2depth.status == ModelInferenceStatus::Ready
                && std::filesystem::exists(frame.rig_json_path)
                && std::filesystem::exists(frame.betas_json_path)
                && std::filesystem::exists(frame.iso_path)
                && std::filesystem::exists(frame.depth_path);
            if (!ok)
            {
                frame.inference_status = CanvaFrameInferenceStatus::Failed;
                canva.sequence_waiting_sketch_inference = false;
                canva.sequence_status = "Failed";
                console_add_line(state, "Sequence: sketch inference failed for frame "
                    + std::to_string(sketch_index));
                return false;
            }

            SketchPoseBuildResult result;
            std::string err;
            if (!save_sketch_pose_from_outputs(
                    frame.rig_json_path,
                    frame.betas_json_path,
                    frame.pose_json_path,
                    frame.pose_npz_path,
                    &result,
                    &err))
            {
                frame.inference_status = CanvaFrameInferenceStatus::Failed;
                canva.sequence_waiting_sketch_inference = false;
                canva.sequence_status = "Failed";
                console_add_line(state, "Sequence: failed to save sketch pose: "
                    + (err.empty() ? std::string("(unknown error)") : err));
                return false;
            }

            frame.pose_npz_path = result.npz_path;
            frame.pose_json_path = result.json_path;
            frame.gender = result.gender;
            frame.inference_status = CanvaFrameInferenceStatus::Done;
            sequence_apply_sketch_pose_result(state, result);
            canva.sequence_waiting_sketch_inference = false;
            canva.sequence_waiting_pose_reload = true;
            canva.sequence_status = "Loading pose";
            console_add_line(state, "Sequence: sketch inference ready for frame "
                + std::to_string(sketch_index));
            return true;
        }

        static void sequence_start_next_simulation(
            AppState& state,
            CanvaState& canva,
            int amount,
            bool pose_transition)
        {
            const int rate = std::clamp(canva.sequence_frame_rate, 1, 240);
            const int frame_amount = std::clamp(amount, 1, 1000);
            const double duration = (double)frame_amount / (double)rate;
            const double dt = std::max(1e-6, (double)solver_dt_seconds(state.xpbd_params));
            const int max_frames = std::max(2, (int)std::ceil(duration / dt) + 1);

            state.cloth_simulate_continue_from_last_frame = true;
            state.cloth_simulate_duration_override_seconds = (float)duration;
            state.cloth_simulate_max_frames_override = max_frames;
            state.cloth_simulate_pose_transition_enabled = pose_transition;
            state.cloth_simulate_transition_from_pose_path =
                pose_transition ? canva.sequence_pose_transition_from_pose_path.string() : std::string();
            state.cloth_simulate_pose_transition_frames =
                pose_transition ? std::clamp(state.canva_sequence_pose_transition_frames, 0, 240) : 0;
            canva.sequence_sim_frame_base_index = std::max(0, state.controls.frame_count);
            canva.sequence_sim_batch_amount = frame_amount;
            canva.sequence_processing_pose_transition = pose_transition;
            state.controls.current_frame = std::max(0, state.controls.frame_count - 1);
            state.controls.playing = false;
            state.controls.simulating = true;
            state.cloth_simulate_request_id++;

            canva.sequence_waiting_simulation = true;
            canva.sequence_status = pose_transition ? "Pose transition" : "Simulating";
            console_add_line(state, "Sequence: queued XPBD batch for "
                + std::to_string(frame_amount)
                + (pose_transition ? " pose-transition frames" : " frames"));
        }

        static bool sequence_sketch_frame_ready(const CanvaSequenceTrackFrame& frame)
        {
            std::error_code ec;
            return frame.inference_status == CanvaFrameInferenceStatus::Done
                && !frame.model_input_path.empty()
                && !frame.iso_path.empty()
                && !frame.depth_path.empty()
                && !frame.pose_npz_path.empty()
                && std::filesystem::exists(frame.model_input_path, ec)
                && std::filesystem::exists(frame.iso_path, ec)
                && std::filesystem::exists(frame.depth_path, ec)
                && std::filesystem::exists(frame.pose_npz_path, ec);
        }

        static bool sequence_ensure_current_sketch_frame(
            CanvaState& canva,
            std::string* out_error)
        {
            if (!canva.sequence_sketch_frames.empty())
                return true;

            std::vector<unsigned char> sketch_pixels;
            if (!canva_composite_visible_layers_rgba(canva, &sketch_pixels, out_error, true))
                return false;
            if (sketch_pixels.size() < (size_t)canva.width * (size_t)canva.height * 4u)
            {
                if (out_error)
                    *out_error = "current canvas has no sketch pixels";
                return false;
            }

            CanvaSequenceTrackFrame frame;
            frame.id = canva.sequence_next_frame_id++;
            frame.sketch_frame_index = 0;
            frame.frame_dir = canva_render_frame_dir(0);
            frame.layer = sequence_make_layer_from_pixels(
                canva,
                std::move(sketch_pixels),
                "Sketch",
                CanvaLayerKind::Generic,
                true);
            frame.inference_status = CanvaFrameInferenceStatus::NotStarted;
            canva.sequence_sketch_frames.push_back(std::move(frame));
            canva.sequence_active_sketch_index = 0;
            return true;
        }

        static int sequence_sketch_index_for_cloth_index(const CanvaState& canva, int cloth_index)
        {
            if (canva.sequence_sketch_frames.empty())
                return -1;
            const int sub_frames = std::max(1, canva.sequence_sub_frames);
            return std::clamp(
                cloth_index / sub_frames,
                0,
                (int)canva.sequence_sketch_frames.size() - 1);
        }

        static int sequence_hold_step_count(const AppState& state, const CanvaState& canva)
        {
            const double dt = std::max(1e-6, (double)solver_dt_seconds(state.xpbd_params));
            const int rate = std::clamp(canva.sequence_frame_rate, 1, 240);
            return std::max(1, (int)std::llround((1.0 / (double)rate) / dt));
        }

        static int sequence_transition_step_count(const AppState& state, const CanvaState& canva)
        {
            const double dt = std::max(1e-6, (double)solver_dt_seconds(state.xpbd_params));
            const int rate = std::clamp(canva.sequence_frame_rate, 1, 240);
            const int blend_frames = std::max(1, std::clamp(state.canva_sequence_pose_transition_frames, 0, 240));
            return std::max(1, (int)std::llround(((double)blend_frames / (double)rate) / dt));
        }

        static int sequence_initial_smooth_step_count(const AppState& state)
        {
            if (!state.controls.smooth_sequence)
                return 0;
            const double frac = std::clamp((double)state.controls.smooth_sequence_length, 0.0, 1.0);
            if (!(frac > 0.0))
                return 0;
            const int initial_steps = std::clamp(state.xpbd_params.simulation_steps, 1, 1000000);
            return std::clamp((int)std::llround(frac * (double)initial_steps), 0, initial_steps);
        }

        static bool sequence_build_batch_simulation_frames(
            const AppState& state,
            const CanvaState& canva,
            const std::vector<int>& sketch_indices,
            bool initial_smooth,
            std::vector<int>* out_sim_frames,
            int* out_step_count,
            std::string* out_error)
        {
            if (!out_sim_frames || !out_step_count)
                return false;
            out_sim_frames->clear();
            *out_step_count = 0;
            if (sketch_indices.empty())
            {
                if (out_error)
                    *out_error = "no cloth frames were planned";
                return false;
            }

            const int hold_steps = sequence_hold_step_count(state, canva);
            const int transition_steps = sequence_transition_step_count(state, canva);
            const int initial_steps = std::clamp(state.xpbd_params.simulation_steps, 1, 1000000);
            const int smooth_steps = initial_smooth
                ? sequence_initial_smooth_step_count(state)
                : 0;
            const int initial_hold_steps = initial_smooth
                ? std::max(0, initial_steps - smooth_steps)
                : hold_steps;
            int event_steps = initial_hold_steps;
            for (int i = 1; i < (int)sketch_indices.size(); ++i)
                event_steps += sketch_indices[(size_t)i] == sketch_indices[(size_t)i - 1u]
                    ? hold_steps
                    : transition_steps;

            int sim_frame = initial_smooth ? initial_steps : hold_steps;
            out_sim_frames->push_back(std::max(1, sim_frame));
            for (int i = 1; i < (int)sketch_indices.size(); ++i)
            {
                sim_frame += sketch_indices[(size_t)i] == sketch_indices[(size_t)i - 1u]
                    ? hold_steps
                    : transition_steps;
                out_sim_frames->push_back(std::max(1, sim_frame));
            }

            *out_step_count = std::max(1, smooth_steps + event_steps);
            return true;
        }

        static void sequence_fail_batch(
            AppState& state,
            CanvaState& canva,
            const std::string& message)
        {
            canva.sequence_generating = false;
            canva.sequence_waiting_simulation = false;
            canva.sequence_waiting_render = false;
            canva.sequence_waiting_sketch_inference = false;
            canva.sequence_waiting_pose_reload = false;
            canva.sequence_remaining_to_generate = 0;
            canva.sequence_pending_render_frame_indices.clear();
            canva.sequence_pending_render_sketch_indices.clear();
            canva.sequence_pending_render_cloth_frame_indices.clear();
            canva.sequence_pending_sketch_indices.clear();
            canva.sequence_pending_inference_cursor = 0;
            canva.sequence_pending_render_index = 0;
            canva.sequence_fitting_render_index = -1;
            canva.sequence_sim_frame_base_index = 0;
            canva.sequence_status = "Failed";
            console_add_line(state, "Sequence: " + message);
        }

        static bool sequence_start_batch_generation(
            AppState& state,
            CanvaState& canva,
            int amount,
            std::string* out_error)
        {
            if (!sequence_ensure_current_sketch_frame(canva, out_error))
                return false;
            const int frame_amount = std::clamp(amount, 1, 10000);
            const int start_cloth_index = (int)canva.sequence_cloth_frames.size();

            canva.sequence_pending_render_frame_indices.clear();
            canva.sequence_pending_render_sketch_indices.clear();
            canva.sequence_pending_render_cloth_frame_indices.clear();
            canva.sequence_pending_sketch_indices.clear();
            canva.sequence_pending_inference_cursor = 0;
            canva.sequence_pending_render_index = 0;
            canva.sequence_fitting_render_index = -1;
            canva.sequence_sim_batch_amount = frame_amount;
            canva.sequence_sim_frame_base_index = -1;
            canva.sequence_batch_initial_smooth = start_cloth_index == 0;
            canva.sequence_remaining_to_generate = frame_amount;

            std::vector<int> planned_sketch_indices;
            planned_sketch_indices.reserve((size_t)frame_amount);
            std::vector<int> sketch_counts(canva.sequence_sketch_frames.size(), 0);
            for (int i = 0; i < (int)canva.sequence_sketch_frames.size(); ++i)
                sketch_counts[(size_t)i] = canva_count_cloth_frames_for_sketch(canva, i);

            for (int offset = 0; offset < frame_amount; ++offset)
            {
                const int cloth_index = start_cloth_index + offset;
                const int sketch_index = sequence_sketch_index_for_cloth_index(canva, cloth_index);
                if (sketch_index < 0)
                {
                    if (out_error)
                        *out_error = "no sketch frame is available for planned cloth frame";
                    return false;
                }
                planned_sketch_indices.push_back(sketch_index);
                canva.sequence_pending_render_sketch_indices.push_back(sketch_index);
                canva.sequence_pending_render_cloth_frame_indices.push_back(sketch_counts[(size_t)sketch_index]++);
            }

            for (int sketch_index : planned_sketch_indices)
            {
                CanvaSequenceTrackFrame& frame = canva.sequence_sketch_frames[(size_t)sketch_index];
                if (sequence_sketch_frame_ready(frame))
                    continue;
                if (std::find(
                        canva.sequence_pending_sketch_indices.begin(),
                        canva.sequence_pending_sketch_indices.end(),
                        sketch_index) == canva.sequence_pending_sketch_indices.end())
                    canva.sequence_pending_sketch_indices.push_back(sketch_index);
            }

            int step_count = 0;
            if (!sequence_build_batch_simulation_frames(
                    state,
                    canva,
                    planned_sketch_indices,
                    canva.sequence_batch_initial_smooth,
                    &canva.sequence_pending_render_frame_indices,
                    &step_count,
                    out_error))
                return false;

            canva.sequence_generating = true;
            canva.sequence_waiting_simulation = false;
            canva.sequence_waiting_render = false;
            canva.sequence_waiting_sketch_inference = false;
            canva.sequence_waiting_pose_reload = false;
            canva.sequence_status = canva.sequence_pending_sketch_indices.empty()
                ? "Preparing simulation"
                : "Sketch inference";
            console_add_line(state, "Sequence: planned "
                + std::to_string(frame_amount)
                + " cloth frames across "
                + std::to_string((int)canva.sequence_pending_sketch_indices.size())
                + " sketch inference jobs");
            return true;
        }

        static bool sequence_start_batch_simulation(
            AppState& state,
            CanvaState& canva,
            std::string* out_error)
        {
            if (canva.sequence_pending_render_sketch_indices.empty()
                || canva.sequence_pending_render_frame_indices.empty())
            {
                if (out_error)
                    *out_error = "no planned cloth frames to simulate";
                return false;
            }
            std::vector<std::string> pose_paths;
            std::vector<std::string> rig_paths;
            pose_paths.reserve(canva.sequence_pending_render_sketch_indices.size());
            rig_paths.reserve(canva.sequence_pending_render_sketch_indices.size());
            for (int sketch_index : canva.sequence_pending_render_sketch_indices)
            {
                if (sketch_index < 0 || sketch_index >= (int)canva.sequence_sketch_frames.size())
                {
                    if (out_error)
                        *out_error = "planned sketch index is invalid";
                    return false;
                }
                const CanvaSequenceTrackFrame& frame = canva.sequence_sketch_frames[(size_t)sketch_index];
                if (!sequence_sketch_frame_ready(frame))
                {
                    if (out_error)
                        *out_error = "planned sketch frame is missing predictions or pose";
                    return false;
                }
                pose_paths.push_back(frame.pose_npz_path.string());
                rig_paths.push_back(frame.rig_json_path.string());
            }

            const int last_capture = *std::max_element(
                canva.sequence_pending_render_frame_indices.begin(),
                canva.sequence_pending_render_frame_indices.end());
            if (!pose_paths.empty())
            {
                const std::optional<std::string> previous_gender = sequence_previous_cloth_gender(canva);
                const std::string locked_gender = previous_gender.has_value()
                    ? *previous_gender
                    : sequence_majority_sketch_gender(state, canva);
                canva.sequence_locked_gender = locked_gender;
                sequence_apply_gender_to_state(state, locked_gender);

                std::optional<Eigen::VectorXd> betas;
                std::string metadata_err;
                if (smplh_load_pose_npz_metadata(
                        sequence_metadata_pose_for_gender(canva, pose_paths, locked_gender),
                        &betas,
                        nullptr,
                        &metadata_err))
                {
                    if (betas.has_value())
                    {
                        const int n = std::min<int>(
                            (int)state.smpl_shape.betas.size(),
                            (int)betas->size());
                        for (int i = 0; i < n; ++i)
                            state.smpl_shape.betas[(size_t)i] = (float)(*betas)[i];
                        for (int i = n; i < (int)state.smpl_shape.betas.size(); ++i)
                            state.smpl_shape.betas[(size_t)i] = 0.0f;
                    }
                }
                console_add_line(state, "Sequence: locked gender for generated frames: " + locked_gender);
            }
            const double dt = std::max(1e-6, (double)solver_dt_seconds(state.xpbd_params));
            state.cloth_simulate_continue_from_last_frame = false;
            state.cloth_simulate_duration_override_seconds = (float)((double)std::max(1, last_capture) * dt);
            state.cloth_simulate_max_frames_override = std::max(2, last_capture + 1);
            state.cloth_simulate_sequence_enabled = true;
            state.cloth_simulate_sequence_pose_paths = std::move(pose_paths);
            state.cloth_simulate_sequence_rig_paths = std::move(rig_paths);
            state.cloth_simulate_sequence_frame_rate = std::clamp(canva.sequence_frame_rate, 1, 240);
            state.cloth_simulate_sequence_transition_frames =
                std::clamp(state.canva_sequence_pose_transition_frames, 0, 240);
            state.cloth_simulate_sequence_initial_smooth = canva.sequence_batch_initial_smooth;
            state.controls.save_smpl_sequence = true;
            state.controls.current_frame = 0;
            state.controls.playing = false;
            state.controls.simulating = true;
            state.show_smpl = true;
            state.show_smpl_character = true;
            state.show_garment_mesh = true;
            state.cloth_simulate_request_id++;

            canva.sequence_waiting_simulation = true;
            canva.sequence_sim_frame_base_index = 0;
            canva.sequence_status = "Simulating";
            console_add_line(state, "Sequence: queued one XPBD simulation for "
                + std::to_string((int)canva.sequence_pending_render_sketch_indices.size())
                + " planned cloth frames");
            return true;
        }

        static bool sequence_queue_pending_render(
            AppState& state,
            CanvaState& canva,
            std::string* out_error)
        {
            const int index = canva.sequence_pending_render_index;
            if (index < 0 || index >= (int)canva.sequence_pending_render_frame_indices.size())
            {
                if (out_error)
                    *out_error = "pending render index is out of range";
                return false;
            }
            const int sketch_index = canva.sequence_pending_render_sketch_indices[(size_t)index];
            if (sketch_index < 0 || sketch_index >= (int)canva.sequence_sketch_frames.size())
            {
                if (out_error)
                    *out_error = "pending render sketch index is invalid";
                return false;
            }

            CanvaSequenceTrackFrame& sketch = canva.sequence_sketch_frames[(size_t)sketch_index];
            canva.sequence_active_sketch_index = sketch_index;
            canva.sequence_model_input_path = sketch.model_input_path;
            canva.sequence_iso_path = sketch.iso_path;
            canva.sequence_depth_path = sketch.depth_path;
            state.controls.current_frame = std::clamp(
                canva.sequence_pending_render_frame_indices[(size_t)index],
                0,
                std::max(0, state.controls.frame_count - 1));
            state.controls.playing = false;
            state.cloth_render_path.clear();
            state.cloth_render_fine = false;
            state.cloth_render_use_current_camera = false;
            state.cloth_render_stroke_reference_path = sketch.model_input_path.string();
            state.cloth_render_camera_rig_path = sketch.rig_json_path.string();
            canva_prepare_cloth_output_dirs(
                state,
                canva,
                sketch_index,
                canva.sequence_pending_render_cloth_frame_indices[(size_t)index]);
            canva.sequence_render_request_id = ++state.cloth_render_request_id;
            canva.sequence_waiting_render = true;
            canva.sequence_fitting_render_index = -1;
            canva.sequence_status = "Rendering";
            return true;
        }

    }

    bool canva_sequence_import_sketch_file(
        AppState& state,
        CanvaState& canva,
        const std::filesystem::path& file_path,
        std::string* out_error)
    {
        if (file_path.empty())
        {
            if (out_error)
                *out_error = "empty file path";
            return false;
        }
        if (!sequence_ffmpeg_available())
        {
            if (out_error)
                *out_error = "ffmpeg is required to import GIF/MP4 files";
            return false;
        }

        std::error_code ec;
        const std::filesystem::path import_root = std::filesystem::path("renders") / "sequence_imports";
        std::filesystem::create_directories(import_root, ec);
        if (ec)
        {
            if (out_error)
                *out_error = "failed to create sequence import directory: " + ec.message();
            return false;
        }

        std::string stem = file_path.stem().string();
        if (stem.empty())
            stem = "video";
        std::filesystem::path import_dir;
        for (int i = 0; i < 10000; ++i)
        {
            char suffix[32];
            std::snprintf(suffix, sizeof(suffix), "_%04d", i);
            import_dir = import_root / (stem + suffix);
            if (!std::filesystem::exists(import_dir, ec))
                break;
        }
        std::filesystem::create_directories(import_dir, ec);
        if (ec)
        {
            if (out_error)
                *out_error = "failed to create sequence import directory: " + ec.message();
            return false;
        }

        const std::filesystem::path output_pattern = import_dir / "frame_%05d.png";
        std::ostringstream cmd;
        cmd << "ffmpeg -y -loglevel error"
            << " -i " << sequence_shell_quote(file_path.string())
            << " -vsync 0 " << sequence_shell_quote(output_pattern.string());
        if (std::system(cmd.str().c_str()) != 0)
        {
            if (out_error)
                *out_error = "ffmpeg failed to extract frames";
            return false;
        }

        std::vector<std::filesystem::path> frames;
        for (const auto& entry : std::filesystem::directory_iterator(import_dir, ec))
        {
            if (ec)
                break;
            if (!entry.is_regular_file())
                continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                [](unsigned char c) { return (char)std::tolower(c); });
            if (ext == ".png")
                frames.push_back(entry.path());
        }
        std::sort(frames.begin(), frames.end());
        if (frames.empty())
        {
            if (out_error)
                *out_error = "video contained no decodable frames";
            return false;
        }

        int added = 0;
        int merged = 0;
        const int first_added_index = (int)canva.sequence_sketch_frames.size();
        std::vector<unsigned char> current_run_pixels;
        for (const std::filesystem::path& frame_path : frames)
        {
            std::vector<unsigned char> pixels;
            if (!sequence_load_image_as_canvas_pixels(frame_path, canva.width, canva.height, &pixels, out_error))
                return false;
            if (!current_run_pixels.empty() && sequence_pixels_identical_for_merge(current_run_pixels, pixels))
            {
                ++merged;
                continue;
            }

            CanvaSequenceTrackFrame frame;
            const int sketch_index = (int)canva.sequence_sketch_frames.size();
            frame.id = canva.sequence_next_frame_id++;
            frame.sketch_frame_index = sketch_index;
            frame.frame_dir = canva_render_frame_dir(sketch_index);
            frame.layer = sequence_make_layer_from_pixels(
                canva,
                pixels,
                "Sketch " + std::to_string(sketch_index + 1),
                CanvaLayerKind::Generic,
                true);
            frame.inference_status = CanvaFrameInferenceStatus::NotStarted;
            canva.sequence_sketch_frames.push_back(std::move(frame));
            current_run_pixels = std::move(pixels);
            ++added;
        }

        if (added <= 0)
        {
            if (out_error)
                *out_error = "all decoded frames were duplicates";
            return false;
        }

        canva.sequence_has_imported_video = true;
        canva.sequence_video_first_sketch_index = first_added_index;
        (void)sequence_put_sketch_frame_on_canvas(canva, first_added_index, "Video first frame");
        canva.sequence_status = "Ready";
        console_add_line(state,
            "Sequence: imported " + std::to_string(added)
                + " sketch frames from video"
                + (merged > 0 ? (" (merged " + std::to_string(merged) + " duplicate run frames)") : std::string())
                + "; each sketch frame uses Sub frames = "
                + std::to_string(std::max(1, canva.sequence_sub_frames)));
        return true;
    }

    void canva_sequence_select_time(
        AppState& state,
        CanvaState& canva,
        float time_seconds,
        bool sync_simulation)
    {
        const bool have_cloth = !canva.sequence_cloth_frames.empty();
        const bool have_sketch = !canva.sequence_sketch_frames.empty();
        if (!have_cloth && !have_sketch)
        {
            canva.sequence_selected_frame = -1;
            canva.sequence_time_seconds = 0.0f;
            canva.sequence_preview_active = false;
            canva.sequence_preview_frame_id = 0;
            return;
        }

        const float duration = sequence_duration_seconds(canva);
        canva.sequence_time_seconds = duration > 0.0f
            ? std::clamp(time_seconds, 0.0f, duration)
            : 0.0f;

        const int cloth_index = sequence_cloth_index_at_time(canva, canva.sequence_time_seconds);
        const int sketch_index = sequence_sketch_index_at_time(canva, canva.sequence_time_seconds);
        const CanvaLayer* cloth_layer = cloth_index >= 0
            ? &canva.sequence_cloth_frames[(size_t)cloth_index].layer
            : nullptr;
        const CanvaLayer* sketch_layer = sketch_index >= 0
            ? &canva.sequence_sketch_frames[(size_t)sketch_index].layer
            : nullptr;

        sequence_assign_preview_layer(canva, 0, cloth_layer);
        sequence_assign_preview_layer(canva, 1, sketch_layer);
        for (int i = 2; i < (int)canva.sequence_preview_layers.size(); ++i)
            canva.sequence_preview_layers[(size_t)i].enabled = false;

        canva.sequence_selected_frame = cloth_index;
        canva.sequence_preview_active = true;
        canva.sequence_preview_frame_id = cloth_index >= 0
            ? canva.sequence_cloth_frames[(size_t)cloth_index].id
            : 0;

        if (sync_simulation)
        {
            const int sim_frame = sequence_sim_frame_at_time(
                canva,
                canva.sequence_time_seconds,
                state.controls.frame_count);
            if (sim_frame >= 0)
            {
                state.controls.current_frame = sim_frame;
                canva.sequence_last_synced_sim_frame = sim_frame;
            }
        }
    }

    void canva_sequence_select_frame(AppState& state, CanvaState& canva, int frame_index)
    {
        if (frame_index < 0 || frame_index >= (int)canva.sequence_cloth_frames.size())
        {
            canva.sequence_selected_frame = -1;
            canva.sequence_preview_active = false;
            canva.sequence_preview_frame_id = 0;
            return;
        }

        canva_sequence_select_time(
            state,
            canva,
            (float)frame_index * sequence_frame_duration(canva.sequence_frame_rate));
    }

    void canva_sequence_sync_from_simulation(AppState& state, CanvaState& canva)
    {
        sequence_update_rig_overlay_for_sim_frame(state, canva);

        if (canva.sequence_cloth_frames.empty()
            || canva.sequence_playing
            || canva.sequence_generating
            || state.controls.frame_count <= 0)
            return;

        const int sim_frame = std::clamp(
            state.controls.current_frame,
            0,
            std::max(0, state.controls.frame_count - 1));
        if (sim_frame == canva.sequence_last_synced_sim_frame)
            return;

        const float sequence_time = sequence_time_for_sim_frame(canva, sim_frame);
        canva_sequence_select_time(state, canva, sequence_time, false);
        canva.sequence_last_synced_sim_frame = sim_frame;
    }

    void canva_sequence_capture_current_frame(AppState& state, CanvaState& canva)
    {
        const CanvaLayer* warped = sequence_find_latest_warped_layer(canva);
        if (!warped || warped->pixels.size() < (size_t)canva.width * (size_t)canva.height * 4u)
            return;

        sequence_store_cached_fitting_inputs(canva);

        const bool capture_sketch = canva.sequence_sketch_frames.empty();
        std::vector<unsigned char> sketch_pixels;
        if (capture_sketch)
        {
            std::string err;
            if (!canva_composite_visible_layers_rgba(canva, &sketch_pixels, &err, true))
            {
                console_add_line(state, "Sequence: failed to capture sketch layer: "
                    + (err.empty() ? std::string("(unknown error)") : err));
                return;
            }
        }

        const int new_cloth_index = (int)canva.sequence_cloth_frames.size();
        CanvaSequenceTrackFrame cloth_frame;
        cloth_frame.id = canva.sequence_next_frame_id++;
        cloth_frame.sketch_frame_index = std::max(0, canva.sequence_active_sketch_index);
        cloth_frame.cloth_frame_index = canva_count_cloth_frames_for_sketch(
            canva,
            cloth_frame.sketch_frame_index);
        cloth_frame.frame_dir = canva.sequence_current_frame_dir.empty()
            ? canva_render_frame_dir(cloth_frame.sketch_frame_index)
            : canva.sequence_current_frame_dir;
        cloth_frame.cloth_frame_dir = canva.sequence_current_cloth_frame_dir.empty()
            ? canva_render_cloth_frame_dir(cloth_frame.sketch_frame_index, cloth_frame.cloth_frame_index)
            : canva.sequence_current_cloth_frame_dir;
        cloth_frame.layer = sequence_copy_layer(
            *warped,
            canva.next_layer_id++,
            "Garment (warped)");
        cloth_frame.layer.enabled = true;
        cloth_frame.gender = sequence_gender_valid(canva.sequence_locked_gender)
            ? canva.sequence_locked_gender
            : sequence_gender_from_smpl_type(state.selection.gender);
        cloth_frame.cloth_status = CanvaClothFrameStatus::Ready;
        cloth_frame.sim_frame_index = std::clamp(
            state.controls.current_frame,
            0,
            std::max(0, state.controls.frame_count - 1));
        canva.sequence_cloth_frames.push_back(std::move(cloth_frame));

        if (capture_sketch)
        {
            CanvaSequenceTrackFrame sketch_frame;
            sketch_frame.id = canva.sequence_next_frame_id++;
            sketch_frame.sketch_frame_index = 0;
            sketch_frame.frame_dir = canva.sequence_current_frame_dir.empty()
                ? canva_render_frame_dir(0)
                : canva.sequence_current_frame_dir;
            sketch_frame.layer = sequence_make_layer_from_pixels(
                canva,
                std::move(sketch_pixels),
                "Sketch",
                CanvaLayerKind::Generic,
                true);
            sketch_frame.inference_status = CanvaFrameInferenceStatus::Done;
            sketch_frame.model_input_path = canva.sequence_model_input_path;
            sketch_frame.iso_path = canva.sequence_iso_path;
            sketch_frame.depth_path = canva.sequence_depth_path;
            sketch_frame.gender = cloth_frame.gender;
            sketch_frame.cloth_frame_indices.push_back(new_cloth_index);
            canva.sequence_sketch_frames.push_back(std::move(sketch_frame));
            canva.sequence_active_sketch_index = 0;
        }
        else if (canva.sequence_active_sketch_index >= 0
            && canva.sequence_active_sketch_index < (int)canva.sequence_sketch_frames.size())
        {
            CanvaSequenceTrackFrame& sketch_frame =
                canva.sequence_sketch_frames[(size_t)canva.sequence_active_sketch_index];
            sketch_frame.sketch_frame_index = canva.sequence_active_sketch_index;
            if (sketch_frame.frame_dir.empty())
                sketch_frame.frame_dir = canva.sequence_current_frame_dir.empty()
                    ? canva_render_frame_dir(canva.sequence_active_sketch_index)
                    : canva.sequence_current_frame_dir;
            sketch_frame.inference_status = CanvaFrameInferenceStatus::Done;
            sketch_frame.model_input_path = canva.sequence_model_input_path;
            sketch_frame.iso_path = canva.sequence_iso_path;
            sketch_frame.depth_path = canva.sequence_depth_path;
            if (std::find(
                    sketch_frame.cloth_frame_indices.begin(),
                    sketch_frame.cloth_frame_indices.end(),
                    new_cloth_index) == sketch_frame.cloth_frame_indices.end())
                sketch_frame.cloth_frame_indices.push_back(new_cloth_index);
            const std::filesystem::path pose_path = sequence_current_pose_path(state);
            if (!pose_path.empty())
                sketch_frame.pose_npz_path = pose_path;
        }

        canva_sequence_select_frame(state, canva, (int)canva.sequence_cloth_frames.size() - 1);
        canva.sequence_status = "Ready";
        console_add_line(state, "Sequence: captured cloth frame "
            + std::to_string((int)canva.sequence_cloth_frames.size()));
    }

    void canva_sequence_update_pipeline(AppState& state, CanvaState& canva, Renderer& renderer)
    {
        if (!canva.sequence_generating)
            return;

        if (canva.sequence_waiting_sketch_inference)
        {
            if (!sequence_update_pending_sketch_inference(state, canva))
            {
                sequence_fail_batch(state, canva, "sketch inference failed");
                return;
            }
            if (canva.sequence_waiting_sketch_inference)
                return;
            canva.sequence_waiting_pose_reload = false;
        }

        while (canva.sequence_pending_inference_cursor
            < (int)canva.sequence_pending_sketch_indices.size())
        {
            const int sketch_index =
                canva.sequence_pending_sketch_indices[(size_t)canva.sequence_pending_inference_cursor];
            if (sketch_index < 0 || sketch_index >= (int)canva.sequence_sketch_frames.size())
            {
                sequence_fail_batch(state, canva, "planned sketch index is invalid");
                return;
            }

            CanvaSequenceTrackFrame& sketch_frame = canva.sequence_sketch_frames[(size_t)sketch_index];
            if (sequence_sketch_frame_ready(sketch_frame))
            {
                ++canva.sequence_pending_inference_cursor;
                continue;
            }
            if (sketch_frame.inference_status == CanvaFrameInferenceStatus::Busy)
                return;

            std::string err;
            if (!sequence_queue_sketch_inference(state, canva, sketch_index, &err))
            {
                sequence_fail_batch(
                    state,
                    canva,
                    "failed to queue sketch inference: "
                        + (err.empty() ? std::string("(unknown error)") : err));
                return;
            }
            return;
        }

        if (!canva.sequence_waiting_simulation
            && !canva.sequence_waiting_render
            && !canva.cloth_fitting_active
            && !canva.sequence_pending_render_frame_indices.empty()
            && canva.sequence_sim_frame_base_index < 0)
        {
            std::string sim_err;
            if (!sequence_start_batch_simulation(state, canva, &sim_err))
            {
                sequence_fail_batch(
                    state,
                    canva,
                    "failed to start simulation: "
                        + (sim_err.empty() ? std::string("(unknown error)") : sim_err));
            }
            return;
        }

        if (canva.sequence_waiting_simulation && !state.controls.simulating)
        {
            canva.sequence_waiting_simulation = false;
            if (state.controls.frame_count <= 0)
            {
                sequence_fail_batch(state, canva, "XPBD produced no frames");
                return;
            }

            state.controls.playing = true;
            state.show_smpl = true;
            state.show_smpl_character = true;
            state.show_garment_mesh = true;
            canva.sequence_pending_render_index = 0;
            std::string render_err;
            if (!sequence_queue_pending_render(state, canva, &render_err))
            {
                sequence_fail_batch(
                    state,
                    canva,
                    "failed to queue first render: "
                        + (render_err.empty() ? std::string("(unknown error)") : render_err));
            }
            return;
        }

        if (canva.sequence_waiting_render
            && state.cloth_render_completed_id >= canva.sequence_render_request_id)
        {
            canva.sequence_waiting_render = false;
            if (state.cloth_render_path.empty())
            {
                sequence_fail_batch(state, canva, "cloth render failed");
                return;
            }

            sequence_remove_live_frame_output_layers(canva);

            std::string add_err;
            if (!canva_add_generated_cloth_layer(
                    canva,
                    state.cloth_render_path,
                    canva.sequence_model_input_path,
                    false,
                    &add_err))
            {
                sequence_fail_batch(
                    state,
                    canva,
                    "failed to add rendered cloth layer: "
                        + (add_err.empty() ? std::string("(unknown error)") : add_err));
                return;
            }

            std::string fit_err;
            canva.sequence_fitting_render_index = canva.sequence_pending_render_index;
            if (!canva_queue_cloth_fitting_cached_predictions(
                    state,
                    canva,
                    renderer,
                    canva.layers.front(),
                    canva.sequence_iso_path,
                    canva.sequence_depth_path,
                    canva.sequence_model_input_path,
                    canva.sequence_sketch_frames[(size_t)canva.sequence_active_sketch_index].rig_json_path,
                    &fit_err))
            {
                canva.sequence_fitting_render_index = -1;
                sequence_fail_batch(
                    state,
                    canva,
                    "cached fitting failed: "
                        + (fit_err.empty() ? std::string("(unknown error)") : fit_err));
                return;
            }
            canva.sequence_fitting_render_index = -1;

            canva.sequence_remaining_to_generate =
                std::max(0, canva.sequence_remaining_to_generate - 1);
            canva.sequence_pending_render_index++;
            if (canva.sequence_pending_render_index >= 0
                && canva.sequence_pending_render_index < (int)canva.sequence_pending_render_frame_indices.size())
            {
                std::string render_err;
                if (!sequence_queue_pending_render(state, canva, &render_err))
                {
                    sequence_fail_batch(
                        state,
                        canva,
                        "failed to queue render: "
                            + (render_err.empty() ? std::string("(unknown error)") : render_err));
                }
            }
            else
            {
                canva.sequence_generating = false;
                canva.sequence_pending_render_frame_indices.clear();
                canva.sequence_pending_render_sketch_indices.clear();
                canva.sequence_pending_render_cloth_frame_indices.clear();
                canva.sequence_pending_sketch_indices.clear();
                canva.sequence_pending_inference_cursor = 0;
                canva.sequence_pending_render_index = 0;
                canva.sequence_fitting_render_index = -1;
                canva.sequence_sim_batch_amount = 0;
                canva.sequence_status = "Done";
                state.controls.playing = true;
                console_add_line(state, "Sequence: generation complete");
            }
        }
    }

    void draw_sequence_tab(AppState& state, Renderer& renderer)
    {
        CanvaState& canva = canva_state();
        canva_ensure_initialized(state, canva);
        update_canva_frame(state, canva, renderer);
        sequence_update_rig_overlay_for_sim_frame(state, canva);

        canva.sequence_frame_rate = std::clamp(canva.sequence_frame_rate, 1, 240);
        canva.sequence_sub_frames = std::clamp(canva.sequence_sub_frames, 1, 1000);
        canva.sequence_generate_amount = std::clamp(canva.sequence_generate_amount, 1, 1000);

        const bool have_cloth = !canva.sequence_cloth_frames.empty();
        const bool have_sketch = !canva.sequence_sketch_frames.empty();
        const bool have_sequence = have_cloth || have_sketch;
        const float play_sequence_duration = sequence_duration_seconds(canva);

        if (canva.sequence_playing && have_sequence && play_sequence_duration > 0.0f)
        {
            const float dt = ImGui::GetIO().DeltaTime;
            float next_time = canva.sequence_time_seconds + dt;
            while (next_time >= play_sequence_duration && play_sequence_duration > 0.0f)
                next_time -= play_sequence_duration;
            canva_sequence_select_time(state, canva, next_time);
        }

        ImGui::SetNextItemWidth(120.0f);
        bool timing_changed = ImGui::InputInt("Cloth fps", &canva.sequence_frame_rate);
        canva.sequence_frame_rate = std::clamp(canva.sequence_frame_rate, 1, 240);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        timing_changed |= ImGui::InputInt("Sub frames", &canva.sequence_sub_frames);
        canva.sequence_sub_frames = std::clamp(canva.sequence_sub_frames, 1, 1000);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        ImGui::InputInt("Amount", &canva.sequence_generate_amount);
        canva.sequence_generate_amount = std::clamp(canva.sequence_generate_amount, 1, 1000);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        ImGui::InputInt("Pose blend", &state.canva_sequence_pose_transition_frames);
        state.canva_sequence_pose_transition_frames =
            std::clamp(state.canva_sequence_pose_transition_frames, 0, 240);

        if (timing_changed && have_sequence)
            canva_sequence_select_time(state, canva, canva.sequence_time_seconds);

        const float sequence_duration = sequence_duration_seconds(canva);

        if (have_sequence && sequence_duration > 0.0f)
        {
            float time = std::clamp(canva.sequence_time_seconds, 0.0f, sequence_duration);
            ImGui::SetNextItemWidth(360.0f);
            if (ImGui::SliderFloat("Time", &time, 0.0f, sequence_duration, "%.3f s"))
            {
                canva.sequence_playing = false;
                canva_sequence_select_time(state, canva, time);
            }
        }
        else
        {
            ImGui::TextUnformatted("Duration: 0.000 s");
        }

        const float cloth_step = sequence_frame_duration(canva.sequence_frame_rate);
        ImGui::BeginDisabled(!have_sequence);
        if (ImGui::Button("<<"))
        {
            canva.sequence_playing = false;
            canva_sequence_select_time(state, canva, 0.0f);
        }
        ImGui::SameLine();
        if (ImGui::Button("<"))
        {
            canva.sequence_playing = false;
            canva_sequence_select_time(state, canva, canva.sequence_time_seconds - cloth_step);
        }
        ImGui::SameLine();
        const char* play_label = canva.sequence_playing ? "II##sequence_play" : "Play##sequence_play";
        if (ImGui::Button(play_label))
            canva.sequence_playing = !canva.sequence_playing;
        ImGui::SameLine();
        if (ImGui::Button(">"))
        {
            canva.sequence_playing = false;
            canva_sequence_select_time(state, canva, canva.sequence_time_seconds + cloth_step);
        }
        ImGui::SameLine();
        if (ImGui::Button(">>"))
        {
            canva.sequence_playing = false;
            const float last_cloth_time = have_cloth
                ? (float)((int)canva.sequence_cloth_frames.size() - 1) * cloth_step
                : std::max(0.0f, sequence_duration - 0.001f);
            canva_sequence_select_time(state, canva, last_cloth_time);
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        const bool garment_selected =
            state.selection.garment_index >= 0
            && state.selection.garment_index < (int)state.selection.garments.size();
        const bool sequence_models_ready =
            state.sketch2rig.service != nullptr
            && state.sketch2rig.status == ModelInferenceStatus::Ready
            && state.sketch2betas.service != nullptr
            && state.sketch2betas.status == ModelInferenceStatus::Ready
            && state.sketch2iso.service != nullptr
            && state.sketch2iso.status == ModelInferenceStatus::Ready
            && state.sketch2depth.service != nullptr
            && state.sketch2depth.status == ModelInferenceStatus::Ready;
        const bool sequence_idle =
            !canva.extract_pose_active
            && !canva.generate_cloth_active
            && !canva.render_cloth_active
            && !canva.sequence_generating
            && !canva.cloth_fitting_active
            && !state.controls.simulating;
        const bool can_init =
            garment_selected
            && sequence_models_ready
            && sequence_idle;
        ImGui::BeginDisabled(!can_init);
        if (ImGui::Button("Init"))
        {
            canva.sequence_playing = false;
            std::string err;
            if (!sequence_start_batch_generation(state, canva, 1, &err))
            {
                canva.sequence_status = "Failed";
                console_add_line(state, "Sequence: Init failed: "
                    + (err.empty() ? std::string("(unknown error)") : err));
            }
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        const bool can_generate_next =
            garment_selected
            && sequence_models_ready
            && sequence_idle;
        ImGui::BeginDisabled(!can_generate_next);
        if (ImGui::Button("Generate Frame"))
        {
            canva.sequence_playing = false;
            std::string err;
            if (!sequence_start_batch_generation(state, canva, canva.sequence_generate_amount, &err))
            {
                canva.sequence_status = "Failed";
                console_add_line(state, "Sequence: Generate Frame failed: "
                    + (err.empty() ? std::string("(unknown error)") : err));
            }
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        const int generate_all_target = canva.sequence_has_imported_video
            ? sequence_video_target_cloth_frames(canva)
            : std::max(1, (int)canva.sequence_sketch_frames.size() * std::max(1, canva.sequence_sub_frames));
        const int generate_all_remaining = std::max(
            0,
            generate_all_target - (int)canva.sequence_cloth_frames.size());
        const bool can_generate_all =
            garment_selected
            && sequence_models_ready
            && sequence_idle
            && generate_all_remaining > 0;
        ImGui::BeginDisabled(!can_generate_all);
        if (ImGui::Button("Generate All"))
        {
            canva.sequence_playing = false;
            std::string err;
            if (!sequence_start_batch_generation(state, canva, generate_all_remaining, &err))
            {
                canva.sequence_status = "Failed";
                console_add_line(state, "Sequence: Generate All failed: "
                    + (err.empty() ? std::string("(unknown error)") : err));
            }
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::Text("Status: %s", canva.sequence_status.c_str());

        ImGui::BeginDisabled(!have_cloth);
        if (ImGui::Button("Export Garment"))
        {
            std::string export_err;
            if (!sequence_export_pngs_and_mp4(state, canva, false, &export_err))
            {
                canva.sequence_export_status = "Failed: "
                    + (export_err.empty() ? std::string("unknown export error") : export_err);
                console_add_line(state, "Sequence: garment export failed: "
                    + (export_err.empty() ? std::string("unknown export error") : export_err));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Export Body + Garment"))
        {
            std::string export_err;
            if (!sequence_export_pngs_and_mp4(state, canva, true, &export_err))
            {
                canva.sequence_export_status = "Failed: "
                    + (export_err.empty() ? std::string("unknown export error") : export_err);
                console_add_line(state, "Sequence: body + garment export failed: "
                    + (export_err.empty() ? std::string("unknown export error") : export_err));
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Text("Export: %s", canva.sequence_export_status.c_str());

        ImGui::Separator();
        ImGui::BeginChild("##sequence_timeline", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_HorizontalScrollbar);
        int displayed_cloth_frames = (int)canva.sequence_cloth_frames.size();
        if (canva.sequence_generating)
        {
            if (!canva.sequence_pending_render_frame_indices.empty())
                displayed_cloth_frames = std::max(
                    displayed_cloth_frames,
                    (int)canva.sequence_cloth_frames.size()
                        + (int)canva.sequence_pending_render_frame_indices.size()
                        - canva.sequence_pending_render_index);
            else if (canva.sequence_waiting_simulation)
                displayed_cloth_frames += std::max(
                    1,
                    canva.sequence_processing_pose_transition
                        ? std::clamp(state.canva_sequence_pose_transition_frames, 1, 240)
                        : canva.sequence_remaining_to_generate);
            else
                displayed_cloth_frames += std::max(0, canva.sequence_remaining_to_generate);
        }

        const int sub_frames = std::max(1, canva.sequence_sub_frames);
        const int sketch_slots = (int)canva.sequence_sketch_frames.size() * sub_frames;
        const int timeline_slots = std::max(1, std::max(displayed_cloth_frames, sketch_slots));
        const float cell_size = 28.0f;
        const float track_w = std::max(ImGui::GetContentRegionAvail().x - 72.0f, cell_size * (float)timeline_slots);
        auto sketch_cell_color = [&](int i) -> ImVec4
        {
            CanvaFrameInferenceStatus status = CanvaFrameInferenceStatus::NotStarted;
            if (i >= 0 && i < (int)canva.sequence_sketch_frames.size())
                status = canva.sequence_sketch_frames[(size_t)i].inference_status;
            switch (status)
            {
            case CanvaFrameInferenceStatus::Done: return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
            case CanvaFrameInferenceStatus::Busy: return ImVec4(1.0f, 0.55f, 0.12f, 1.0f);
            case CanvaFrameInferenceStatus::Failed:
            case CanvaFrameInferenceStatus::NotStarted:
            default: return ImVec4(1.0f, 0.18f, 0.18f, 1.0f);
            }
        };

        enum class ClothCellPhase
        {
            Done,
            NotStarted,
            SmplSequence,
            Xpbd,
            Fitting,
            Failed
        };

        auto planned_pending_index_for_cell = [&](int i) -> int
        {
            const int ready_count = (int)canva.sequence_cloth_frames.size();
            if (i < ready_count)
                return -1;
            const int pending_index = canva.sequence_pending_render_index + (i - ready_count);
            if (pending_index < 0 || pending_index >= (int)canva.sequence_pending_render_frame_indices.size())
                return -1;
            return pending_index;
        };

        auto planned_sim_frame_for_pending = [&](int pending_index) -> int
        {
            if (pending_index < 0 || pending_index >= (int)canva.sequence_pending_render_frame_indices.size())
                return 0;
            return std::max(0, canva.sequence_pending_render_frame_indices[(size_t)pending_index]);
        };

        auto cloth_cell_phase = [&](int i) -> ClothCellPhase
        {
            CanvaClothFrameStatus status = CanvaClothFrameStatus::NotStarted;
            if (i >= 0 && i < (int)canva.sequence_cloth_frames.size())
            {
                status = canva.sequence_cloth_frames[(size_t)i].cloth_status;
                switch (status)
                {
                case CanvaClothFrameStatus::Ready: return ClothCellPhase::Done;
                case CanvaClothFrameStatus::Fitting: return ClothCellPhase::Fitting;
                case CanvaClothFrameStatus::Simulating: return ClothCellPhase::Xpbd;
                case CanvaClothFrameStatus::Failed: return ClothCellPhase::Failed;
                case CanvaClothFrameStatus::NotStarted:
                default: return ClothCellPhase::NotStarted;
                }
            }

            const int pending_index = planned_pending_index_for_cell(i);
            if (pending_index < 0)
                return ClothCellPhase::NotStarted;

            if (canva.cloth_fitting_active
                && canva.sequence_fitting_render_index == pending_index)
                return ClothCellPhase::Fitting;

            const int planned_sim_frame = planned_sim_frame_for_pending(pending_index);
            const bool xpbd_reached =
                state.controls.sim_solver_steps_done >= planned_sim_frame
                || (!state.controls.simulating
                    && !canva.sequence_waiting_simulation
                    && state.controls.frame_count > planned_sim_frame);
            if (xpbd_reached)
                return ClothCellPhase::Xpbd;

            if (state.controls.sim_sequence_steps_done >= planned_sim_frame)
                return ClothCellPhase::SmplSequence;

            return ClothCellPhase::NotStarted;
        };

        auto cloth_cell_color = [](ClothCellPhase phase) -> ImVec4
        {
            switch (phase)
            {
            case ClothCellPhase::Done: return ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
            case ClothCellPhase::Fitting: return ImVec4(0.15f, 0.45f, 0.95f, 1.0f);
            case ClothCellPhase::Xpbd: return ImVec4(0.95f, 0.82f, 0.18f, 1.0f);
            case ClothCellPhase::SmplSequence: return ImVec4(0.95f, 0.50f, 0.12f, 1.0f);
            case ClothCellPhase::Failed:
            case ClothCellPhase::NotStarted:
            default: return ImVec4(0.82f, 0.16f, 0.16f, 1.0f);
            }
        };

        auto scaled_color = [](ImVec4 color, float scale) -> ImVec4
        {
            color.x = std::clamp(color.x * scale, 0.0f, 1.0f);
            color.y = std::clamp(color.y * scale, 0.0f, 1.0f);
            color.z = std::clamp(color.z * scale, 0.0f, 1.0f);
            return color;
        };

        auto draw_track = [&](const char* name, int frame_count, bool sketch_track)
        {
            ImGui::TextUnformatted(name);
            ImGui::SameLine(64.0f);
            if (frame_count <= 0)
            {
                ImGui::Button(("##empty_" + std::string(name)).c_str(), ImVec2(track_w, cell_size));
                return;
            }

            const int active_index = sketch_track
                ? sequence_sketch_index_at_time(canva, canva.sequence_time_seconds)
                : sequence_cloth_index_at_time(canva, canva.sequence_time_seconds);
            for (int i = 0; i < frame_count; ++i)
            {
                ImGui::PushID(name);
                ImGui::PushID(i);
                const bool selected = (i == active_index);
                if (selected)
                {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                }

                const float w = sketch_track ? cell_size * (float)sub_frames : cell_size;
                const std::string label = std::to_string(i);
                int pushed_cell_colors = 0;
                if (sketch_track)
                {
                    const ImVec4 base = sketch_cell_color(i);
                    ImGui::PushStyleColor(ImGuiCol_Button, base);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, scaled_color(base, 1.12f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, scaled_color(base, 0.88f));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.05f, 0.05f, 0.05f, 1.0f));
                    pushed_cell_colors = 4;
                }
                else
                {
                    const ClothCellPhase phase = cloth_cell_phase(i);
                    if (phase != ClothCellPhase::Done)
                    {
                        const ImVec4 base = cloth_cell_color(phase);
                        ImGui::PushStyleColor(ImGuiCol_Button, base);
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, scaled_color(base, 1.12f));
                        ImGui::PushStyleColor(ImGuiCol_ButtonActive, scaled_color(base, 0.88f));
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.05f, 0.05f, 0.05f, 1.0f));
                        pushed_cell_colors = 4;
                    }
                }
                if (ImGui::Button(label.c_str(), ImVec2(w, cell_size)))
                {
                    canva.sequence_playing = false;
                    const float select_time = sketch_track
                        ? (float)(i * sub_frames) * sequence_frame_duration(canva.sequence_frame_rate)
                        : (float)i * sequence_frame_duration(canva.sequence_frame_rate);
                    canva_sequence_select_time(state, canva, select_time);
                }
                if (pushed_cell_colors > 0)
                    ImGui::PopStyleColor(pushed_cell_colors);

                if (selected)
                    ImGui::PopStyleColor(2);
                if (i + 1 < frame_count)
                    ImGui::SameLine(0.0f, 0.0f);
                ImGui::PopID();
                ImGui::PopID();
            }
        };

        draw_track("Cloth", displayed_cloth_frames, false);
        draw_track("Sketch", (int)canva.sequence_sketch_frames.size(), true);
        ImGui::EndChild();
    }
}
