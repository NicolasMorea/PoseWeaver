#include "ui/canva/CanvaPanel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
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

namespace poseweaver::ui_canva
{
    namespace
    {
        enum class CanvaTool
        {
            Brush,
            Transform
        };

        struct CanvaLayer
        {
            int id = 0;
            std::string name;
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

            int active_layer = 0;
            int next_layer_number = 2;
            int next_layer_id = 1;

            std::vector<CanvaLayer> layers;
            std::vector<CanvaUndo> undo;
        };

        static CanvaLayer canva_make_layer(int w, int h, int id, const std::string& name);
        static void canva_cancel_interactions(CanvaState& canva);

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

        static bool canva_save_visible_layers_png(const CanvaState& canva, std::filesystem::path* out_path, std::string* out_error)
        {
            const int w = canva.width;
            const int h = canva.height;
            if (w <= 0 || h <= 0)
            {
                if (out_error)
                    *out_error = "Invalid canva size";
                return false;
            }

            std::vector<unsigned char> out_rgba;
            out_rgba.assign((size_t)w * (size_t)h * 4u, 255u); // opaque white background

            // Composite bottom->top to match rendering order.
            for (int li = (int)canva.layers.size() - 1; li >= 0; --li)
            {
                const CanvaLayer& layer = canva.layers[(size_t)li];
                if (!layer.enabled || layer.pixels.empty())
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
                        const float dr = (float)out_rgba[idx + 0u] / 255.0f;
                        const float dg = (float)out_rgba[idx + 1u] / 255.0f;
                        const float db = (float)out_rgba[idx + 2u] / 255.0f;

                        const float inv = 1.0f - sa;
                        const float or_ = sr * sa + dr * inv;
                        const float og_ = sg * sa + dg * inv;
                        const float ob_ = sb * sa + db * inv;

                        out_rgba[idx + 0u] = (unsigned char)std::clamp<int>((int)(or_ * 255.0f + 0.5f), 0, 255);
                        out_rgba[idx + 1u] = (unsigned char)std::clamp<int>((int)(og_ * 255.0f + 0.5f), 0, 255);
                        out_rgba[idx + 2u] = (unsigned char)std::clamp<int>((int)(ob_ * 255.0f + 0.5f), 0, 255);
                        out_rgba[idx + 3u] = 255u;
                    }
                }
            }

            std::error_code ec;
            const std::filesystem::path renders_dir = std::filesystem::path("renders");
            std::filesystem::create_directories(renders_dir, ec);

            const auto now = std::chrono::system_clock::now();
            const std::time_t tt = std::chrono::system_clock::to_time_t(now);
            std::tm tm{};
        #if defined(_WIN32)
            localtime_s(&tm, &tt);
        #else
            localtime_r(&tt, &tm);
        #endif

            std::ostringstream oss;
            oss << "canva_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
            const std::string stem = oss.str();

            std::filesystem::path out;
            for (int i = 0; i < 1000; ++i)
            {
                const std::string suffix = (i == 0) ? std::string() : ("_" + std::to_string(i));
                out = renders_dir / (stem + suffix + ".png");
                if (!std::filesystem::exists(out, ec))
                    break;
            }

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
    }

    void draw_canva_tab(AppState& state)
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

        // Consume undo requests routed from the GLFW key callback.
        while (canva.last_undo_request_id < state.canva_undo_request_id)
        {
            canva.last_undo_request_id++;
            console_add_line(state, "Canva: Ctrl+Z consumed (GLFW request)");
            const bool ok = canva_apply_undo(canva);
            console_add_line(state, ok ? "Canva: Undo applied" : "Canva: Undo stack empty");
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
        float right_w = 320.0f;
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

                // Hide OS cursor and draw a custom cursor/preview when hovering the canvas.
                if (hovered && in_canvas)
                {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_None);
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
                const char* path = tinyfd_openFileDialog(
                    "Load image",
                    "",
                    (int)(sizeof(patterns) / sizeof(patterns[0])),
                    patterns,
                    "Image files (*.png, *.jpg, *.jpeg)",
                    0);

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
