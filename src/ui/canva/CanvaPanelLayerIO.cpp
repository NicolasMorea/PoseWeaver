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
     bool canva_parse_number_array_field(
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

     std::string canva_parse_string_field(const std::string& text, const char* key)
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

     bool canva_parse_number_field(const std::string& text, const char* key, double* out_value)
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

     bool canva_apply_sketch2betas_result(
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

     bool canva_try_finish_extract_pose(AppState& state, CanvaState& canva)
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
            canva.generate_cloth_fit_after_render = false;
            canva_generate_cloth_finish_step(canva, GenerateClothStepId::Inference);
            canva_generate_cloth_finish_pipeline(canva);
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
            canva_generate_cloth_finish_step(canva, GenerateClothStepId::Inference);
            canva_generate_cloth_start_step(canva, GenerateClothStepId::Pose);
            console_add_line(state, "Canva: Generate Cloth pose extracted; waiting for pose reload");
        }
        canva.extract_pose_hide_rig_layer = false;
        console_add_line(state, std::string("Canva: Saved sketch pose JSON: ") + result.json_path.string());
        console_add_line(state, std::string("Canva: Saved sketch pose NPZ: ") + result.npz_path.string());
        return true;
    }

     bool canva_load_preprocess_transform(
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

     std::string canva_basename(std::string path)
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


     CanvaBounds canva_layer_content_bounds(CanvaLayer& layer, int w, int h)
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

     bool canva_add_png_layer(CanvaState& canva, const char* file_path, std::string* out_error)
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

     bool canva_add_canvas_filled_png_layer(
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

     bool canva_add_preprocess_aligned_png_layer(
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

     bool canva_add_smpl_render_layer(
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

     bool canva_add_generated_cloth_layer(
        CanvaState& canva,
        const std::string& file_path,
        const std::filesystem::path& model_input_path,
        bool enabled,
        std::string* out_error)
    {
        return canva_add_preprocess_aligned_png_layer(
            canva,
            file_path,
            model_input_path,
            "Generated cloth",
            CanvaLayerKind::GeneratedCloth,
            enabled,
            out_error);
    }

     bool canva_add_rendered_cloth_layer(
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

     bool canva_add_sketch2uv_layer(
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

     bool canva_add_sketch2depth_layer(
        CanvaState& canva,
        const std::filesystem::path& json_path,
        const std::filesystem::path& model_input_path,
        std::string* out_error)
    {
        std::ifstream in(json_path);
        if (!in)
        {
            if (out_error)
                *out_error = "Failed to open " + json_path.string();
            return false;
        }

        std::ostringstream stream;
        stream << in.rdbuf();
        std::filesystem::path render_path =
            canva_parse_string_field(stream.str(), "render_path");
        if (render_path.empty())
            render_path = json_path.parent_path() / (json_path.stem().string() + "_render.png");
        if (render_path.is_relative())
            render_path = json_path.parent_path() / render_path;

        return canva_add_preprocess_aligned_png_layer(
            canva,
            render_path.string(),
            model_input_path,
            "Sketch2Depth",
            CanvaLayerKind::Sketch2Depth,
            true,
            out_error);
    }

     bool canva_add_iso_debug_layer(
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

     int canva_find_layer_index_by_id(const CanvaState& canva, int layer_id)
    {
        for (int i = 0; i < (int)canva.layers.size(); ++i)
        {
            if (canva.layers[(size_t)i].id == layer_id)
                return i;
        }
        return -1;
    }

     bool canva_manual_uv_source_path(const AppState& state, std::filesystem::path* out_path)
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

     bool canva_manual_iso_source_path(const AppState& state, std::filesystem::path* out_path)
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

}
