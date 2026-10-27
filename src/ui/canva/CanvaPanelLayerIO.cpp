#include "ui/canva/CanvaPanelInternal.h"

#include "ui/canva/CanvaPreprocess.h"
#include "core/Paths.h"
#include "render/Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
        const bool saved_pose = canva.extract_pose_direct_nlf
            ? save_nlf_pose_from_output(
                canva.extract_pose_rig_path,
                out_json,
                out_npz,
                &result,
                &err)
            : save_sketch_pose_from_outputs(
                canva.extract_pose_rig_path,
                canva.extract_pose_betas_path,
                out_json,
                out_npz,
                &result,
                &err);
        if (!saved_pose)
        {
            console_add_line(state, std::string("Canva: Extract Pose failed: ") + (err.empty() ? "(unknown error)" : err));
            canva.extract_pose_active = false;
            canva.extract_pose_hide_rig_layer = false;
            canva.extract_pose_generate_cloth_after = false;
            canva.extract_pose_using_nlf = false;
            canva.extract_pose_nlf_fallback_allowed = false;
            canva.extract_pose_direct_nlf = false;
            canva.bone_prediction_debug_mode = 4;
            canva.bone_prediction_debug_status = "Bone prediction failed";
            canva.generate_cloth_active = false;
            canva.generate_cloth_status = "Failed";
            canva.generate_cloth_failed_step = 0;
            canva.generate_cloth_fit_after_render = false;
            canva_generate_cloth_finish_step(canva, GenerateClothStepId::Inference);
            canva_generate_cloth_finish_pipeline(canva);
            return false;
        }

        canva_apply_sketch_pose_result(state, result);
        const bool used_direct_nlf = canva.extract_pose_direct_nlf;
        state.show_smpl = true;
        state.show_smpl_character = true;
        state.show_sketch2rig_rig = true;
        if (used_direct_nlf)
        {
            std::string layer_err;
            if (canva_add_nlf_projected_smpl_layer(
                    canva,
                    canva.extract_pose_rig_path,
                    canva.extract_pose_model_input_path,
                    &layer_err))
            {
                console_add_line(state, "Canva: Added NLF projected SMPL layer");
                canva.pending_scene_view_snapshot_request_id = 0;
                canva.pending_scene_view_snapshot_model_input_path.clear();
            }
            else
            {
                console_add_line(state, std::string("Canva: Failed to add NLF projected SMPL layer: ")
                    + (layer_err.empty() ? "(unknown error)" : layer_err));
            }
        }
        state.scene_view_snapshot_pose_name = result.npz_path.filename().string();
        state.scene_view_snapshot_path.clear();
        state.scene_view_snapshot_include_cloth = false;
        state.scene_view_snapshot_use_rigopt_camera = false;
        canva.pending_scene_view_snapshot_request_id = ++state.scene_view_snapshot_request_id;
        canva.pending_scene_view_snapshot_include_cloth = false;
        canva.pending_scene_view_snapshot_model_input_path = canva.extract_pose_model_input_path;
        state.scene_view_snap_to_target_request_id++;
        canva.extract_pose_active = false;
        canva.extract_pose_using_nlf = false;
        canva.extract_pose_nlf_fallback_allowed = false;
        canva.extract_pose_direct_nlf = false;
        if (canva.extract_pose_generate_cloth_after)
        {
            canva.generate_cloth_active = true;
            canva.generate_cloth_waiting_pose = true;
            canva.generate_cloth_waiting_simulation = false;
            canva.generate_cloth_waiting_render = false;
            if (canva.generate_cloth_model_input_path.empty())
                canva.generate_cloth_model_input_path = canva.extract_pose_model_input_path;
            canva.generate_cloth_rig_json_path = canva.extract_pose_rig_path;
            canva.generate_cloth_pose_name = result.npz_path.filename().string();
            canva.extract_pose_generate_cloth_after = false;
            canva.generate_cloth_status = "Posing";
            canva.generate_cloth_failed_step = -1;
            canva_generate_cloth_finish_step(canva, GenerateClothStepId::Inference);
            canva_generate_cloth_start_step(canva, GenerateClothStepId::Pose);
            console_add_line(state, "Canva: Generate Cloth pose extracted; waiting for pose reload");
        }
        canva.extract_pose_hide_rig_layer = false;
        canva.extract_pose_using_nlf = false;
        canva.extract_pose_nlf_fallback_allowed = false;
        canva.extract_pose_direct_nlf = false;
        console_add_line(state, std::string(used_direct_nlf
                ? "Canva: Saved direct NLF pose JSON: "
                : "Canva: Saved sketch pose JSON: ")
            + result.json_path.string());
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

        // Fit the image to the current canva while preserving aspect ratio.
        const float s = std::min((float)w / (float)std::max(1, iw), (float)h / (float)std::max(1, ih));
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
                false,
                out_error))
            return false;
        return true;
    }

     bool canva_add_scene_render_layer(
        CanvaState& canva,
        const std::string& file_path,
        const std::filesystem::path& model_input_path,
        std::string* out_error)
    {
        return canva_add_preprocess_aligned_png_layer(
            canva,
            file_path,
            model_input_path,
            "3D scene",
            CanvaLayerKind::Scene,
            true,
            out_error);
    }

    static void canva_blend_rgba(
        std::vector<unsigned char>& pixels,
        int w,
        int h,
        int x,
        int y,
        unsigned char r,
        unsigned char g,
        unsigned char b,
        unsigned char a)
    {
        if (x < 0 || y < 0 || x >= w || y >= h || a == 0)
            return;

        const size_t idx = (((size_t)y * (size_t)w) + (size_t)x) * 4u;
        const float sa = (float)a / 255.0f;
        const float da = (float)pixels[idx + 3u] / 255.0f;
        const float out_a = sa + da * (1.0f - sa);
        if (out_a <= 1e-6f)
            return;

        const float sr = (float)r / 255.0f;
        const float sg = (float)g / 255.0f;
        const float sb = (float)b / 255.0f;
        const float dr = (float)pixels[idx + 0u] / 255.0f;
        const float dg = (float)pixels[idx + 1u] / 255.0f;
        const float db = (float)pixels[idx + 2u] / 255.0f;

        pixels[idx + 0u] = (unsigned char)std::clamp<int>(
            (int)(((sr * sa + dr * da * (1.0f - sa)) / out_a) * 255.0f + 0.5f),
            0,
            255);
        pixels[idx + 1u] = (unsigned char)std::clamp<int>(
            (int)(((sg * sa + dg * da * (1.0f - sa)) / out_a) * 255.0f + 0.5f),
            0,
            255);
        pixels[idx + 2u] = (unsigned char)std::clamp<int>(
            (int)(((sb * sa + db * da * (1.0f - sa)) / out_a) * 255.0f + 0.5f),
            0,
            255);
        pixels[idx + 3u] = (unsigned char)std::clamp<int>((int)(out_a * 255.0f + 0.5f), 0, 255);
    }

    static void canva_draw_line_rgba(
        std::vector<unsigned char>& pixels,
        int w,
        int h,
        float x0,
        float y0,
        float x1,
        float y1,
        unsigned char r,
        unsigned char g,
        unsigned char b,
        unsigned char a)
    {
        const float dx = x1 - x0;
        const float dy = y1 - y0;
        const int steps = std::max(1, (int)std::ceil(std::max(std::abs(dx), std::abs(dy))));
        for (int i = 0; i <= steps; ++i)
        {
            const float t = (float)i / (float)steps;
            const int x = (int)std::lround(x0 + dx * t);
            const int y = (int)std::lround(y0 + dy * t);
            canva_blend_rgba(pixels, w, h, x, y, r, g, b, a);
        }
    }

    static bool canva_load_smpl_faces(std::vector<std::uint32_t>* out_faces, std::string* out_error)
    {
        if (!out_faces)
            return false;
        out_faces->clear();

        cnpy::NpyArray faces;
        try
        {
            faces = cnpy::npz_load((assets_dir() / "smplh" / "neutral" / "model.npz").string(), "f");
        }
        catch (const std::exception& e)
        {
            if (out_error)
                *out_error = std::string("Failed to load SMPL faces: ") + e.what();
            return false;
        }

        if (faces.shape.size() != 2 || faces.shape[1] != 3 || (faces.word_size != 4 && faces.word_size != 8))
        {
            if (out_error)
                *out_error = "SMPL face topology has an unsupported shape or integer type";
            return false;
        }

        const size_t n = faces.shape[0] * 3u;
        out_faces->resize(n);
        if (faces.word_size == 4)
        {
            const std::uint32_t* src = faces.data<std::uint32_t>();
            std::copy(src, src + n, out_faces->begin());
        }
        else
        {
            const std::uint64_t* src = faces.data<std::uint64_t>();
            for (size_t i = 0; i < n; ++i)
                (*out_faces)[i] = (std::uint32_t)src[i];
        }
        return true;
    }

    static bool canva_render_nlf_projected_smpl_png(
        const std::string& nlf_text,
        const std::filesystem::path& out_path,
        std::string* out_error)
    {
        std::vector<float> vertices2d;
        if (!canva_parse_number_array_field(nlf_text, "vertices2d_flat", &vertices2d, out_error))
            return false;

        double vertex_count_d = 0.0;
        if (!canva_parse_number_field(nlf_text, "vertices2d_count", &vertex_count_d))
        {
            if (out_error)
                *out_error = "NLF result is missing vertices2d_count";
            return false;
        }
        const int vertex_count = std::max(0, (int)std::lround(vertex_count_d));
        if (vertex_count <= 0 || vertices2d.size() < (size_t)vertex_count * 2u)
        {
            if (out_error)
                *out_error = "NLF result has invalid projected SMPL vertices";
            return false;
        }

        double image_width_d = 512.0;
        double image_height_d = 512.0;
        double image_size_d = 512.0;
        (void)canva_parse_number_field(nlf_text, "image_size", &image_size_d);
        if (!canva_parse_number_field(nlf_text, "image_width", &image_width_d))
            image_width_d = image_size_d;
        if (!canva_parse_number_field(nlf_text, "image_height", &image_height_d))
            image_height_d = image_size_d;
        const int w = std::clamp((int)std::lround(image_width_d), 2, 8192);
        const int h = std::clamp((int)std::lround(image_height_d), 2, 8192);

        std::vector<std::uint32_t> faces;
        if (!canva_load_smpl_faces(&faces, out_error))
            return false;

        std::vector<unsigned char> pixels((size_t)w * (size_t)h * 4u, 0u);
        constexpr unsigned char r = 10u;
        constexpr unsigned char g = 122u;
        constexpr unsigned char b = 219u;
        constexpr unsigned char face_a = 42u;
        constexpr unsigned char edge_a = 90u;

        for (size_t f = 0; f + 2u < faces.size(); f += 3u)
        {
            const std::uint32_t ia = faces[f + 0u];
            const std::uint32_t ib = faces[f + 1u];
            const std::uint32_t ic = faces[f + 2u];
            if (ia >= (std::uint32_t)vertex_count || ib >= (std::uint32_t)vertex_count || ic >= (std::uint32_t)vertex_count)
                continue;

            const float ax = vertices2d[(size_t)ia * 2u + 0u];
            const float ay = vertices2d[(size_t)ia * 2u + 1u];
            const float bx = vertices2d[(size_t)ib * 2u + 0u];
            const float by = vertices2d[(size_t)ib * 2u + 1u];
            const float cx = vertices2d[(size_t)ic * 2u + 0u];
            const float cy = vertices2d[(size_t)ic * 2u + 1u];
            if (!std::isfinite(ax) || !std::isfinite(ay)
                || !std::isfinite(bx) || !std::isfinite(by)
                || !std::isfinite(cx) || !std::isfinite(cy))
                continue;

            const float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            if (std::abs(area) < 1e-6f)
                continue;

            int x0 = (int)std::floor(std::min({ ax, bx, cx }));
            int y0 = (int)std::floor(std::min({ ay, by, cy }));
            int x1 = (int)std::ceil(std::max({ ax, bx, cx }));
            int y1 = (int)std::ceil(std::max({ ay, by, cy }));
            x0 = std::clamp(x0, 0, w - 1);
            y0 = std::clamp(y0, 0, h - 1);
            x1 = std::clamp(x1, 0, w - 1);
            y1 = std::clamp(y1, 0, h - 1);

            for (int y = y0; y <= y1; ++y)
            {
                const float py = (float)y + 0.5f;
                for (int x = x0; x <= x1; ++x)
                {
                    const float px = (float)x + 0.5f;
                    const float w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
                    const float w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
                    const float w2 = 1.0f - w0 - w1;
                    if (w0 >= -1e-4f && w1 >= -1e-4f && w2 >= -1e-4f)
                        canva_blend_rgba(pixels, w, h, x, y, r, g, b, face_a);
                }
            }

            canva_draw_line_rgba(pixels, w, h, ax, ay, bx, by, r, g, b, edge_a);
            canva_draw_line_rgba(pixels, w, h, bx, by, cx, cy, r, g, b, edge_a);
            canva_draw_line_rgba(pixels, w, h, cx, cy, ax, ay, r, g, b, edge_a);
        }

        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec)
        {
            if (out_error)
                *out_error = "Failed to create NLF SMPL render directory: " + ec.message();
            return false;
        }
        if (stbi_write_png(out_path.string().c_str(), w, h, 4, pixels.data(), w * 4) == 0)
        {
            if (out_error)
                *out_error = "Failed to write NLF projected SMPL layer: " + out_path.string();
            return false;
        }
        return true;
    }

     bool canva_add_nlf_projected_smpl_layer(
        CanvaState& canva,
        const std::filesystem::path& nlf_json_path,
        const std::filesystem::path& model_input_path,
        std::string* out_error)
    {
        std::ifstream in(nlf_json_path);
        if (!in)
        {
            if (out_error)
                *out_error = "Failed to open NLF JSON: " + nlf_json_path.string();
            return false;
        }
        std::ostringstream ss;
        ss << in.rdbuf();

        const std::filesystem::path out_path = std::filesystem::path("renders") / "canva_nlf_smpl_layer.png";
        if (!canva_render_nlf_projected_smpl_png(ss.str(), out_path, out_error))
            return false;

        return canva_add_preprocess_aligned_png_layer(
            canva,
            out_path.string(),
            model_input_path,
            "NLF mesh overlay",
            CanvaLayerKind::Smpl,
            false,
            out_error);
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
            layer_name,
            CanvaLayerKind::Sketch2Depth,
            enabled,
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
