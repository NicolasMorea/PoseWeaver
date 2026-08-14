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
    static bool canva_parse_json_number(const std::string& text, size_t* pos, double* out_value)
    {
        if (!pos || !out_value)
            return false;

        size_t i = *pos;
        while (i < text.size() && std::isspace((unsigned char)text[i]))
            ++i;

        const char* begin = text.c_str() + i;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin)
            return false;

        *out_value = value;
        *pos = (size_t)(end - text.c_str());
        return true;
    }

    static bool canva_parse_json_int_field(const std::string& object, const char* key, int* out_value)
    {
        const std::string needle = std::string("\"") + key + "\":";
        size_t pos = object.find(needle);
        if (pos == std::string::npos)
            return false;
        pos += needle.size();

        double value = 0.0;
        if (!canva_parse_json_number(object, &pos, &value))
            return false;

        *out_value = (int)std::lround(value);
        return true;
    }

    static bool canva_parse_json_float_field(const std::string& object, const char* key, float* out_value)
    {
        if (!out_value)
            return false;
        const std::string needle = std::string("\"") + key + "\":";
        size_t pos = object.find(needle);
        if (pos == std::string::npos)
            return false;
        pos += needle.size();

        double value = 0.0;
        if (!canva_parse_json_number(object, &pos, &value) || !std::isfinite(value))
            return false;

        *out_value = (float)value;
        return true;
    }

    static bool canva_parse_json_number_field(const std::string& text, const char* key, double* out_value)
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
        return canva_parse_json_number(text, &pos, out_value);
    }

    static void canva_parse_sketch2rig_image_dimensions(const std::string& text, int* out_width, int* out_height)
    {
        if (!out_width || !out_height)
            return;

        double width = 0.0;
        double height = 0.0;
        if (canva_parse_json_number_field(text, "image_width", &width)
            && canva_parse_json_number_field(text, "image_height", &height))
        {
            *out_width = std::clamp((int)std::lround(width), 1, 32768);
            *out_height = std::clamp((int)std::lround(height), 1, 32768);
            return;
        }

        double image_size = 0.0;
        if (canva_parse_json_number_field(text, "image_size", &image_size))
        {
            const int size = std::clamp((int)std::lround(image_size), 1, 32768);
            *out_width = size;
            *out_height = size;
            return;
        }

        const std::string image_size_key = "\"image_size\"";
        const size_t image_size_pos = text.find(image_size_key);
        const size_t object_begin = image_size_pos == std::string::npos
            ? std::string::npos
            : text.find('{', image_size_pos + image_size_key.size());
        const size_t object_end = object_begin == std::string::npos
            ? std::string::npos
            : text.find('}', object_begin + 1);
        if (object_begin != std::string::npos && object_end != std::string::npos)
        {
            const std::string object = text.substr(object_begin, object_end - object_begin + 1);
            if (canva_parse_json_number_field(object, "width", &width)
                && canva_parse_json_number_field(object, "height", &height))
            {
                *out_width = std::clamp((int)std::lround(width), 1, 32768);
                *out_height = std::clamp((int)std::lround(height), 1, 32768);
            }
        }
    }

    static bool canva_parse_json_position_field(const std::string& object, float* out_x, float* out_y)
    {
        const std::string needle = "\"position\":";
        size_t pos = object.find(needle);
        if (pos == std::string::npos)
            return false;
        pos += needle.size();

        while (pos < object.size() && std::isspace((unsigned char)object[pos]))
            ++pos;
        if (pos >= object.size() || object[pos] != '[')
            return false;
        ++pos;

        double x = 0.0;
        double y = 0.0;
        if (!canva_parse_json_number(object, &pos, &x))
            return false;
        while (pos < object.size() && (std::isspace((unsigned char)object[pos]) || object[pos] == ','))
            ++pos;
        if (!canva_parse_json_number(object, &pos, &y))
            return false;

        *out_x = (float)x;
        *out_y = (float)y;
        return true;
    }

    static bool canva_load_sketch2rig_joints(
        const std::filesystem::path& json_path,
        std::vector<Sketch2RigJoint2d>* out_joints,
        int* out_image_width,
        int* out_image_height,
        std::string* out_error)
    {
        if (!out_joints)
            return false;
        out_joints->clear();
        if (out_image_width)
            *out_image_width = 0;
        if (out_image_height)
            *out_image_height = 0;

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
        canva_parse_sketch2rig_image_dimensions(text, out_image_width, out_image_height);

        const size_t joints_key = text.find("\"joints\"");
        if (joints_key == std::string::npos)
        {
            if (out_error)
                *out_error = "Missing joints array in " + json_path.string();
            return false;
        }

        size_t array_pos = text.find('[', joints_key);
        if (array_pos == std::string::npos)
        {
            if (out_error)
                *out_error = "Malformed joints array in " + json_path.string();
            return false;
        }

        std::vector<Sketch2RigJoint2d> parsed;
        size_t pos = array_pos + 1;
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
            Sketch2RigJoint2d joint;
            if (canva_parse_json_int_field(object, "index", &joint.index)
                && canva_parse_json_int_field(object, "parent", &joint.parent)
                && canva_parse_json_position_field(object, &joint.x, &joint.y))
            {
                joint.valid = true;
                parsed.push_back(joint);
            }

            pos = object_end + 1;
        }

        if (parsed.empty())
        {
            if (out_error)
                *out_error = "No valid joints in " + json_path.string();
            return false;
        }

        int max_index = -1;
        for (const auto& joint : parsed)
            max_index = std::max(max_index, joint.index);
        if (max_index < 0 || max_index > 4096)
        {
            if (out_error)
                *out_error = "Invalid joint indices in " + json_path.string();
            return false;
        }

        out_joints->assign((size_t)max_index + 1u, Sketch2RigJoint2d{});
        for (const auto& joint : parsed)
        {
            if (joint.index >= 0 && joint.index < (int)out_joints->size())
                (*out_joints)[(size_t)joint.index] = joint;
        }

        return true;
    }

    static void canva_project_sketch2rig_joints(
        std::vector<Sketch2RigJoint2d>& joints,
        int layer_w,
        int layer_h,
        int source_w,
        int source_h)
    {
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();

        for (const auto& joint : joints)
        {
            if (!joint.valid)
                continue;
            min_x = std::min(min_x, joint.x);
            min_y = std::min(min_y, joint.y);
            max_x = std::max(max_x, joint.x);
            max_y = std::max(max_y, joint.y);
        }

        const bool looks_normalized =
            min_x >= -0.25f && min_y >= -0.25f && max_x <= 1.25f && max_y <= 1.25f;
        const int raster_w = source_w > 1 ? source_w : layer_w;
        const int raster_h = source_h > 1 ? source_h : layer_h;

        if (looks_normalized)
        {
            for (auto& joint : joints)
            {
                if (!joint.valid)
                    continue;
                joint.x = std::clamp(joint.x, 0.0f, 1.0f) * (float)(raster_w - 1);
                joint.y = std::clamp(joint.y, 0.0f, 1.0f) * (float)(raster_h - 1);
            }
        }
        else if (source_w > 1 && source_h > 1)
        {
            for (auto& joint : joints)
            {
                if (!joint.valid)
                    continue;
                joint.x = std::clamp(joint.x, 0.0f, (float)(source_w - 1));
                joint.y = std::clamp(joint.y, 0.0f, (float)(source_h - 1));
            }
        }
        else
        {
            const float dx = std::max(1e-6f, max_x - min_x);
            const float dy = std::max(1e-6f, max_y - min_y);
            const float margin = 0.10f * (float)std::min(layer_w, layer_h);
            const float scale = std::min(((float)layer_w - 2.0f * margin) / dx, ((float)layer_h - 2.0f * margin) / dy);
            const float used_w = dx * scale;
            const float used_h = dy * scale;
            const float ox = ((float)layer_w - used_w) * 0.5f;
            const float oy = ((float)layer_h - used_h) * 0.5f;

            for (auto& joint : joints)
            {
                if (!joint.valid)
                    continue;
                joint.x = ox + (joint.x - min_x) * scale;
                joint.y = oy + (joint.y - min_y) * scale;
            }
        }

        if (source_w <= 1 || source_h <= 1 || (source_w == layer_w && source_h == layer_h))
            return;

        for (auto& joint : joints)
        {
            if (!joint.valid)
                continue;
            joint.x *= (float)(layer_w - 1) / (float)(source_w - 1);
            joint.y *= (float)(layer_h - 1) / (float)(source_h - 1);
        }
    }

     bool canva_add_sketch2rig_layer(
        CanvaState& canva,
        const std::filesystem::path& json_path,
        const std::filesystem::path& model_input_path,
        bool enabled,
        std::string* out_error)
    {
        std::vector<Sketch2RigJoint2d> joints;
        int image_width = 0;
        int image_height = 0;
        if (!canva_load_sketch2rig_joints(json_path, &joints, &image_width, &image_height, out_error))
            return false;

        canva_project_sketch2rig_joints(joints, canva.width, canva.height, image_width, image_height);

        CanvaPreprocessTransform preprocess_transform;
        std::string transform_error;
        const bool have_preprocess_transform =
            canva_load_preprocess_transform(model_input_path, &preprocess_transform, &transform_error);
        if (!have_preprocess_transform)
        {
            if (out_error)
                *out_error = transform_error.empty() ? "Missing preprocessing transform" : transform_error;
            return false;
        }

        CanvaLayer layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "Sketch2Rig rig");
        layer.enabled = enabled;

        const float inv_scale = 1.0f / std::max(1e-6f, preprocess_transform.scale);
        layer.scale = inv_scale;
        layer.offset = ImVec2(
            preprocess_transform.src_center.x - preprocess_transform.dst_center.x * inv_scale,
            preprocess_transform.src_center.y - preprocess_transform.dst_center.y * inv_scale);

        const ImVec4 bone_col = ImVec4(0.05f, 0.30f, 1.00f, 0.92f);
        const ImVec4 joint_col = ImVec4(1.00f, 0.20f, 0.10f, 0.96f);
        const ImVec4 root_col = ImVec4(0.10f, 0.80f, 0.35f, 0.98f);

        for (const auto& joint : joints)
        {
            if (!joint.valid || joint.parent < 0 || joint.parent >= (int)joints.size())
                continue;
            const auto& parent = joints[(size_t)joint.parent];
            if (!parent.valid)
                continue;
            canva_draw_segment(layer, canva.width, canva.height,
                ImVec2(parent.x, parent.y), ImVec2(joint.x, joint.y), 2.5f, bone_col);
        }

        for (const auto& joint : joints)
        {
            if (!joint.valid)
                continue;
            canva_stamp_circle(layer, canva.width, canva.height,
                joint.x, joint.y, joint.parent < 0 ? 6.5f : 5.0f, joint.parent < 0 ? root_col : joint_col);
        }

        canva.layers.insert(canva.layers.begin(), std::move(layer));
        canva.active_layer = 0;
        canva_cancel_interactions(canva);
        return true;
    }

    static bool canva_load_vitpose_keypoints(
        const std::filesystem::path& json_path,
        std::vector<VitPoseKeypoint2d>* out_keypoints,
        std::string* out_error)
    {
        if (!out_keypoints)
            return false;
        out_keypoints->clear();

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
        const size_t keypoints_key = text.find("\"keypoints\"");
        const size_t array_begin = keypoints_key == std::string::npos
            ? std::string::npos
            : text.find('[', keypoints_key);
        if (array_begin == std::string::npos)
        {
            if (out_error)
                *out_error = "Missing keypoints array in " + json_path.string();
            return false;
        }

        std::vector<std::pair<int, VitPoseKeypoint2d>> parsed;
        size_t pos = array_begin + 1;
        while (pos < text.size())
        {
            const size_t object_begin = text.find('{', pos);
            const size_t array_end = text.find(']', pos);
            if (array_end != std::string::npos
                && (object_begin == std::string::npos || array_end < object_begin))
                break;
            if (object_begin == std::string::npos)
                break;

            const size_t object_end = text.find('}', object_begin + 1);
            if (object_end == std::string::npos)
                break;

            const std::string object = text.substr(object_begin, object_end - object_begin + 1);
            int index = -1;
            VitPoseKeypoint2d keypoint;
            if (canva_parse_json_int_field(object, "index", &index)
                && canva_parse_json_position_field(object, &keypoint.x, &keypoint.y)
                && canva_parse_json_float_field(object, "score", &keypoint.score)
                && index >= 0 && index < 4096)
            {
                keypoint.valid = true;
                parsed.emplace_back(index, keypoint);
            }
            pos = object_end + 1;
        }

        if (parsed.empty())
        {
            if (out_error)
                *out_error = "No valid VitPose keypoints in " + json_path.string();
            return false;
        }

        int max_index = 0;
        for (const auto& item : parsed)
            max_index = std::max(max_index, item.first);
        out_keypoints->assign((size_t)max_index + 1u, VitPoseKeypoint2d{});
        for (const auto& item : parsed)
            (*out_keypoints)[(size_t)item.first] = item.second;
        return true;
    }

     bool canva_add_vitpose_layer(
        CanvaState& canva,
        const std::filesystem::path& json_path,
        const std::filesystem::path& model_input_path,
        std::string* out_error)
    {
        std::vector<VitPoseKeypoint2d> keypoints;
        if (!canva_load_vitpose_keypoints(json_path, &keypoints, out_error))
            return false;

        CanvaPreprocessTransform preprocess_transform;
        std::string transform_error;
        if (!canva_load_preprocess_transform(model_input_path, &preprocess_transform, &transform_error))
        {
            if (out_error)
                *out_error = transform_error.empty() ? "Missing preprocessing transform" : transform_error;
            return false;
        }

        constexpr float kScoreThreshold = 0.30f;
        int visible_keypoint_count = 0;
        for (const auto& keypoint : keypoints)
        {
            if (keypoint.valid && keypoint.score >= kScoreThreshold)
                visible_keypoint_count++;
        }
        if (visible_keypoint_count == 0)
        {
            if (out_error)
                *out_error = "VitPose returned no keypoints above the 0.30 confidence threshold";
            return false;
        }

        CanvaLayer layer = canva_make_layer(
            canva.width,
            canva.height,
            canva.next_layer_id++,
            "VitPose 2D pose");
        layer.kind = CanvaLayerKind::VitPose;

        const float inverse_scale = 1.0f / std::max(1e-6f, preprocess_transform.scale);
        layer.scale = inverse_scale;
        layer.offset = ImVec2(
            preprocess_transform.src_center.x - preprocess_transform.dst_center.x * inverse_scale,
            preprocess_transform.src_center.y - preprocess_transform.dst_center.y * inverse_scale);

        static constexpr int kCocoSkeleton[][2] = {
            { 15, 13 }, { 13, 11 }, { 16, 14 }, { 14, 12 }, { 11, 12 },
            { 5, 11 }, { 6, 12 }, { 5, 6 }, { 5, 7 }, { 6, 8 },
            { 7, 9 }, { 8, 10 }, { 1, 2 }, { 0, 1 }, { 0, 2 },
            { 1, 3 }, { 2, 4 }, { 3, 5 }, { 4, 6 }
        };
        const ImVec4 bone_color = ImVec4(0.10f, 0.72f, 1.00f, 0.92f);
        const ImVec4 joint_color = ImVec4(1.00f, 0.22f, 0.12f, 0.97f);

        for (const auto& edge : kCocoSkeleton)
        {
            if (edge[0] >= (int)keypoints.size() || edge[1] >= (int)keypoints.size())
                continue;
            const auto& a = keypoints[(size_t)edge[0]];
            const auto& b = keypoints[(size_t)edge[1]];
            if (!a.valid || !b.valid || a.score < kScoreThreshold || b.score < kScoreThreshold)
                continue;
            canva_draw_segment(
                layer,
                canva.width,
                canva.height,
                ImVec2(a.x, a.y),
                ImVec2(b.x, b.y),
                2.5f,
                bone_color);
        }

        for (const auto& keypoint : keypoints)
        {
            if (!keypoint.valid || keypoint.score < kScoreThreshold)
                continue;
            canva_stamp_circle(
                layer,
                canva.width,
                canva.height,
                keypoint.x,
                keypoint.y,
                5.0f,
                joint_color);
        }

        canva.layers.insert(canva.layers.begin(), std::move(layer));
        canva.active_layer = 0;
        canva_cancel_interactions(canva);
        return true;
    }

}
