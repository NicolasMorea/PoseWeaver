#include "ui/canva/CanvaPreprocess.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include <stb_image_write.h>

namespace poseweaver::ui_canva
{
    namespace
    {
        bool next_render_path(
            const char* tag_suffix,
            const char* extension,
            std::filesystem::path* out_path,
            std::string* out_error)
        {
            std::error_code ec;
            const std::filesystem::path renders_dir = std::filesystem::path("renders");
            std::filesystem::create_directories(renders_dir, ec);
            if (ec)
            {
                if (out_error)
                    *out_error = "Failed to create renders directory: " + ec.message();
                return false;
            }

            const std::string tag = (tag_suffix && tag_suffix[0] != '\0')
                ? (std::string("_") + tag_suffix)
                : std::string();

            const std::filesystem::path out = renders_dir / ("canva" + tag + extension);

            if (out.empty())
            {
                if (out_error)
                    *out_error = "Failed to allocate render output path";
                return false;
            }

            if (out_path)
                *out_path = out;
            return true;
        }

        std::string json_escape(const std::string& s)
        {
            std::string out;
            out.reserve(s.size() + 8);
            for (char c : s)
            {
                switch (c)
                {
                case '\\': out += "\\\\"; break;
                case '"': out += "\\\""; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out += c; break;
                }
            }
            return out;
        }

        bool save_filtered_bbox_debug_png(
            int width,
            int height,
            const std::vector<float>& ink,
            float step,
            int min_x,
            int min_y,
            int max_x,
            int max_y,
            const std::filesystem::path& model_input_path,
            std::string* out_error)
        {
            std::filesystem::path out = model_input_path.parent_path()
                / (model_input_path.stem().string() + "_filtered_bbox.png");
            if (out.empty())
            {
                if (!next_render_path("filtered_bbox", ".png", &out, out_error))
                    return false;
            }

            std::error_code ec;
            std::filesystem::create_directories(out.parent_path(), ec);
            if (ec)
            {
                if (out_error)
                    *out_error = "Failed to create preprocessing debug directory: " + ec.message();
                return false;
            }

            std::vector<unsigned char> debug((size_t)width * (size_t)height * 4u, 0u);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t gi = (size_t)y * (size_t)width + (size_t)x;
                    const size_t pi = gi * 4u;
                    const bool keep = ink[gi] > step;
                    const unsigned char value = keep
                        ? (unsigned char)std::clamp<int>((int)(std::clamp(ink[gi], 0.0f, 1.0f) * 255.0f + 0.5f), 0, 255)
                        : 0u;
                    debug[pi + 0u] = value;
                    debug[pi + 1u] = value;
                    debug[pi + 2u] = value;
                    debug[pi + 3u] = 255u;
                }
            }

            const auto set_red = [&](int x, int y)
            {
                if (x < 0 || y < 0 || x >= width || y >= height)
                    return;
                const size_t pi = ((size_t)y * (size_t)width + (size_t)x) * 4u;
                debug[pi + 0u] = 255u;
                debug[pi + 1u] = 32u;
                debug[pi + 2u] = 32u;
                debug[pi + 3u] = 255u;
            };

            constexpr int kThickness = 3;
            for (int t = 0; t < kThickness; ++t)
            {
                for (int x = min_x; x <= max_x; ++x)
                {
                    set_red(x, min_y + t);
                    set_red(x, max_y - t);
                }
                for (int y = min_y; y <= max_y; ++y)
                {
                    set_red(min_x + t, y);
                    set_red(max_x - t, y);
                }
            }

            const int ok = stbi_write_png(out.string().c_str(), width, height, 4, debug.data(), width * 4);
            if (ok == 0)
            {
                if (out_error)
                    *out_error = std::string("Failed to write PNG: ") + out.string();
                return false;
            }
            return true;
        }

        bool save_preprocess_metadata_json(
            const std::filesystem::path& image_path,
            int width,
            int height,
            int min_x,
            int min_y,
            int max_x,
            int max_y,
            float scale,
            float src_cx,
            float src_cy,
            float dst_cx,
            float dst_cy,
            std::string* out_error)
        {
            const std::filesystem::path meta_path = image_path.string() + ".preprocess.json";
            std::ofstream out(meta_path);
            if (!out)
            {
                if (out_error)
                    *out_error = "Failed to write preprocessing metadata: " + meta_path.string();
                return false;
            }

            out << "{\n"
                << "  \"schema\": \"poseweaver.canva_preprocess.v1\",\n"
                << "  \"model_input\": \"" << json_escape(image_path.string()) << "\",\n"
                << "  \"width\": " << width << ",\n"
                << "  \"height\": " << height << ",\n"
                << "  \"bbox\": [" << min_x << ", " << min_y << ", " << max_x << ", " << max_y << "],\n"
                << "  \"scale\": " << scale << ",\n"
                << "  \"src_center\": [" << src_cx << ", " << src_cy << "],\n"
                << "  \"dst_center\": [" << dst_cx << ", " << dst_cy << "],\n"
                << "  \"inverse\": {\n"
                << "    \"x\": \"src_center_x + (x - dst_center_x) / scale\",\n"
                << "    \"y\": \"src_center_y + (y - dst_center_y) / scale\"\n"
                << "  }\n"
                << "}\n";

            return true;
        }

        float sample_gray_bilinear(
            const std::vector<float>& gray,
            int width,
            int height,
            float x,
            float y)
        {
            if (gray.empty() || width <= 0 || height <= 0)
                return 0.0f;
            if (x < 0.0f || y < 0.0f || x > (float)(width - 1) || y > (float)(height - 1))
                return 0.0f;

            const int x0 = (int)std::floor(x);
            const int y0 = (int)std::floor(y);
            const int x1 = std::min(x0 + 1, width - 1);
            const int y1 = std::min(y0 + 1, height - 1);
            const float tx = x - (float)x0;
            const float ty = y - (float)y0;

            const float c00 = gray[(size_t)y0 * (size_t)width + (size_t)x0];
            const float c10 = gray[(size_t)y0 * (size_t)width + (size_t)x1];
            const float c01 = gray[(size_t)y1 * (size_t)width + (size_t)x0];
            const float c11 = gray[(size_t)y1 * (size_t)width + (size_t)x1];
            const float c0 = c00 + (c10 - c00) * tx;
            const float c1 = c01 + (c11 - c01) * tx;
            return c0 + (c1 - c0) * ty;
        }
    }

    bool save_normalized_model_input_png(
        int width,
        int height,
        const std::vector<unsigned char>& rgba,
        std::filesystem::path* out_path,
        std::string* out_error)
    {
        std::filesystem::path out;
        if (!next_render_path("model_input", ".png", &out, out_error))
            return false;
        return save_normalized_model_input_png_to_path(
            width,
            height,
            rgba,
            out,
            out_path,
            out_error);
    }

    bool save_normalized_model_input_png_to_path(
        int width,
        int height,
        const std::vector<unsigned char>& rgba,
        const std::filesystem::path& output_path,
        std::filesystem::path* out_path,
        std::string* out_error)
    {
        if (width <= 0 || height <= 0)
        {
            if (out_error)
                *out_error = "Invalid canva size";
            return false;
        }
        if (rgba.size() < (size_t)width * (size_t)height * 4u)
        {
            if (out_error)
                *out_error = "Invalid RGBA buffer";
            return false;
        }

        constexpr float kTargetStrokeMean = 0.68f;
        constexpr float kTargetStrokeDarkness = 1.0f - kTargetStrokeMean;
        constexpr float kMaxStrokeDarkness = 0.8f;

        std::vector<float> ink((size_t)width * (size_t)height, 0.0f);
        float max_ink = 0.0f;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t p = ((size_t)y * (size_t)width + (size_t)x) * 4u;
                const float r = (float)rgba[p + 0u] / 255.0f;
                const float g = (float)rgba[p + 1u] / 255.0f;
                const float b = (float)rgba[p + 2u] / 255.0f;
                const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                const size_t gi = (size_t)y * (size_t)width + (size_t)x;
                ink[gi] = std::clamp(1.0f - lum, 0.0f, 1.0f);
                max_ink = std::max(max_ink, ink[gi]);
            }
        }

        if (max_ink <= 1e-6f)
        {
            if (out_error)
                *out_error = "No visible sketch pixels found";
            return false;
        }

        const float step = max_ink / 3.0f;
        int min_x = width;
        int min_y = height;
        int max_x = -1;
        int max_y = -1;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (ink[(size_t)y * (size_t)width + (size_t)x] <= step)
                    continue;
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }

        if (max_x < min_x || max_y < min_y)
        {
            if (out_error)
                *out_error = "No sketch pixels survived preprocessing threshold";
            return false;
        }

        const float bbox_w = (float)(max_x - min_x + 1);
        const float bbox_h = (float)(max_y - min_y + 1);
        const float desired_max = 0.70f * (float)std::min(width, height);
        const float scale = desired_max / std::max(1.0f, std::max(bbox_w, bbox_h));
        const float src_cx = ((float)min_x + (float)max_x) * 0.5f;
        const float src_cy = ((float)min_y + (float)max_y) * 0.5f;
        const float dst_cx = ((float)width - 1.0f) * 0.5f;
        const float dst_cy = ((float)height - 1.0f) * 0.5f;

        std::vector<float> centered((size_t)width * (size_t)height, 0.0f);
        float centered_max = 0.0f;
        double centered_stroke_sum = 0.0;
        int centered_stroke_count = 0;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const float src_x = src_cx + ((float)x - dst_cx) / scale;
                const float src_y = src_cy + ((float)y - dst_cy) / scale;
                const float value = (src_x >= 0.0f && src_y >= 0.0f
                    && src_x <= (float)(width - 1) && src_y <= (float)(height - 1))
                    ? sample_gray_bilinear(ink, width, height, src_x, src_y)
                    : 0.0f;
                centered[(size_t)y * (size_t)width + (size_t)x] = value;
                centered_max = std::max(centered_max, value);
                if (value > step)
                {
                    centered_stroke_sum += (double)value;
                    centered_stroke_count++;
                }
            }
        }

        if (centered_max <= 1e-6f || centered_stroke_count <= 0)
        {
            if (out_error)
                *out_error = "Preprocessed sketch has no stroke pixels";
            return false;
        }

        const std::vector<float> base_centered = centered;
        const auto clipped_stroke_darkness_mean = [&](float scale_value) -> float
        {
            double sum = 0.0;
            int count = 0;
            for (float value : base_centered)
            {
                if (value <= step)
                    continue;
                sum += (double)std::clamp(value * scale_value, 0.0f, kMaxStrokeDarkness);
                count++;
            }
            return (count > 0) ? (float)(sum / (double)count) : 0.0f;
        };

        float lo = 0.0f;
        float hi = kTargetStrokeDarkness / std::max(1e-6f, (float)(centered_stroke_sum / (double)centered_stroke_count));
        while (clipped_stroke_darkness_mean(hi) < kTargetStrokeDarkness && hi < 1024.0f)
            hi *= 2.0f;

        for (int iter = 0; iter < 32; ++iter)
        {
            const float mid = (lo + hi) * 0.5f;
            if (clipped_stroke_darkness_mean(mid) < kTargetStrokeDarkness)
                lo = mid;
            else
                hi = mid;
        }

        const float output_scale = hi;
        std::vector<unsigned char> out_gray((size_t)width * (size_t)height, 0u);
        for (size_t i = 0; i < base_centered.size(); ++i)
        {
            const float value = (base_centered[i] > step)
                ? (1.0f - std::clamp(base_centered[i] * output_scale, 0.0f, kMaxStrokeDarkness))
                : 1.0f;
            out_gray[i] = (unsigned char)std::clamp<int>((int)(value * 255.0f + 0.5f), 0, 255);
        }

        std::filesystem::path out = output_path;
        if (out.empty())
        {
            if (out_error)
                *out_error = "Failed to allocate model input path";
            return false;
        }

        std::error_code ec;
        std::filesystem::create_directories(out.parent_path(), ec);
        if (ec)
        {
            if (out_error)
                *out_error = "Failed to create model input directory: " + ec.message();
            return false;
        }

        if (!save_filtered_bbox_debug_png(width, height, ink, step, min_x, min_y, max_x, max_y, out, out_error))
            return false;

        const int ok = stbi_write_png(out.string().c_str(), width, height, 1, out_gray.data(), width);
        if (ok == 0)
        {
            if (out_error)
                *out_error = std::string("Failed to write PNG: ") + out.string();
            return false;
        }

        if (!save_preprocess_metadata_json(
                out,
                width,
                height,
                min_x,
                min_y,
                max_x,
                max_y,
                scale,
                src_cx,
                src_cy,
                dst_cx,
                dst_cy,
                out_error))
            return false;

        if (out_path)
            *out_path = out;
        return true;
    }
}
