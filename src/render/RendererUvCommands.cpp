#include "render/Renderer.h"
#include "render/RendererInternal.h"

#include <glad/glad.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cnpy.h>
#include <stb_image.h>

namespace poseweaver
{
    namespace
    {
        static std::string read_text_file_uv(const std::filesystem::path& path)
        {
            std::ifstream f(path, std::ios::binary);
            if (!f)
                return {};
            std::ostringstream ss;
            ss << f.rdbuf();
            return ss.str();
        }

        static std::string parse_json_string_field_uv(const std::string& text, const char* key)
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

        template <typename T>
        static float npy_value_as_float_uv(const cnpy::NpyArray& a, size_t idx)
        {
            return (float)a.data<T>()[idx];
        }

        static bool npy_float_at_uv(const cnpy::NpyArray& a, size_t idx, float* out)
        {
            if (!out)
                return false;
            if (a.word_size == sizeof(float))
            {
                *out = npy_value_as_float_uv<float>(a, idx);
                return true;
            }
            if (a.word_size == sizeof(double))
            {
                *out = npy_value_as_float_uv<double>(a, idx);
                return true;
            }
            return false;
        }

        static std::vector<unsigned char> make_uv_background_rgba8(int w, int h)
        {
            std::vector<unsigned char> px;
            if (w <= 0 || h <= 0)
                return px;

            px.resize((size_t)w * (size_t)h * 4u);
            for (int y = 0; y < h; ++y)
            {
                const float v = (h > 1) ? (float)(h - 1 - y) / (float)(h - 1) : 0.0f; // v increases upward
                for (int x = 0; x < w; ++x)
                {
                    const float u = (w > 1) ? (float)x / (float)(w - 1) : 0.0f;

                    unsigned char r = (unsigned char)std::clamp<int>((int)(u * 255.0f + 0.5f), 0, 255);
                    unsigned char g = (unsigned char)std::clamp<int>((int)(v * 255.0f + 0.5f), 0, 255);
                    unsigned char b = render_internal::kUvBlueByte;
                    unsigned char a = 255;

                    // Subtle UV grid lines (every 0.1)
                    const int gx = (w >= 10) ? (w / 10) : w;
                    const int gy = (h >= 10) ? (h / 10) : h;
                    if ((gx > 0 && (x % gx) == 0) || (gy > 0 && (y % gy) == 0))
                    {
                        r = (unsigned char)(r * 3 / 5);
                        g = (unsigned char)(g * 3 / 5);
                        b = (unsigned char)(b * 3 / 5);
                    }

                    const size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 4u;
                    px[idx + 0] = r;
                    px[idx + 1] = g;
                    px[idx + 2] = b;
                    px[idx + 3] = a;
                }
            }
            return px;
        }
    }

    void Renderer::generate_uv_debug_image(AppState& state)
    {
        console_add_line(state, "Generating UV debug image...");

        // Generate a square UV debug background.
        constexpr int kTexSize = 1024;
        const int w = kTexSize;
        const int h = kTexSize;

        const auto pixels = make_uv_background_rgba8(w, h);
        if (pixels.empty())
            return;

        if (m_uv_debug_texture_id == 0)
            glGenTextures(1, &m_uv_debug_texture_id);

        glBindTexture(GL_TEXTURE_2D, m_uv_debug_texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // (Re)allocate each time for simplicity.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        m_uv_debug_texture_w = w;
        m_uv_debug_texture_h = h;

        // Build/refresh UV wireframe segments.
        regenerate_uv_wireframe_segments();
        console_add_line(state,
            std::string("UV debug image ready: ") + std::to_string(w) + "x" + std::to_string(h) +
            " (segments=" + std::to_string((int)m_smpl_uv_segments.size()) + ")");

        // UV view assets refreshed -> recompute garment UV overlay if present.
        m_garment_uv_overlay_valid = false;
    }

    void Renderer::ensure_sketch2uv_prediction_texture(AppState& state)
    {
        const std::string desired_json = state.sketch2uv.last_output_path;
        const std::uint64_t desired_request = state.sketch2uv.completed_request_id;
        if (desired_json.empty())
            return;
        if (desired_json == m_sketch2uv_prediction_json_path
            && desired_request == m_sketch2uv_prediction_request_id)
            return;

        m_sketch2uv_prediction_json_path = desired_json;
        m_sketch2uv_prediction_request_id = desired_request;
        m_sketch2uv_prediction_pixels.clear();
        m_sketch2uv_prediction_uv.clear();
        m_sketch2uv_prediction_mask.clear();
        if (m_sketch2uv_prediction_texture_id != 0)
        {
            glDeleteTextures(1, &m_sketch2uv_prediction_texture_id);
            m_sketch2uv_prediction_texture_id = 0;
        }
        m_sketch2uv_prediction_texture_w = 0;
        m_sketch2uv_prediction_texture_h = 0;

        const std::filesystem::path json_path = desired_json;
        const std::string json_text = read_text_file_uv(json_path);
        if (json_text.empty())
        {
            console_add_line(state, "Sketch2UV preview failed: could not read " + json_path.string());
            return;
        }

        std::filesystem::path render_path = parse_json_string_field_uv(json_text, "render_path");
        if (render_path.empty())
            render_path = json_path.parent_path() / (json_path.stem().string() + "_render.png");
        if (render_path.is_relative())
            render_path = json_path.parent_path() / render_path;

        int w = 0;
        int h = 0;
        int comp = 0;
        stbi_uc* data = stbi_load(render_path.string().c_str(), &w, &h, &comp, 4);
        if (!data || w <= 0 || h <= 0)
        {
            if (data)
                stbi_image_free(data);
            console_add_line(state, "Sketch2UV preview failed: could not load " + render_path.string());
            return;
        }

        m_sketch2uv_prediction_pixels.assign(data, data + (size_t)w * (size_t)h * 4u);
        stbi_image_free(data);

        if (m_sketch2uv_prediction_texture_id == 0)
            glGenTextures(1, &m_sketch2uv_prediction_texture_id);

        glBindTexture(GL_TEXTURE_2D, m_sketch2uv_prediction_texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_sketch2uv_prediction_pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        m_sketch2uv_prediction_texture_w = w;
        m_sketch2uv_prediction_texture_h = h;

        std::filesystem::path uv_npz_path = parse_json_string_field_uv(json_text, "uv_npz_path");
        if (uv_npz_path.empty())
            uv_npz_path = json_path.parent_path() / (json_path.stem().string() + "_uv.npz");
        if (uv_npz_path.is_relative())
            uv_npz_path = json_path.parent_path() / uv_npz_path;

        try
        {
            cnpy::NpyArray uv_arr = cnpy::npz_load(uv_npz_path.string(), "uv");
            cnpy::NpyArray mask_arr = cnpy::npz_load(uv_npz_path.string(), "mask");
            if (uv_arr.shape.size() == 3
                && uv_arr.shape[0] >= 2
                && mask_arr.shape.size() == 2
                && (int)uv_arr.shape[1] == h
                && (int)uv_arr.shape[2] == w
                && (int)mask_arr.shape[0] == h
                && (int)mask_arr.shape[1] == w
                && (uv_arr.word_size == sizeof(float) || uv_arr.word_size == sizeof(double))
                && (mask_arr.word_size == sizeof(float) || mask_arr.word_size == sizeof(double)))
            {
                const size_t count = (size_t)w * (size_t)h;
                m_sketch2uv_prediction_uv.resize(count * 2u, 0.0f);
                m_sketch2uv_prediction_mask.resize(count, 0.0f);
                for (size_t i = 0; i < count; ++i)
                {
                    float u = 0.0f;
                    float v = 0.0f;
                    float mask = 0.0f;
                    if (npy_float_at_uv(uv_arr, i, &u)
                        && npy_float_at_uv(uv_arr, count + i, &v)
                        && npy_float_at_uv(mask_arr, i, &mask))
                    {
                        m_sketch2uv_prediction_uv[i] = u;
                        m_sketch2uv_prediction_uv[count + i] = v;
                        m_sketch2uv_prediction_mask[i] = mask;
                    }
                }
            }
        }
        catch (const std::exception&)
        {
            m_sketch2uv_prediction_uv.clear();
            m_sketch2uv_prediction_mask.clear();
        }

        if (m_uv_debug_texture_id == 0 || m_uv_debug_texture_w <= 0 || m_uv_debug_texture_h <= 0)
            generate_uv_debug_image(state);

        console_add_line(state, "Sketch2UV preview ready: " + render_path.string());
    }
}
