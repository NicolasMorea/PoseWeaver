#include "render/Renderer.h"
#include "core/Paths.h"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
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
        constexpr float kSketch2RigVerticalFovDeg =
            kSketch2RigVerticalFovRad * 180.0f / 3.14159265358979323846f;
        constexpr int kIsoDimensions = 64;

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

        static int build_red_deformation_warp(
            const std::vector<unsigned char>& deformation_rgba,
            const std::vector<unsigned char>& body_mask,
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
            if (deformation_rgba.size() < count * 4u || body_mask.size() < count || force <= 0.0f)
                return 0;

            std::vector<unsigned char> red_mask(count, 0u);
            int red_count = 0;
            for (size_t i = 0; i < count; ++i)
            {
                const size_t p = i * 4u;
                // The deformation-map red is deliberately detected narrowly so
                // blue and magenta cases cannot become warp sources.
                if (deformation_rgba[p + 3u] != 0u
                    && deformation_rgba[p + 0u] >= 240u
                    && deformation_rgba[p + 1u] <= 96u
                    && deformation_rgba[p + 2u] <= 80u)
                {
                    red_mask[i] = 1u;
                    ++red_count;
                }
            }
            if (red_count == 0)
                return 0;

            std::vector<int> body_x;
            std::vector<int> body_y;
            if (!nearest_mask_pixels(body_mask, width, height, body_x, body_y))
                return 0;

            std::vector<float> red_weight(count, 0.0f);
            std::vector<float> red_dx(count, 0.0f);
            std::vector<float> red_dy(count, 0.0f);
            for (size_t i = 0; i < count; ++i)
            {
                if (red_mask[i] == 0u)
                    continue;
                const int x = (int)(i % (size_t)width);
                const int y = (int)(i / (size_t)width);
                red_weight[i] = 1.0f;
                red_dx[i] = (float)(body_x[i] - x);
                red_dy[i] = (float)(body_y[i] - y);
            }

            std::vector<int> red_x;
            std::vector<int> red_y;
            if (!nearest_mask_pixels(red_mask, width, height, red_x, red_y))
                return 0;

            const float radius = std::max(0.0f, smooth_radius);
            if (radius > 0.0f)
            {
                gaussian_like_blur_scalar(red_weight, width, height, radius);
                gaussian_like_blur_scalar(red_dx, width, height, radius);
                gaussian_like_blur_scalar(red_dy, width, height, radius);
            }
            const float radius_squared = radius * radius;
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t i = (size_t)y * (size_t)width + (size_t)x;
                    const int sx = red_x[i];
                    const int sy = red_y[i];
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
                    const float density = red_weight[i];
                    if (!(density > 1e-6f))
                        continue;

                    // Normalize the blurred vectors by blurred seed density.
                    // Adjacent red pixels therefore act as one continuous
                    // liquify stroke instead of competing nearest-pixel sites.
                    out_dx[i] = (red_dx[i] / density) * force * influence;
                    out_dy[i] = (red_dy[i] / density) * force * influence;
                }
            }
            return red_count;
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

        static void warp_rgba_inverse_smooth(
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

                const int x0 = (int)std::floor(x);
                const int y0 = (int)std::floor(y);
                const int x1 = std::min(x0 + 1, width - 1);
                const int y1 = std::min(y0 + 1, height - 1);
                const float fx = x - (float)x0;
                const float fy = y - (float)y0;
                const std::array<size_t, 4> samples = {
                    (size_t)y0 * (size_t)width + (size_t)x0,
                    (size_t)y0 * (size_t)width + (size_t)x1,
                    (size_t)y1 * (size_t)width + (size_t)x0,
                    (size_t)y1 * (size_t)width + (size_t)x1
                };
                const std::array<float, 4> weights = {
                    (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy),
                    (1.0f - fx) * fy, fx * fy
                };
                float alpha = 0.0f;
                std::array<float, 3> premultiplied{ 0.0f, 0.0f, 0.0f };
                for (size_t sample = 0; sample < samples.size(); ++sample)
                {
                    const size_t source_pixel = samples[sample];
                    const float sample_alpha = (float)source[source_pixel * 4u + 3u] / 255.0f;
                    alpha += weights[sample] * sample_alpha;
                    for (size_t c = 0; c < 3u; ++c)
                        premultiplied[c] += weights[sample] * sample_alpha * (float)source[source_pixel * 4u + c];
                }
                out[target * 4u + 3u] = (unsigned char)std::clamp((int)std::lround(alpha * 255.0f), 0, 255);
                if (alpha > 1e-6f)
                {
                    for (size_t c = 0; c < 3u; ++c)
                        out[target * 4u + c] = (unsigned char)std::clamp(
                            (int)std::lround(premultiplied[c] / alpha), 0, 255);
                }
            }
        }

        static void warp_rgba_inverse_categorical(
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
                const int x = (int)std::lround(source_x[target]);
                const int y = (int)std::lround(source_y[target]);
                if (x < 0 || y < 0 || x >= width || y >= height)
                    continue;
                const size_t source_pixel = (size_t)y * (size_t)width + (size_t)x;
                for (size_t c = 0; c < 4u; ++c)
                    out[target * 4u + c] = source[source_pixel * 4u + c];
            }
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
            const float f_h = std::tan(kSketch2RigVerticalFovDeg / 360.0f * kPi) * dnear;
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
            std::vector<float>& body_to_cloth)
        {
            std::vector<double> cloth_squared(
                (size_t)cloth_vertices.rows(),
                std::numeric_limits<double>::infinity());
            std::vector<double> body_squared(
                (size_t)body_vertices.rows(),
                std::numeric_limits<double>::infinity());
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
                    body_squared[(size_t)body_vertex] =
                        std::min(body_squared[(size_t)body_vertex], squared);
                }
            }

            cloth_to_body.resize(cloth_squared.size());
            for (size_t i = 0; i < cloth_squared.size(); ++i)
                cloth_to_body[i] = (float)std::sqrt(cloth_squared[i]);
            body_to_cloth.resize(body_squared.size());
            for (size_t i = 0; i < body_squared.size(); ++i)
                body_to_cloth[i] = (float)std::sqrt(body_squared[i]);
        }

        static void rasterize_thresholded_vertex_distances(
            const Eigen::MatrixXd& vertices,
            const Eigen::MatrixXi& faces,
            const std::vector<float>& vertex_distances,
            const Eigen::Matrix4f& view,
            const Eigen::Matrix4f& proj,
            int width,
            int height,
            float threshold,
            std::vector<unsigned char>& mask)
        {
            mask.assign((size_t)width * (size_t)height, 0u);
            if (width <= 0 || height <= 0 || vertices.rows() <= 0 || faces.rows() <= 0
                || vertex_distances.size() < (size_t)vertices.rows())
                return;

            for (int fi = 0; fi < faces.rows(); ++fi)
            {
                const int i0 = faces(fi, 0);
                const int i1 = faces(fi, 1);
                const int i2 = faces(fi, 2);
                if (i0 < 0 || i1 < 0 || i2 < 0
                    || i0 >= vertices.rows() || i1 >= vertices.rows() || i2 >= vertices.rows())
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

                        const float distance =
                            w0 * vertex_distances[(size_t)i0]
                            + w1 * vertex_distances[(size_t)i1]
                            + w2 * vertex_distances[(size_t)i2];
                        if (distance < threshold)
                            mask[(size_t)py * (size_t)width + (size_t)px] = 1u;
                    }
                }
            }
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
        float prediction_from_cloth_scale,
        float prediction_from_cloth_offset_x,
        float prediction_from_cloth_offset_y,
        std::vector<unsigned char>& out_rgba,
        std::vector<unsigned char>& out_warped_rgba,
        std::vector<unsigned char>& out_debug_rgba,
        std::vector<unsigned char>& out_warped_debug_rgba,
        std::vector<unsigned char>& out_occlusion_rgba,
        std::vector<unsigned char>& out_scene_occlusion_rgba,
        std::vector<unsigned char>& out_predicted_depth_occlusion_rgba,
        std::vector<unsigned char>& out_aligned_predicted_depth_occlusion_rgba,
        std::vector<unsigned char>& out_scene_debug_rgba,
        std::vector<unsigned char>& out_deformation_rgba,
        std::vector<unsigned char>& out_close_to_body_rgba,
        int& out_tested_count,
        int& out_removed_count,
        std::string* out_error)
    {
        out_rgba.clear();
        out_warped_rgba.clear();
        out_debug_rgba.clear();
        out_warped_debug_rgba.clear();
        out_occlusion_rgba.clear();
        out_scene_occlusion_rgba.clear();
        out_predicted_depth_occlusion_rgba.clear();
        out_aligned_predicted_depth_occlusion_rgba.clear();
        out_scene_debug_rgba.clear();
        out_deformation_rgba.clear();
        out_close_to_body_rgba.clear();
        out_tested_count = 0;
        out_removed_count = 0;

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
        if (!state.scene_view_target_valid)
            return fail("predicted camera view is not available");

        ensure_smplh_mesh_loaded(state);
        ensure_garment_mesh_loaded(state);
        handle_garment_posing(state, false);

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
        if (!make_target_camera_matrices(state, core_ref.camera_dnear, core_ref.camera_dfar, view, proj))
            return fail("could not build predicted camera matrices");

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

        out_rgba = cloth_rgba;
        out_debug_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_occlusion_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_scene_occlusion_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_predicted_depth_occlusion_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_aligned_predicted_depth_occlusion_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_scene_debug_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_deformation_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
        out_close_to_body_rgba.assign((size_t)width * (size_t)height * 4u, 0u);
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
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float predicted_distance = 0.0f;
                if (predicted_body_distance_at(x, y, predicted_distance))
                    predicted_stats.add(predicted_distance);

                const size_t depth_idx =
                    (size_t)(height - 1 - y) * (size_t)width + (size_t)x;
                if (smpl_depth_gl[depth_idx] < 0.999999f)
                    body_model_stats.add(camera_distance_from_depth01(proj, smpl_depth_gl[depth_idx]));
            }
        }

        const float predicted_mean = predicted_stats.mean();
        const float predicted_stddev = predicted_stats.standard_deviation();
        const float body_model_mean = body_model_stats.mean();
        const float body_model_stddev = body_model_stats.standard_deviation();
        const bool have_alignment_stats = predicted_stats.count > 0 && body_model_stats.count > 0;
        const float aligned_depth_scale = have_alignment_stats && predicted_stddev > 1e-6f
            ? body_model_stddev / predicted_stddev
            : 1.0f;
        const float aligned_depth_translation = have_alignment_stats
            ? body_model_mean - aligned_depth_scale * predicted_mean
            : 0.0f;

        auto set_visibility_pixel = [](std::vector<unsigned char>& rgba, size_t pixel, bool occluded)
        {
            unsigned char* dst = rgba.data() + pixel * 4u;
            dst[0] = occluded ? 255u : 32u;
            dst[1] = occluded ? 48u : 220u;
            dst[2] = occluded ? 48u : 80u;
            dst[3] = 220u;
        };

        // Keep the depth tests separate here. They are combined with the ISO
        // result only when the final visibility map is built below.
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t out_pix = (size_t)y * (size_t)width + (size_t)x;
                const size_t depth_idx =
                    (size_t)(height - 1 - y) * (size_t)width + (size_t)x;
                const float cloth_depth = cloth_depth_gl[depth_idx];
                if (!(cloth_depth < 0.999999f))
                    continue;

                const float cloth_distance = camera_distance_from_depth01(proj, cloth_depth);
                const bool true_depth_occluded =
                    smpl_depth_gl[depth_idx] < 0.999999f
                    && camera_distance_from_depth01(proj, smpl_depth_gl[depth_idx])
                        + depth_margin < cloth_distance;
                set_visibility_pixel(out_scene_occlusion_rgba, out_pix, true_depth_occluded);

                float predicted_distance = 0.0f;
                const bool have_predicted_depth = predicted_body_distance_at(x, y, predicted_distance);
                const bool predicted_depth_occluded =
                    have_predicted_depth && predicted_distance + depth_margin_pred < cloth_distance;
                set_visibility_pixel(
                    out_predicted_depth_occlusion_rgba, out_pix, predicted_depth_occluded);

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

        console_add_line(
            state,
            "Cloth fitting predicted-depth alignment: scale=" + std::to_string(aligned_depth_scale)
                + ", translation=" + std::to_string(aligned_depth_translation)
                + ", predicted(mean/std)=" + std::to_string(predicted_mean)
                + "/" + std::to_string(predicted_stddev)
                + ", body(mean/std)=" + std::to_string(body_model_mean)
                + "/" + std::to_string(body_model_stddev));

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
        nearest_vertex_distances(
            garment.V,
            smpl.V,
            cloth_body_vertex_distances,
            body_cloth_vertex_distances);
        std::vector<unsigned char> close_to_body_mask;
        rasterize_thresholded_vertex_distances(
            garment.V,
            garment.F,
            cloth_body_vertex_distances,
            view,
            proj,
            width,
            height,
            body_distance_threshold,
            close_to_body_mask);

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
        // Cloth fitting debug colors:
        // grey=no ISO, blue=ISO miss, cyan=projection miss,
        // black=ISO pixel outside source cloth with no cloth over match,
        // dark grey=ISO pixel outside source cloth with cloth over match,
        // red=no cloth, orange=matching SMPL occludes cloth,
        // crimson=other SMPL occludes cloth but source cloth remains visible,
        // green=cloth on match, yellow=cloth over other SMPL, magenta=two cloth layers,
        // purple=two cloth layers over another SMPL layer.
        const float iso_distance_threshold = std::max(0.0f, state.iso_distance_threshold);
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
                const bool source_cloth_pixel = out_rgba[out_pix * 4u + 3u] != 0u;

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
                        set_debug_pixel(out_pix, 128u, 128u, 128u);
                        if (prediction_x_valid && prediction_y_valid)
                        {
                            const int prediction_x = (int)std::floor(prediction_layer_x);
                            const int prediction_y = (int)std::floor(prediction_layer_y);
                            const size_t prediction_pix =
                                (size_t)prediction_y * (size_t)width + (size_t)prediction_x;
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
                            if (prediction_pix < close_to_body_mask.size()
                                && close_to_body_mask[prediction_pix] != 0u)
                            {
                                unsigned char* dst = out_close_to_body_rgba.data() + out_pix * 4u;
                                dst[0] = 0u;
                                dst[1] = 220u;
                                dst[2] = 255u;
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
                // close to the ISO-matched body vertex.
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
                {
                    set_debug_pixel(out_pix, 0u, 96u, 255u);
                    continue;
                }

                int vertex_idx = -1;
                float iso_distance_squared = 0.0f;
                if (!find_nearest_iso_vertex(
                        iso_source.descriptors.data() + descriptor_offset,
                        vertex_idx,
                        iso_distance_squared)
                    || vertex_idx < 0
                    || vertex_idx >= m_smplh_vertices.rows()
                    || !std::isfinite(iso_distance_squared)
                    || std::sqrt(std::max(0.0f, iso_distance_squared)) > iso_distance_threshold)
                {
                    set_debug_pixel(out_pix, 0u, 96u, 255u);
                    continue;
                }

                if (!source_cloth_pixel
                    && vertex_idx < (int)body_cloth_vertex_distances.size()
                    && body_cloth_vertex_distances[(size_t)vertex_idx] < body_distance_threshold)
                {
                    unsigned char* dst = out_deformation_rgba.data() + out_pix * 4u;
                    dst[0] = 220u;
                    dst[1] = 64u;
                    dst[2] = 255u;
                    dst[3] = 220u;
                }

                const Eigen::Vector3d p = m_smplh_vertices.row(vertex_idx).transpose();

                float px = 0.0f;
                float py = 0.0f;
                float body_depth = 1.0f;
                if (!project_to_image_depth(view, proj, p, width, height, px, py, body_depth))
                {
                    set_debug_pixel(out_pix, 0u, 200u, 220u);
                    continue;
                }

                const int sx = (int)std::floor(px);
                const int sy = (int)std::floor(py);
                if (sx < 0 || sy < 0 || sx >= width || sy >= height)
                {
                    set_debug_pixel(out_pix, 0u, 200u, 220u);
                    continue;
                }

                if (source_cloth_pixel)
                    ++out_tested_count;
                const size_t depth_idx = (size_t)(height - 1 - sy) * (size_t)width + (size_t)sx;
                const float cloth_depth = cloth_depth_gl[depth_idx];
                const float cloth_second_depth = cloth_second_depth_cpu.size() == (size_t)width * (size_t)height
                    ? cloth_second_depth_cpu[depth_idx]
                    : 1.0f;
                const bool cloth_present_at_body = cloth_depth < 0.999999f;
                const bool cloth_on_top_of_match =
                    cloth_present_at_body
                    && cloth_depth + depth_margin < body_depth;
                const float nearest_smpl_depth = smpl_depth_gl[depth_idx] < 0.999999f
                    ? smpl_depth_gl[depth_idx]
                    : body_depth;
                const bool other_smpl_layer_on_top = nearest_smpl_depth + depth_margin < body_depth;
                const bool two_cloth_layers_on_top =
                    cloth_second_depth < 0.999999f
                    && cloth_second_depth + depth_margin < body_depth;
                const bool smpl_visible_on_top = !cloth_present_at_body
                    || nearest_smpl_depth + depth_margin < cloth_depth;
                if (!source_cloth_pixel)
                {
                    if (cloth_on_top_of_match)
                        set_debug_pixel(out_pix, 48u, 48u, 48u);
                    else
                        set_debug_pixel(out_pix, 0u, 0u, 0u);
                }
                else if (smpl_visible_on_top)
                {
                    if (other_smpl_layer_on_top)
                    {
                        set_debug_pixel(out_pix, 190u, 0u, 32u);
                    }
                    else
                    {
                        if (!cloth_present_at_body)
                            set_debug_pixel(out_pix, 255u, 32u, 32u);
                        else
                            set_debug_pixel(out_pix, 255u, 144u, 0u);
                    }
                }
                else if (other_smpl_layer_on_top && two_cloth_layers_on_top)
                {
                    set_debug_pixel(out_pix, 160u, 64u, 255u);
                }
                else if (other_smpl_layer_on_top)
                {
                    set_debug_pixel(out_pix, 255u, 224u, 0u);
                }
                else if (two_cloth_layers_on_top)
                {
                    set_debug_pixel(out_pix, 255u, 0u, 255u);
                }
                else
                {
                    set_debug_pixel(out_pix, 32u, 220u, 80u);
                }
            }
        }

        // Only red deformation-map pixels seed the warp. Their target is the
        // closest valid ISO/body pixel, and the displacement fades smoothly to
        // zero over the configured radius. Other deformation colors never seed
        // a displacement.
        std::vector<float> warp_dx;
        std::vector<float> warp_dy;
        const float warp_radius = std::clamp(state.cloth_fitting_warp_radius, 0.0f, 512.0f);
        const float warp_force = std::clamp(state.cloth_fitting_warp_force, 0.0f, 4.0f);
        const int warped_red_count = build_red_deformation_warp(
            out_deformation_rgba,
            iso_body_mask,
            width,
            height,
            warp_radius,
            warp_force,
            warp_dx,
            warp_dy);

        out_warped_rgba = out_rgba;
        out_warped_debug_rgba = out_debug_rgba;
        std::vector<unsigned char> warped_scene_debug = out_scene_debug_rgba;
        std::vector<unsigned char> warped_scene_occlusion = out_scene_occlusion_rgba;
        std::vector<unsigned char> warped_aligned_predicted_depth_occlusion =
            out_aligned_predicted_depth_occlusion_rgba;
        if (warped_red_count > 0)
        {
            std::vector<float> inverse_source_x;
            std::vector<float> inverse_source_y;
            build_inverse_warp_map(
                width,
                height,
                warp_dx,
                warp_dy,
                inverse_source_x,
                inverse_source_y);
            warp_rgba_inverse_smooth(
                out_rgba,
                width,
                height,
                inverse_source_x,
                inverse_source_y,
                out_warped_rgba);
            warp_rgba_inverse_categorical(
                out_debug_rgba,
                width,
                height,
                inverse_source_x,
                inverse_source_y,
                out_warped_debug_rgba);
            warp_rgba_inverse_categorical(
                out_scene_debug_rgba,
                width,
                height,
                inverse_source_x,
                inverse_source_y,
                warped_scene_debug);
            warp_rgba_inverse_categorical(
                out_scene_occlusion_rgba,
                width,
                height,
                inverse_source_x,
                inverse_source_y,
                warped_scene_occlusion);
            warp_rgba_inverse_categorical(
                out_aligned_predicted_depth_occlusion_rgba,
                width,
                height,
                inverse_source_x,
                inverse_source_y,
                warped_aligned_predicted_depth_occlusion);
        }

        // Apply the final visibility decision independently in both spaces. The
        // original branch remains exactly unwarped; the second branch uses the
        // same diagnostics after the liquify transform.
        auto apply_visibility = [&](std::vector<unsigned char>& cloth,
                                    const std::vector<unsigned char>& debug,
                                    const std::vector<unsigned char>& scene_debug,
                                    const std::vector<unsigned char>& scene_occlusion,
                                    const std::vector<unsigned char>& aligned_predicted_depth_occlusion,
                                    std::vector<unsigned char>& occlusion,
                                    int& removed_count)
        {
            occlusion.assign((size_t)width * (size_t)height * 4u, 0u);
            removed_count = 0;
            for (size_t pixel = 0; pixel < (size_t)width * (size_t)height; ++pixel)
            {
                unsigned char* cloth_pixel = cloth.data() + pixel * 4u;
                if (cloth_pixel[3] == 0u)
                    continue;

                const bool depth_occluded =
                    scene_occlusion[pixel * 4u + 3u] == 0u
                    || rgba_is_color(scene_occlusion, pixel, 255u, 48u, 48u);
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

                unsigned char* occlusion_pixel = occlusion.data() + pixel * 4u;
                occlusion_pixel[0] = keep ? 32u : 255u;
                occlusion_pixel[1] = keep ? 220u : 48u;
                occlusion_pixel[2] = keep ? 80u : 48u;
                occlusion_pixel[3] = 220u;
                if (!keep)
                {
                    cloth_pixel[3] = 0u;
                    ++removed_count;
                }
            }
        };

        std::vector<unsigned char> warped_occlusion;
        int warped_removed_count = 0;
        apply_visibility(
            out_rgba,
            out_debug_rgba,
            out_scene_debug_rgba,
            out_scene_occlusion_rgba,
            out_aligned_predicted_depth_occlusion_rgba,
            out_occlusion_rgba,
            out_removed_count);
        apply_visibility(
            out_warped_rgba,
            out_warped_debug_rgba,
            warped_scene_debug,
            warped_scene_occlusion,
            warped_aligned_predicted_depth_occlusion,
            warped_occlusion,
            warped_removed_count);

        console_add_line(
            state,
            "Cloth fitting red warp: seeds=" + std::to_string(warped_red_count)
                + ", radius=" + std::to_string(warp_radius)
                + ", force=" + std::to_string(warp_force)
                + ", removed(no warp/warped)=" + std::to_string(out_removed_count)
                + "/" + std::to_string(warped_removed_count));

        const std::filesystem::path renders_dir = "renders";
        std::error_code ec;
        std::filesystem::create_directories(renders_dir, ec);
        if (ec)
            return fail("could not create renders directory: " + ec.message());

        struct DebugImageOutput
        {
            const char* filename = nullptr;
            const std::vector<unsigned char>* pixels = nullptr;
        };
        const std::array<DebugImageOutput, 13> debug_outputs = {{
            { "cloth_fitting_no_warp.png", &out_rgba },
            { "cloth_fitting_warped.png", &out_warped_rgba },
            { "cloth_fitting_iso_mapping_colors.png", &out_debug_rgba },
            { "cloth_fitting_debug_no_warp.png", &out_debug_rgba },
            { "cloth_fitting_debug_warped.png", &out_warped_debug_rgba },
            { "cloth_fitting_occlusion.png", &out_occlusion_rgba },
            { "cloth_fitting_occlusion_warped.png", &warped_occlusion },
            { "cloth_fitting_true_depth_visibility.png", &out_scene_occlusion_rgba },
            { "cloth_fitting_predicted_depth_visibility.png", &out_predicted_depth_occlusion_rgba },
            { "cloth_fitting_aligned_predicted_depth_visibility.png", &out_aligned_predicted_depth_occlusion_rgba },
            { "cloth_fitting_3d_scene_debug.png", &out_scene_debug_rgba },
            { "cloth_fitting_deformation.png", &out_deformation_rgba },
            { "cloth_fitting_close_to_body.png", &out_close_to_body_rgba },
        }};
        for (const DebugImageOutput& output : debug_outputs)
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
        console_add_line(state, "Cloth fitting debug images saved to renders/");

        return true;
    }
}
