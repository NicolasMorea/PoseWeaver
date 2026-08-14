#include "postproc/PredictionFilters.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cnpy.h>
#include <stb_image_write.h>

namespace poseweaver::postproc
{
    namespace
    {
        struct IsoFilterParams
        {
            float sigma = 2.0f;
            float amount = 3.0f;
            int side_offset = 5;
            float side_sigma = 1.5f;
            float min_contrast = 0.50f;
            float full_contrast = 0.80f;
            float min_coherence_ratio = 4.0f;
            int iterations = 3;
        };

        struct DepthFilterParams
        {
            float sigma = 1.4f;
            float amount = 2.5f;
            int side_offset = 3;
            float side_sigma = 1.2f;
            float min_contrast = 0.035f;
            float full_contrast = 0.12f;
            float min_coherence_ratio = 3.0f;
            int iterations = 2;
        };

        static std::string read_text_file(const std::filesystem::path& path)
        {
            std::ifstream in(path);
            if (!in)
                return {};
            std::ostringstream ss;
            ss << in.rdbuf();
            return ss.str();
        }

        static std::string json_escape(const std::string& value)
        {
            std::string out;
            out.reserve(value.size() + 8u);
            for (char c : value)
            {
                switch (c)
                {
                case '\\': out += "\\\\"; break;
                case '"': out += "\\\""; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out.push_back(c); break;
                }
            }
            return out;
        }

        static std::string parse_json_string_field(const std::string& text, const char* key)
        {
            if (!key || !key[0])
                return {};
            const std::string needle = "\"" + std::string(key) + "\"";
            size_t pos = text.find(needle);
            if (pos == std::string::npos)
                return {};
            pos = text.find(':', pos + needle.size());
            if (pos == std::string::npos)
                return {};
            pos = text.find('"', pos + 1u);
            if (pos == std::string::npos)
                return {};
            ++pos;

            std::string out;
            bool escape = false;
            for (; pos < text.size(); ++pos)
            {
                const char c = text[pos];
                if (escape)
                {
                    switch (c)
                    {
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: out.push_back(c); break;
                    }
                    escape = false;
                    continue;
                }
                if (c == '\\')
                {
                    escape = true;
                    continue;
                }
                if (c == '"')
                    break;
                out.push_back(c);
            }
            return out;
        }

        static std::filesystem::path filtered_path_for(
            const std::filesystem::path& input,
            const char* suffix,
            const char* extension)
        {
            return input.parent_path() / (input.stem().string() + suffix + extension);
        }

        static float npy_value(const cnpy::NpyArray& a, size_t i)
        {
            if (a.word_size == sizeof(float))
                return a.data<float>()[i];
            if (a.word_size == sizeof(double))
                return (float)a.data<double>()[i];
            if (a.word_size == sizeof(std::uint8_t))
                return a.data<std::uint8_t>()[i] != 0u ? 1.0f : 0.0f;
            if (a.word_size == sizeof(bool))
                return a.data<bool>()[i] ? 1.0f : 0.0f;
            return 0.0f;
        }

        static std::vector<float> gaussian_kernel(float sigma)
        {
            const int radius = std::max(1, (int)std::ceil(3.0f * sigma));
            std::vector<float> kernel((size_t)radius * 2u + 1u, 0.0f);
            float sum = 0.0f;
            for (int i = -radius; i <= radius; ++i)
            {
                const float v = std::exp(-0.5f * ((float)i * (float)i) / (sigma * sigma));
                kernel[(size_t)(i + radius)] = v;
                sum += v;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 1.0f;
            for (float& v : kernel)
                v *= inv;
            return kernel;
        }

        static void gaussian_blur_scalar(
            const std::vector<float>& src,
            int w,
            int h,
            float sigma,
            std::vector<float>* out)
        {
            out->assign(src.size(), 0.0f);
            if (w <= 0 || h <= 0 || src.size() != (size_t)w * (size_t)h)
                return;
            const std::vector<float> kernel = gaussian_kernel(sigma);
            const int radius = (int)kernel.size() / 2;
            std::vector<float> tmp(src.size(), 0.0f);

            for (int y = 0; y < h; ++y)
            {
                const size_t row = (size_t)y * (size_t)w;
                for (int x = 0; x < w; ++x)
                {
                    float sum = 0.0f;
                    for (int k = -radius; k <= radius; ++k)
                    {
                        const int sx = std::clamp(x + k, 0, w - 1);
                        sum += src[row + (size_t)sx] * kernel[(size_t)(k + radius)];
                    }
                    tmp[row + (size_t)x] = sum;
                }
            }

            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    float sum = 0.0f;
                    for (int k = -radius; k <= radius; ++k)
                    {
                        const int sy = std::clamp(y + k, 0, h - 1);
                        sum += tmp[(size_t)sy * (size_t)w + (size_t)x] * kernel[(size_t)(k + radius)];
                    }
                    (*out)[(size_t)y * (size_t)w + (size_t)x] = sum;
                }
            }
        }

        static void gaussian_blur_channels(
            const std::vector<float>& src,
            int w,
            int h,
            int channels,
            float sigma,
            std::vector<float>* out)
        {
            out->assign(src.size(), 0.0f);
            if (w <= 0 || h <= 0 || channels <= 0
                || src.size() != (size_t)w * (size_t)h * (size_t)channels)
                return;
            const std::vector<float> kernel = gaussian_kernel(sigma);
            const int radius = (int)kernel.size() / 2;
            std::vector<float> tmp(src.size(), 0.0f);
            const size_t csz = (size_t)channels;

            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    float* dst = tmp.data() + ((size_t)y * (size_t)w + (size_t)x) * csz;
                    for (int k = -radius; k <= radius; ++k)
                    {
                        const int sx = std::clamp(x + k, 0, w - 1);
                        const float weight = kernel[(size_t)(k + radius)];
                        const float* sp = src.data() + ((size_t)y * (size_t)w + (size_t)sx) * csz;
                        for (int c = 0; c < channels; ++c)
                            dst[c] += sp[c] * weight;
                    }
                }
            }

            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    float* dst = out->data() + ((size_t)y * (size_t)w + (size_t)x) * csz;
                    for (int k = -radius; k <= radius; ++k)
                    {
                        const int sy = std::clamp(y + k, 0, h - 1);
                        const float weight = kernel[(size_t)(k + radius)];
                        const float* sp = tmp.data() + ((size_t)sy * (size_t)w + (size_t)x) * csz;
                        for (int c = 0; c < channels; ++c)
                            dst[c] += sp[c] * weight;
                    }
                }
            }
        }

        static void masked_gaussian_channels(
            const std::vector<float>& values,
            const std::vector<float>& weights,
            int w,
            int h,
            int channels,
            float sigma,
            std::vector<float>* out)
        {
            std::vector<float> weighted(values.size(), 0.0f);
            for (size_t p = 0, count = weights.size(); p < count; ++p)
            {
                const float weight = weights[p];
                float* dst = weighted.data() + p * (size_t)channels;
                const float* src = values.data() + p * (size_t)channels;
                for (int c = 0; c < channels; ++c)
                    dst[c] = src[c] * weight;
            }
            std::vector<float> numerator;
            std::vector<float> denominator;
            gaussian_blur_channels(weighted, w, h, channels, sigma, &numerator);
            gaussian_blur_scalar(weights, w, h, sigma, &denominator);
            out->assign(values.size(), 0.0f);
            for (size_t p = 0, count = weights.size(); p < count; ++p)
            {
                const float inv = 1.0f / std::max(denominator[p], 1e-6f);
                float* dst = out->data() + p * (size_t)channels;
                const float* src = numerator.data() + p * (size_t)channels;
                for (int c = 0; c < channels; ++c)
                    dst[c] = src[c] * inv;
            }
        }

        static void masked_gaussian_scalar(
            const std::vector<float>& values,
            const std::vector<float>& weights,
            int w,
            int h,
            float sigma,
            std::vector<float>* out)
        {
            std::vector<float> weighted(values.size(), 0.0f);
            for (size_t i = 0; i < values.size(); ++i)
                weighted[i] = values[i] * weights[i];
            std::vector<float> numerator;
            std::vector<float> denominator;
            gaussian_blur_scalar(weighted, w, h, sigma, &numerator);
            gaussian_blur_scalar(weights, w, h, sigma, &denominator);
            out->assign(values.size(), 0.0f);
            for (size_t i = 0; i < values.size(); ++i)
                (*out)[i] = numerator[i] / std::max(denominator[i], 1e-6f);
        }

        static float smoothstep01(float x)
        {
            x = std::clamp(x, 0.0f, 1.0f);
            return x * x * (3.0f - 2.0f * x);
        }

        static bool shifted_valid(int x, int y, int dx, int dy, int w, int h, int* out_x, int* out_y)
        {
            const int sx = x + dx;
            const int sy = y + dy;
            if (sx < 0 || sx >= w || sy < 0 || sy >= h)
                return false;
            *out_x = sx;
            *out_y = sy;
            return true;
        }

        static void two_sided_gate_channels(
            const std::vector<float>& descriptors,
            const std::vector<float>& weights,
            const std::vector<uint8_t>& mask,
            int w,
            int h,
            int channels,
            const IsoFilterParams& params,
            std::vector<float>* out_gate)
        {
            std::vector<float> side_mean;
            std::vector<float> side_second_input(descriptors.size(), 0.0f);
            for (size_t i = 0; i < descriptors.size(); ++i)
                side_second_input[i] = descriptors[i] * descriptors[i];
            masked_gaussian_channels(descriptors, weights, w, h, channels, params.side_sigma, &side_mean);
            std::vector<float> side_second;
            masked_gaussian_channels(side_second_input, weights, w, h, channels, params.side_sigma, &side_second);

            const size_t pixel_count = (size_t)w * (size_t)h;
            std::vector<float> side_std(pixel_count, 0.0f);
            for (size_t p = 0; p < pixel_count; ++p)
            {
                float mean_var = 0.0f;
                const float* mean = side_mean.data() + p * (size_t)channels;
                const float* second = side_second.data() + p * (size_t)channels;
                for (int c = 0; c < channels; ++c)
                    mean_var += std::max(second[c] - mean[c] * mean[c], 0.0f);
                side_std[p] = std::sqrt(mean_var / (float)channels);
            }

            out_gate->assign(pixel_count, 0.0f);
            constexpr int dirs[4][2] = { { 1, 0 }, { 0, 1 }, { 1, 1 }, { -1, 1 } };
            const float sqrt_channels = std::sqrt((float)channels);
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    const size_t p = (size_t)y * (size_t)w + (size_t)x;
                    if (!mask[p])
                        continue;
                    float best_gate = 0.0f;
                    const float* center = descriptors.data() + p * (size_t)channels;
                    for (const auto& dir : dirs)
                    {
                        int lx = 0, ly = 0, rx = 0, ry = 0;
                        if (!shifted_valid(x, y, -dir[0] * params.side_offset, -dir[1] * params.side_offset, w, h, &lx, &ly)
                            || !shifted_valid(x, y, dir[0] * params.side_offset, dir[1] * params.side_offset, w, h, &rx, &ry))
                            continue;
                        const size_t lp = (size_t)ly * (size_t)w + (size_t)lx;
                        const size_t rp = (size_t)ry * (size_t)w + (size_t)rx;
                        if (!mask[lp] || !mask[rp])
                            continue;

                        const float* left = side_mean.data() + lp * (size_t)channels;
                        const float* right = side_mean.data() + rp * (size_t)channels;
                        float sq = 0.0f;
                        float dot_center = 0.0f;
                        for (int c = 0; c < channels; ++c)
                        {
                            const float d = right[c] - left[c];
                            sq += d * d;
                            dot_center += (center[c] - left[c]) * d;
                        }
                        const float contrast = std::sqrt(sq);
                        const float within = (side_std[lp] + side_std[rp]) * sqrt_channels;
                        const float coherence = contrast / std::max(within, 1e-5f);
                        const float contrast_weight = smoothstep01(
                            (contrast - params.min_contrast)
                            / std::max(params.full_contrast - params.min_contrast, 1e-6f));
                        const float coherence_weight = smoothstep01(
                            (coherence - params.min_coherence_ratio) / 0.8f);
                        const float t = dot_center / std::max(sq, 1e-8f);
                        const float mixture_weight = std::clamp(4.0f * t * (1.0f - t), 0.0f, 1.0f);
                        best_gate = std::max(best_gate, contrast_weight * coherence_weight * mixture_weight);
                    }
                    (*out_gate)[p] = best_gate;
                }
            }
        }

        static void two_sided_gate_scalar(
            const std::vector<float>& values,
            const std::vector<float>& weights,
            const std::vector<uint8_t>& mask,
            int w,
            int h,
            const DepthFilterParams& params,
            std::vector<float>* out_gate)
        {
            std::vector<float> side_mean;
            std::vector<float> squared(values.size(), 0.0f);
            for (size_t i = 0; i < values.size(); ++i)
                squared[i] = values[i] * values[i];
            masked_gaussian_scalar(values, weights, w, h, params.side_sigma, &side_mean);
            std::vector<float> side_second;
            masked_gaussian_scalar(squared, weights, w, h, params.side_sigma, &side_second);
            std::vector<float> side_std(values.size(), 0.0f);
            for (size_t i = 0; i < values.size(); ++i)
                side_std[i] = std::sqrt(std::max(side_second[i] - side_mean[i] * side_mean[i], 0.0f));

            out_gate->assign(values.size(), 0.0f);
            constexpr int dirs[4][2] = { { 1, 0 }, { 0, 1 }, { 1, 1 }, { -1, 1 } };
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    const size_t p = (size_t)y * (size_t)w + (size_t)x;
                    if (!mask[p])
                        continue;
                    float best_gate = 0.0f;
                    for (const auto& dir : dirs)
                    {
                        int lx = 0, ly = 0, rx = 0, ry = 0;
                        if (!shifted_valid(x, y, -dir[0] * params.side_offset, -dir[1] * params.side_offset, w, h, &lx, &ly)
                            || !shifted_valid(x, y, dir[0] * params.side_offset, dir[1] * params.side_offset, w, h, &rx, &ry))
                            continue;
                        const size_t lp = (size_t)ly * (size_t)w + (size_t)lx;
                        const size_t rp = (size_t)ry * (size_t)w + (size_t)rx;
                        if (!mask[lp] || !mask[rp])
                            continue;

                        const float left = side_mean[lp];
                        const float right = side_mean[rp];
                        const float diff = right - left;
                        const float contrast = std::abs(diff);
                        const float coherence = contrast / std::max(side_std[lp] + side_std[rp], 1e-5f);
                        const float contrast_weight = smoothstep01(
                            (contrast - params.min_contrast)
                            / std::max(params.full_contrast - params.min_contrast, 1e-6f));
                        const float coherence_weight = smoothstep01(
                            (coherence - params.min_coherence_ratio) / 0.8f);
                        const float t = (values[p] - left) * diff / std::max(diff * diff, 1e-8f);
                        const float mixture_weight = std::clamp(4.0f * t * (1.0f - t), 0.0f, 1.0f);
                        best_gate = std::max(best_gate, contrast_weight * coherence_weight * mixture_weight);
                    }
                    (*out_gate)[p] = best_gate;
                }
            }
        }

        static void local_min_max_channels(
            const std::vector<float>& values,
            int w,
            int h,
            int channels,
            std::vector<float>* out_min,
            std::vector<float>* out_max)
        {
            out_min->assign(values.size(), std::numeric_limits<float>::max());
            out_max->assign(values.size(), -std::numeric_limits<float>::max());
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    const size_t p = ((size_t)y * (size_t)w + (size_t)x) * (size_t)channels;
                    for (int oy = -1; oy <= 1; ++oy)
                    {
                        const int sy = std::clamp(y + oy, 0, h - 1);
                        for (int ox = -1; ox <= 1; ++ox)
                        {
                            const int sx = std::clamp(x + ox, 0, w - 1);
                            const float* src = values.data()
                                + ((size_t)sy * (size_t)w + (size_t)sx) * (size_t)channels;
                            for (int c = 0; c < channels; ++c)
                            {
                                (*out_min)[p + (size_t)c] = std::min((*out_min)[p + (size_t)c], src[c]);
                                (*out_max)[p + (size_t)c] = std::max((*out_max)[p + (size_t)c], src[c]);
                            }
                        }
                    }
                }
            }
        }

        static void local_min_max_scalar(
            const std::vector<float>& values,
            int w,
            int h,
            std::vector<float>* out_min,
            std::vector<float>* out_max)
        {
            out_min->assign(values.size(), std::numeric_limits<float>::max());
            out_max->assign(values.size(), -std::numeric_limits<float>::max());
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    const size_t p = (size_t)y * (size_t)w + (size_t)x;
                    for (int oy = -1; oy <= 1; ++oy)
                    {
                        const int sy = std::clamp(y + oy, 0, h - 1);
                        for (int ox = -1; ox <= 1; ++ox)
                        {
                            const int sx = std::clamp(x + ox, 0, w - 1);
                            const float v = values[(size_t)sy * (size_t)w + (size_t)sx];
                            (*out_min)[p] = std::min((*out_min)[p], v);
                            (*out_max)[p] = std::max((*out_max)[p], v);
                        }
                    }
                }
            }
        }

        static bool load_iso(
            const std::filesystem::path& input_npz,
            int* out_w,
            int* out_h,
            int* out_channels,
            std::vector<float>* out_isomap,
            std::vector<uint8_t>* out_mask,
            std::vector<float>* out_mask_probability,
            std::vector<float>* out_low,
            std::vector<float>* out_high,
            std::string* out_error)
        {
            cnpy::NpyArray isomap;
            try
            {
                isomap = cnpy::npz_load(input_npz.string(), "isomap");
            }
            catch (const std::exception& e)
            {
                if (out_error)
                    *out_error = std::string("could not load isomap: ") + e.what();
                return false;
            }
            if (isomap.shape.size() != 3
                || (isomap.word_size != sizeof(float) && isomap.word_size != sizeof(double)))
            {
                if (out_error)
                    *out_error = "isomap must be a float array shaped [H,W,C] or [C,H,W]";
                return false;
            }
            const bool hwc = isomap.shape[2] >= 3 && isomap.shape[2] <= 256;
            const bool chw = !hwc && isomap.shape[0] >= 3 && isomap.shape[0] <= 256;
            if (!hwc && !chw)
            {
                if (out_error)
                    *out_error = "could not identify isomap channel dimension";
                return false;
            }
            const int h = (int)(hwc ? isomap.shape[0] : isomap.shape[1]);
            const int w = (int)(hwc ? isomap.shape[1] : isomap.shape[2]);
            const int channels = (int)(hwc ? isomap.shape[2] : isomap.shape[0]);
            if (w <= 0 || h <= 0 || channels < 3)
            {
                if (out_error)
                    *out_error = "invalid isomap dimensions";
                return false;
            }

            const size_t pixel_count = (size_t)w * (size_t)h;
            out_isomap->assign(pixel_count * (size_t)channels, 0.0f);
            for (size_t p = 0; p < pixel_count; ++p)
            {
                for (int c = 0; c < channels; ++c)
                {
                    const size_t source = hwc
                        ? p * (size_t)channels + (size_t)c
                        : (size_t)c * pixel_count + p;
                    (*out_isomap)[p * (size_t)channels + (size_t)c] = npy_value(isomap, source);
                }
            }

            out_mask->assign(pixel_count, 1u);
            try
            {
                cnpy::NpyArray mask = cnpy::npz_load(input_npz.string(), "mask");
                if (mask.shape.size() == 2
                    && (int)mask.shape[0] == h
                    && (int)mask.shape[1] == w)
                {
                    for (size_t p = 0; p < pixel_count; ++p)
                        (*out_mask)[p] = npy_value(mask, p) > 0.5f ? 1u : 0u;
                }
            }
            catch (const std::exception&)
            {
            }

            out_mask_probability->assign(pixel_count, 0.0f);
            try
            {
                cnpy::NpyArray prob = cnpy::npz_load(input_npz.string(), "mask_probability");
                if (prob.shape.size() == 2
                    && (int)prob.shape[0] == h
                    && (int)prob.shape[1] == w)
                {
                    for (size_t p = 0; p < pixel_count; ++p)
                        (*out_mask_probability)[p] = std::clamp(npy_value(prob, p), 0.0f, 1.0f);
                }
            }
            catch (const std::exception&)
            {
            }
            for (size_t p = 0; p < pixel_count; ++p)
            {
                if ((*out_mask_probability)[p] <= 0.0f)
                    (*out_mask_probability)[p] = (*out_mask)[p] ? 1.0f : 0.0f;
            }

            out_low->assign((size_t)channels, std::numeric_limits<float>::max());
            out_high->assign((size_t)channels, -std::numeric_limits<float>::max());
            bool have_range = false;
            try
            {
                cnpy::NpyArray low = cnpy::npz_load(input_npz.string(), "descriptor_min");
                cnpy::NpyArray high = cnpy::npz_load(input_npz.string(), "descriptor_max");
                if (low.shape.size() == 1 && high.shape.size() == 1
                    && (int)low.shape[0] == channels && (int)high.shape[0] == channels
                    && (low.word_size == sizeof(float) || low.word_size == sizeof(double))
                    && (high.word_size == sizeof(float) || high.word_size == sizeof(double)))
                {
                    for (int c = 0; c < channels; ++c)
                    {
                        (*out_low)[(size_t)c] = npy_value(low, (size_t)c);
                        (*out_high)[(size_t)c] = npy_value(high, (size_t)c);
                    }
                    have_range = true;
                }
            }
            catch (const std::exception&)
            {
            }
            if (!have_range)
            {
                for (size_t p = 0; p < pixel_count; ++p)
                {
                    if (!(*out_mask)[p])
                        continue;
                    const float* src = out_isomap->data() + p * (size_t)channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        (*out_low)[(size_t)c] = std::min((*out_low)[(size_t)c], src[c]);
                        (*out_high)[(size_t)c] = std::max((*out_high)[(size_t)c], src[c]);
                    }
                }
                for (int c = 0; c < channels; ++c)
                {
                    if (!std::isfinite((*out_low)[(size_t)c])
                        || !std::isfinite((*out_high)[(size_t)c])
                        || (*out_high)[(size_t)c] <= (*out_low)[(size_t)c])
                    {
                        (*out_low)[(size_t)c] = 0.0f;
                        (*out_high)[(size_t)c] = 1.0f;
                    }
                }
            }

            *out_w = w;
            *out_h = h;
            *out_channels = channels;
            return true;
        }

        static void write_depth_json(
            const std::filesystem::path& json_path,
            const std::filesystem::path& source_path,
            const std::filesystem::path& npz_path,
            const std::filesystem::path& render_path,
            int image_size,
            float depth_min,
            float depth_max)
        {
            std::ofstream out(json_path);
            out << "{\n";
            out << "  \"schema\": \"poseweaver.sketch2depth.v1.filtered\",\n";
            out << "  \"source_prediction\": \"" << json_escape(source_path.string()) << "\",\n";
            out << "  \"image_size\": " << image_size << ",\n";
            out << "  \"render_path\": \"" << json_escape(render_path.string()) << "\",\n";
            out << "  \"depth_npz_path\": \"" << json_escape(npz_path.string()) << "\",\n";
            out << "  \"depth_min\": " << depth_min << ",\n";
            out << "  \"depth_max\": " << depth_max << ",\n";
            out << "  \"postprocess_filter\": \"two_sided_depth_unsharp_cpp\"\n";
            out << "}\n";
        }
    }

    bool filter_isomap_prediction(
        const std::filesystem::path& input_npz,
        std::filesystem::path* out_npz,
        std::string* out_error)
    {
        if (input_npz.empty())
        {
            if (out_error)
                *out_error = "missing isomap prediction path";
            return false;
        }
        const std::filesystem::path output_npz =
            filtered_path_for(input_npz, "_filtered", ".npz");

        int w = 0, h = 0, channels = 0;
        std::vector<float> isomap;
        std::vector<uint8_t> mask;
        std::vector<float> mask_probability;
        std::vector<float> low;
        std::vector<float> high;
        if (!load_iso(
                input_npz,
                &w,
                &h,
                &channels,
                &isomap,
                &mask,
                &mask_probability,
                &low,
                &high,
                out_error))
            return false;

        const IsoFilterParams params;
        const size_t pixel_count = (size_t)w * (size_t)h;
        std::vector<float> scale((size_t)channels, 1.0f);
        std::vector<float> normalized(isomap.size(), 0.0f);
        for (int c = 0; c < channels; ++c)
            scale[(size_t)c] = std::max(high[(size_t)c] - low[(size_t)c], 1e-6f);
        for (size_t p = 0; p < pixel_count; ++p)
        {
            for (int c = 0; c < channels; ++c)
            {
                const size_t i = p * (size_t)channels + (size_t)c;
                normalized[i] = (isomap[i] - low[(size_t)c]) / scale[(size_t)c];
            }
        }

        std::vector<float> weights(pixel_count, 0.0f);
        for (size_t p = 0; p < pixel_count; ++p)
            weights[p] = mask[p] ? std::clamp(mask_probability[p], 0.0f, 1.0f) : 0.0f;

        std::vector<float> result = normalized;
        for (int iteration = 0; iteration < params.iterations; ++iteration)
        {
            std::vector<float> metric(result.size(), 0.0f);
            for (size_t p = 0; p < pixel_count; ++p)
            {
                for (int c = 0; c < channels; ++c)
                {
                    const size_t i = p * (size_t)channels + (size_t)c;
                    metric[i] = result[i] * scale[(size_t)c] + low[(size_t)c];
                }
            }

            std::vector<float> gate;
            two_sided_gate_channels(metric, weights, mask, w, h, channels, params, &gate);
            std::vector<float> local_signal;
            masked_gaussian_channels(result, weights, w, h, channels, params.sigma, &local_signal);
            std::vector<float> local_min;
            std::vector<float> local_max;
            local_min_max_channels(result, w, h, channels, &local_min, &local_max);

            for (size_t p = 0; p < pixel_count; ++p)
            {
                for (int c = 0; c < channels; ++c)
                {
                    const size_t i = p * (size_t)channels + (size_t)c;
                    if (!mask[p])
                    {
                        result[i] = normalized[i];
                        continue;
                    }
                    const float candidate = result[i]
                        + params.amount * gate[p] * (result[i] - local_signal[i]);
                    result[i] = std::clamp(candidate, local_min[i], local_max[i]);
                }
            }
        }

        std::vector<float> filtered(isomap.size(), 0.0f);
        for (size_t p = 0; p < pixel_count; ++p)
        {
            for (int c = 0; c < channels; ++c)
            {
                const size_t i = p * (size_t)channels + (size_t)c;
                filtered[i] = mask[p]
                    ? (result[i] * scale[(size_t)c] + low[(size_t)c])
                    : isomap[i];
            }
        }

        try
        {
            cnpy::npz_save(
                output_npz.string(),
                "isomap",
                filtered.data(),
                { (size_t)h, (size_t)w, (size_t)channels },
                "w");
            cnpy::npz_save(
                output_npz.string(),
                "mask",
                mask.data(),
                { (size_t)h, (size_t)w },
                "a");
            cnpy::npz_save(
                output_npz.string(),
                "mask_probability",
                mask_probability.data(),
                { (size_t)h, (size_t)w },
                "a");
            cnpy::npz_save(
                output_npz.string(),
                "descriptor_min",
                low.data(),
                { (size_t)channels },
                "a");
            cnpy::npz_save(
                output_npz.string(),
                "descriptor_max",
                high.data(),
                { (size_t)channels },
                "a");
            const float post_sigma = params.sigma;
            const float post_amount = params.amount;
            const int post_iterations = params.iterations;
            cnpy::npz_save(output_npz.string(), "postprocess_sigma", &post_sigma, { 1u }, "a");
            cnpy::npz_save(output_npz.string(), "postprocess_amount", &post_amount, { 1u }, "a");
            cnpy::npz_save(output_npz.string(), "postprocess_iterations", &post_iterations, { 1u }, "a");
        }
        catch (const std::exception& e)
        {
            if (out_error)
                *out_error = std::string("could not write filtered isomap: ") + e.what();
            return false;
        }

        if (out_npz)
            *out_npz = output_npz;
        return true;
    }

    bool filter_depth_prediction(
        const std::filesystem::path& input_json_or_npz,
        std::filesystem::path* out_json,
        std::string* out_error)
    {
        if (input_json_or_npz.empty())
        {
            if (out_error)
                *out_error = "missing depth prediction path";
            return false;
        }

        std::filesystem::path input_npz = input_json_or_npz;
        if (input_json_or_npz.extension() == ".json")
        {
            const std::string text = read_text_file(input_json_or_npz);
            input_npz = parse_json_string_field(text, "depth_npz_path");
            if (input_npz.empty())
                input_npz = input_json_or_npz.parent_path()
                    / (input_json_or_npz.stem().string() + "_depth.npz");
            else if (input_npz.is_relative())
                input_npz = input_json_or_npz.parent_path() / input_npz;
        }

        cnpy::NpyArray depth;
        cnpy::NpyArray mask_arr;
        cnpy::NpyArray depth_min_arr;
        cnpy::NpyArray depth_max_arr;
        cnpy::NpyArray boundary_arr;
        bool have_boundary = false;
        try
        {
            depth = cnpy::npz_load(input_npz.string(), "depth");
            mask_arr = cnpy::npz_load(input_npz.string(), "mask");
            depth_min_arr = cnpy::npz_load(input_npz.string(), "depth_min");
            depth_max_arr = cnpy::npz_load(input_npz.string(), "depth_max");
            try
            {
                boundary_arr = cnpy::npz_load(input_npz.string(), "boundary");
                have_boundary = boundary_arr.shape == depth.shape;
            }
            catch (const std::exception&)
            {
            }
        }
        catch (const std::exception& e)
        {
            if (out_error)
                *out_error = std::string("could not load depth prediction: ") + e.what();
            return false;
        }

        if (depth.shape.size() != 2 || mask_arr.shape.size() != 2
            || depth.shape != mask_arr.shape
            || depth.shape[0] == 0 || depth.shape[1] == 0
            || (depth.word_size != sizeof(float) && depth.word_size != sizeof(double)))
        {
            if (out_error)
                *out_error = "depth prediction must contain matching [H,W] depth and mask arrays";
            return false;
        }

        const int h = (int)depth.shape[0];
        const int w = (int)depth.shape[1];
        const size_t pixel_count = (size_t)w * (size_t)h;
        std::vector<float> original(pixel_count, 0.0f);
        std::vector<float> mask_probability(pixel_count, 0.0f);
        std::vector<uint8_t> mask(pixel_count, 0u);
        std::vector<float> boundary(pixel_count, 0.0f);
        for (size_t p = 0; p < pixel_count; ++p)
        {
            original[p] = npy_value(depth, p);
            mask_probability[p] = std::clamp(npy_value(mask_arr, p), 0.0f, 1.0f);
            mask[p] = mask_probability[p] > 0.5f ? 1u : 0u;
            if (have_boundary)
                boundary[p] = npy_value(boundary_arr, p);
        }

        const DepthFilterParams params;
        std::vector<float> weights(pixel_count, 0.0f);
        for (size_t p = 0; p < pixel_count; ++p)
            weights[p] = mask[p] ? mask_probability[p] : 0.0f;

        std::vector<float> result = original;
        for (int iteration = 0; iteration < params.iterations; ++iteration)
        {
            std::vector<float> gate;
            two_sided_gate_scalar(result, weights, mask, w, h, params, &gate);
            std::vector<float> local_signal;
            masked_gaussian_scalar(result, weights, w, h, params.sigma, &local_signal);
            std::vector<float> local_min;
            std::vector<float> local_max;
            local_min_max_scalar(result, w, h, &local_min, &local_max);
            for (size_t p = 0; p < pixel_count; ++p)
            {
                if (!mask[p])
                {
                    result[p] = original[p];
                    continue;
                }
                const float candidate = result[p]
                    + params.amount * gate[p] * (result[p] - local_signal[p]);
                result[p] = std::clamp(candidate, local_min[p], local_max[p]);
            }
        }

        const std::filesystem::path output_json =
            filtered_path_for(input_json_or_npz, "_filtered", ".json");
        const std::filesystem::path output_npz =
            filtered_path_for(input_npz, "_filtered", ".npz");
        const std::filesystem::path output_render =
            filtered_path_for(input_json_or_npz, "_filtered_render", ".png");
        const float depth_min = depth_min_arr.num_vals > 0 ? npy_value(depth_min_arr, 0) : 0.0f;
        const float depth_max = depth_max_arr.num_vals > 0 ? npy_value(depth_max_arr, 0) : 1.0f;

        std::vector<unsigned char> rgba(pixel_count * 4u, 0u);
        for (size_t p = 0; p < pixel_count; ++p)
        {
            const unsigned char gray = (unsigned char)std::clamp(
                (int)std::lround(std::clamp(result[p], 0.0f, 1.0f) * 255.0f),
                0,
                255);
            rgba[p * 4u + 0u] = gray;
            rgba[p * 4u + 1u] = gray;
            rgba[p * 4u + 2u] = gray;
            rgba[p * 4u + 3u] = mask[p] ? 255u : 0u;
        }

        try
        {
            cnpy::npz_save(output_npz.string(), "depth", result.data(), { (size_t)h, (size_t)w }, "w");
            cnpy::npz_save(output_npz.string(), "mask", mask_probability.data(), { (size_t)h, (size_t)w }, "a");
            cnpy::npz_save(output_npz.string(), "boundary", boundary.data(), { (size_t)h, (size_t)w }, "a");
            cnpy::npz_save(output_npz.string(), "depth_min", &depth_min, { 1u }, "a");
            cnpy::npz_save(output_npz.string(), "depth_max", &depth_max, { 1u }, "a");
            const float post_sigma = params.sigma;
            const float post_amount = params.amount;
            const int post_iterations = params.iterations;
            cnpy::npz_save(output_npz.string(), "postprocess_sigma", &post_sigma, { 1u }, "a");
            cnpy::npz_save(output_npz.string(), "postprocess_amount", &post_amount, { 1u }, "a");
            cnpy::npz_save(output_npz.string(), "postprocess_iterations", &post_iterations, { 1u }, "a");
        }
        catch (const std::exception& e)
        {
            if (out_error)
                *out_error = std::string("could not write filtered depth NPZ: ") + e.what();
            return false;
        }
        if (stbi_write_png(output_render.string().c_str(), w, h, 4, rgba.data(), w * 4) == 0)
        {
            if (out_error)
                *out_error = "could not write filtered depth render: " + output_render.string();
            return false;
        }
        write_depth_json(
            output_json,
            input_json_or_npz,
            std::filesystem::absolute(output_npz),
            std::filesystem::absolute(output_render),
            w,
            depth_min,
            depth_max);

        if (out_json)
            *out_json = output_json;
        return true;
    }
}
