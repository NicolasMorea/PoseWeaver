#include "render/Renderer.h"

#include "core/Paths.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
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
        constexpr int kIsoDimensions = 64;
        constexpr float kSketch2RigVerticalFovRad = 0.7348438501358032f;
        constexpr float kSketch2RigVerticalFovDeg =
            kSketch2RigVerticalFovRad * 180.0f / 3.14159265358979323846f;

        static float npy_float_value(const cnpy::NpyArray& array, size_t index)
        {
            if (array.word_size == sizeof(float))
                return array.data<float>()[index];
            if (array.word_size == sizeof(double))
                return (float)array.data<double>()[index];
            if (array.word_size == sizeof(std::uint8_t))
                return array.data<std::uint8_t>()[index] != 0u ? 1.0f : 0.0f;
            return 0.0f;
        }

        static std::filesystem::path find_iso_sketch_path(
            const AppState& state,
            const std::filesystem::path& npz_path)
        {
            std::error_code ec;
            if (npz_path == std::filesystem::path(state.sketch2iso.last_output_path)
                && !state.sketch2iso.last_sketch_path.empty())
            {
                const std::filesystem::path sketch_path = state.sketch2iso.last_sketch_path;
                if (std::filesystem::exists(sketch_path, ec))
                    return sketch_path;
            }

            const std::filesystem::path candidates[] = {
                npz_path.parent_path() / (npz_path.stem().string() + ".png"),
                npz_path.parent_path() / (npz_path.stem().string() + ".jpg"),
                npz_path.parent_path().parent_path() / (npz_path.stem().string() + ".png"),
                npz_path.parent_path().parent_path() / (npz_path.stem().string() + ".jpg"),
            };
            for (const auto& candidate : candidates)
            {
                ec.clear();
                if (std::filesystem::exists(candidate, ec))
                    return candidate;
            }
            return {};
        }

        static std::vector<unsigned char> resize_rgba_nearest(
            const unsigned char* source,
            int source_width,
            int source_height,
            int target_width,
            int target_height)
        {
            std::vector<unsigned char> output(
                (size_t)target_width * (size_t)target_height * 4u,
                0u);
            if (!source || source_width <= 0 || source_height <= 0
                || target_width <= 0 || target_height <= 0)
                return output;

            for (int y = 0; y < target_height; ++y)
            {
                const int sy = std::clamp(
                    (int)(((long long)y * source_height) / target_height),
                    0,
                    source_height - 1);
                for (int x = 0; x < target_width; ++x)
                {
                    const int sx = std::clamp(
                        (int)(((long long)x * source_width) / target_width),
                        0,
                        source_width - 1);
                    const unsigned char* src =
                        source + ((size_t)sy * (size_t)source_width + (size_t)sx) * 4u;
                    unsigned char* dst =
                        output.data() + ((size_t)y * (size_t)target_width + (size_t)x) * 4u;
                    dst[0] = src[0];
                    dst[1] = src[1];
                    dst[2] = src[2];
                    dst[3] = src[3];
                }
            }
            return output;
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

            if (!eye.allFinite() || !dir.allFinite() || !up.allFinite()
                || !(dir.squaredNorm() > 1e-12f))
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
            const Eigen::Vector3d& point,
            int width,
            int height,
            float& out_x,
            float& out_y)
        {
            const Eigen::Vector4f hp(
                (float)point.x(),
                (float)point.y(),
                (float)point.z(),
                1.0f);
            const Eigen::Vector4f clip = (proj * view) * hp;
            if (!(clip.w() > 1e-8f) || !std::isfinite(clip.w()))
                return false;

            const Eigen::Vector3f ndc = clip.head<3>() / clip.w();
            if (!ndc.allFinite() || ndc.z() < -1.0f || ndc.z() > 1.0f)
                return false;

            out_x = (ndc.x() * 0.5f + 0.5f) * (float)width;
            out_y = (-ndc.y() * 0.5f + 0.5f) * (float)height;
            return true;
        }
    }

    bool Renderer::ensure_iso_vertex_descriptors(AppState& state)
    {
        if (m_iso_vertex_descriptor_count > 0
            && m_iso_vertex_descriptor_dimensions == kIsoDimensions
            && !m_iso_vertex_descriptors.empty())
            return true;
        if (m_iso_vertex_descriptors_load_attempted)
            return false;
        m_iso_vertex_descriptors_load_attempted = true;

        const std::filesystem::path mapping_path =
            assets_dir() / "smplh" / "isomap_64d.npz";
        cnpy::NpyArray descriptors;
        try
        {
            try
            {
                descriptors = cnpy::npz_load(mapping_path.string(), "Z");
            }
            catch (const std::exception&)
            {
                descriptors = cnpy::npz_load(mapping_path.string(), "embedding");
            }
        }
        catch (const std::exception& e)
        {
            console_add_line(state, "ISO mapping failed to load: " + std::string(e.what()));
            return false;
        }

        if (descriptors.shape.size() != 2
            || descriptors.shape[1] != kIsoDimensions
            || (descriptors.word_size != sizeof(float)
                && descriptors.word_size != sizeof(double)))
        {
            console_add_line(state, "ISO mapping must be a float [vertices,64] array");
            return false;
        }

        const int vertex_count = (int)descriptors.shape[0];
        m_iso_vertex_descriptors.resize((size_t)vertex_count * (size_t)kIsoDimensions);
        for (size_t i = 0; i < m_iso_vertex_descriptors.size(); ++i)
            m_iso_vertex_descriptors[i] = npy_float_value(descriptors, i);

        m_iso_vertex_descriptor_count = vertex_count;
        m_iso_vertex_descriptor_dimensions = kIsoDimensions;

        std::vector<int> indices((size_t)vertex_count);
        std::iota(indices.begin(), indices.end(), 0);
        m_iso_kd_nodes.clear();
        m_iso_kd_nodes.reserve((size_t)vertex_count);
        const auto build_node = [&](const auto& self, int begin, int end) -> int
        {
            if (begin >= end)
                return -1;

            int axis = 0;
            float best_range = -1.0f;
            for (int channel = 0; channel < kIsoDimensions; ++channel)
            {
                float minimum = std::numeric_limits<float>::infinity();
                float maximum = -std::numeric_limits<float>::infinity();
                for (int i = begin; i < end; ++i)
                {
                    const float value = m_iso_vertex_descriptors[
                        (size_t)indices[(size_t)i] * (size_t)kIsoDimensions
                        + (size_t)channel];
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
                const float range = maximum - minimum;
                if (range > best_range)
                {
                    best_range = range;
                    axis = channel;
                }
            }

            const int middle = begin + (end - begin) / 2;
            std::nth_element(
                indices.begin() + begin,
                indices.begin() + middle,
                indices.begin() + end,
                [&](int a, int b)
                {
                    return m_iso_vertex_descriptors[
                        (size_t)a * (size_t)kIsoDimensions + (size_t)axis]
                        < m_iso_vertex_descriptors[
                            (size_t)b * (size_t)kIsoDimensions + (size_t)axis];
                });

            const int node_index = (int)m_iso_kd_nodes.size();
            m_iso_kd_nodes.push_back(
                IsoKdNode{ indices[(size_t)middle], -1, -1, axis });
            const int left = self(self, begin, middle);
            const int right = self(self, middle + 1, end);
            m_iso_kd_nodes[(size_t)node_index].left = left;
            m_iso_kd_nodes[(size_t)node_index].right = right;
            return node_index;
        };
        m_iso_kd_root = build_node(build_node, 0, vertex_count);

        console_add_line(
            state,
            "ISO vertex mapping ready: " + std::to_string(vertex_count)
                + " vertices x " + std::to_string(kIsoDimensions));
        return true;
    }

    void Renderer::ensure_sketch2iso_prediction_texture(AppState& state)
    {
        const std::filesystem::path desired_npz =
            !state.canva_manual_iso_npz_path.empty()
            ? std::filesystem::path(state.canva_manual_iso_npz_path)
            : std::filesystem::path(state.sketch2iso.last_output_path);
        const std::uint64_t desired_request =
            state.canva_manual_iso_npz_path.empty()
            ? state.sketch2iso.completed_request_id
            : 0;
        const std::filesystem::path desired_sketch =
            desired_npz.empty() ? std::filesystem::path() : find_iso_sketch_path(state, desired_npz);

        if (desired_npz.string() == m_sketch2iso_prediction_npz_path
            && desired_sketch.string() == m_sketch2iso_prediction_sketch_path
            && desired_request == m_sketch2iso_prediction_request_id)
            return;

        m_sketch2iso_prediction_npz_path = desired_npz.string();
        m_sketch2iso_prediction_sketch_path = desired_sketch.string();
        m_sketch2iso_prediction_request_id = desired_request;
        m_sketch2iso_prediction_texture_w = 0;
        m_sketch2iso_prediction_texture_h = 0;
        m_sketch2iso_prediction_channels = 0;
        m_sketch2iso_prediction_values.clear();
        m_sketch2iso_prediction_mask.clear();
        if (m_sketch2iso_prediction_texture_id != 0)
        {
            glDeleteTextures(1, &m_sketch2iso_prediction_texture_id);
            m_sketch2iso_prediction_texture_id = 0;
        }

        if (desired_npz.empty())
            return;

        cnpy::NpyArray isomap;
        cnpy::NpyArray mask;
        try
        {
            isomap = cnpy::npz_load(desired_npz.string(), "isomap");
            mask = cnpy::npz_load(desired_npz.string(), "mask");
        }
        catch (const std::exception& e)
        {
            console_add_line(state, "Sketch2ISO preview failed: " + std::string(e.what()));
            return;
        }

        if (isomap.shape.size() != 3
            || (isomap.word_size != sizeof(float) && isomap.word_size != sizeof(double)))
        {
            console_add_line(state, "Sketch2ISO preview failed: isomap must be a float 3D array");
            return;
        }

        const bool hwc = isomap.shape[2] == kIsoDimensions;
        const bool chw = !hwc && isomap.shape[0] == kIsoDimensions;
        if (!hwc && !chw)
        {
            console_add_line(state, "Sketch2ISO preview failed: expected 64 descriptor channels");
            return;
        }

        const int height = (int)(hwc ? isomap.shape[0] : isomap.shape[1]);
        const int width = (int)(hwc ? isomap.shape[1] : isomap.shape[2]);
        if (width <= 0 || height <= 0
            || mask.shape.size() != 2
            || (int)mask.shape[0] != height
            || (int)mask.shape[1] != width)
        {
            console_add_line(state, "Sketch2ISO preview failed: mask dimensions do not match");
            return;
        }

        const size_t pixel_count = (size_t)width * (size_t)height;
        m_sketch2iso_prediction_values.resize(pixel_count * (size_t)kIsoDimensions);
        m_sketch2iso_prediction_mask.resize(pixel_count);
        for (size_t pixel = 0; pixel < pixel_count; ++pixel)
        {
            m_sketch2iso_prediction_mask[pixel] = npy_float_value(mask, pixel);
            for (int channel = 0; channel < kIsoDimensions; ++channel)
            {
                const size_t source = hwc
                    ? pixel * (size_t)kIsoDimensions + (size_t)channel
                    : (size_t)channel * pixel_count + pixel;
                m_sketch2iso_prediction_values[
                    pixel * (size_t)kIsoDimensions + (size_t)channel] =
                    npy_float_value(isomap, source);
            }
        }

        m_sketch2iso_prediction_texture_w = width;
        m_sketch2iso_prediction_texture_h = height;
        m_sketch2iso_prediction_channels = kIsoDimensions;
        (void)ensure_iso_vertex_descriptors(state);

        if (!desired_sketch.empty())
        {
            int source_width = 0;
            int source_height = 0;
            int components = 0;
            stbi_uc* source = stbi_load(
                desired_sketch.string().c_str(),
                &source_width,
                &source_height,
                &components,
                4);
            if (source && source_width > 0 && source_height > 0)
            {
                const std::vector<unsigned char> pixels = resize_rgba_nearest(
                    source,
                    source_width,
                    source_height,
                    width,
                    height);
                stbi_image_free(source);

                glGenTextures(1, &m_sketch2iso_prediction_texture_id);
                glBindTexture(GL_TEXTURE_2D, m_sketch2iso_prediction_texture_id);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGBA8,
                    width,
                    height,
                    0,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    pixels.data());
                glBindTexture(GL_TEXTURE_2D, 0);
            }
            else if (source)
            {
                stbi_image_free(source);
            }
        }

        console_add_line(state, "Sketch2ISO preview ready: " + desired_npz.string());
    }

    Renderer::UvDebugTextureInfo Renderer::sketch2iso_prediction_texture_info() const
    {
        UvDebugTextureInfo info;
        info.texture_id = m_sketch2iso_prediction_texture_id;
        info.width = m_sketch2iso_prediction_texture_w;
        info.height = m_sketch2iso_prediction_texture_h;
        info.valid = m_sketch2iso_prediction_texture_id != 0
            && info.width > 0
            && info.height > 0;
        return info;
    }

    void Renderer::search_iso_kd_node(
        int node_index,
        const float* descriptor,
        int& best_vertex,
        float& best_distance_squared) const
    {
        if (node_index < 0 || node_index >= (int)m_iso_kd_nodes.size())
            return;

        const IsoKdNode& node = m_iso_kd_nodes[(size_t)node_index];
        const float* candidate =
            m_iso_vertex_descriptors.data()
            + (size_t)node.vertex * (size_t)kIsoDimensions;
        float distance_squared = 0.0f;
        for (int channel = 0; channel < kIsoDimensions; ++channel)
        {
            const float delta = descriptor[channel] - candidate[channel];
            distance_squared += delta * delta;
        }
        if (distance_squared < best_distance_squared)
        {
            best_distance_squared = distance_squared;
            best_vertex = node.vertex;
        }

        const float axis_delta = descriptor[node.axis] - candidate[node.axis];
        const int near_node = axis_delta < 0.0f ? node.left : node.right;
        const int far_node = axis_delta < 0.0f ? node.right : node.left;
        search_iso_kd_node(near_node, descriptor, best_vertex, best_distance_squared);
        if (axis_delta * axis_delta < best_distance_squared)
            search_iso_kd_node(far_node, descriptor, best_vertex, best_distance_squared);
    }

    bool Renderer::find_nearest_iso_vertex(
        const float* descriptor,
        int& out_vertex,
        float& out_distance_squared) const
    {
        if (!descriptor
            || m_iso_kd_root < 0
            || m_iso_kd_nodes.empty()
            || m_iso_vertex_descriptor_dimensions != kIsoDimensions)
            return false;

        out_vertex = -1;
        out_distance_squared = std::numeric_limits<float>::infinity();
        search_iso_kd_node(
            m_iso_kd_root,
            descriptor,
            out_vertex,
            out_distance_squared);
        return out_vertex >= 0 && std::isfinite(out_distance_squared);
    }

    bool Renderer::sketch2iso_prediction_match_at(
        int x,
        int y,
        float distance_threshold,
        int& out_vertex,
        float& out_distance) const
    {
        const int width = m_sketch2iso_prediction_texture_w;
        const int height = m_sketch2iso_prediction_texture_h;
        if (x < 0 || y < 0 || x >= width || y >= height
            || m_sketch2iso_prediction_channels != kIsoDimensions
            || m_iso_vertex_descriptor_dimensions != kIsoDimensions
            || m_iso_vertex_descriptor_count <= 0)
            return false;

        const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
        const size_t pixel_count = (size_t)width * (size_t)height;
        if (m_sketch2iso_prediction_mask.size() < pixel_count
            || m_sketch2iso_prediction_values.size() < pixel_count * (size_t)kIsoDimensions
            || !(m_sketch2iso_prediction_mask[pixel] > 0.5f))
            return false;

        const float* descriptor =
            m_sketch2iso_prediction_values.data() + pixel * (size_t)kIsoDimensions;
        float best_distance_squared = 0.0f;
        int best_vertex = -1;
        if (!find_nearest_iso_vertex(
                descriptor,
                best_vertex,
                best_distance_squared))
            return false;

        out_vertex = best_vertex;
        out_distance = std::sqrt(std::max(0.0f, best_distance_squared));
        return out_distance <= std::max(0.0f, distance_threshold);
    }

    void Renderer::process_canva_iso_displacement_map(AppState& state)
    {
        const std::uint64_t request_id = state.canva_iso_displacement_request_id;
        if (request_id == 0 || request_id == m_last_canva_iso_displacement_request_id)
            return;

        m_last_canva_iso_displacement_request_id = request_id;
        state.canva_iso_displacement_completed_id = request_id;
        state.canva_iso_displacement_path.clear();

        const auto fail = [&](const std::string& message)
        {
            console_add_line(state, "Canva ISO displacement failed: " + message);
        };

        if (!state.scene_view_target_valid)
        {
            fail("predicted camera view is not available");
            return;
        }
        if (!m_smplh_loaded || m_smplh_vertices.rows() <= 0)
        {
            fail("SMPL mesh is not ready");
            return;
        }
        if (state.canva_iso_displacement_npz_path.empty())
        {
            fail("missing ISO source path");
            return;
        }

        ensure_sketch2iso_prediction_texture(state);
        if (m_sketch2iso_prediction_npz_path != state.canva_iso_displacement_npz_path
            || m_sketch2iso_prediction_texture_w <= 0
            || m_sketch2iso_prediction_texture_h <= 0
            || m_sketch2iso_prediction_values.empty()
            || !ensure_iso_vertex_descriptors(state))
        {
            fail("ISO source is not ready");
            return;
        }
        if (m_iso_vertex_descriptor_count != m_smplh_vertices.rows())
        {
            fail("ISO vertex table does not match the SMPL mesh");
            return;
        }

        Eigen::Matrix4f view;
        Eigen::Matrix4f proj;
        const auto& core = m_viewer.core();
        if (!make_target_camera_matrices(
                state,
                core.camera_dnear,
                core.camera_dfar,
                view,
                proj))
        {
            fail("could not build predicted camera matrices");
            return;
        }

        const int width = m_sketch2iso_prediction_texture_w;
        const int height = m_sketch2iso_prediction_texture_h;
        const size_t pixel_count = (size_t)width * (size_t)height;
        std::vector<unsigned char> output(pixel_count * 4u, 0u);
        int projected_count = 0;
        int threshold_error_count = 0;
        int foreground_count = 0;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
                if (pixel >= m_sketch2iso_prediction_mask.size()
                    || !(m_sketch2iso_prediction_mask[pixel] > 0.5f))
                    continue;

                ++foreground_count;
                int vertex = -1;
                float distance = 0.0f;
                unsigned char* dst = output.data() + pixel * 4u;
                if (!sketch2iso_prediction_match_at(
                        x,
                        y,
                        state.iso_distance_threshold,
                        vertex,
                        distance))
                {
                    dst[2] = 255u;
                    dst[3] = 255u;
                    ++threshold_error_count;
                    continue;
                }

                if (vertex < 0 || vertex >= m_smplh_vertices.rows())
                    continue;

                float projected_x = 0.0f;
                float projected_y = 0.0f;
                if (!project_to_image(
                        view,
                        proj,
                        m_smplh_vertices.row(vertex).transpose(),
                        width,
                        height,
                        projected_x,
                        projected_y))
                    continue;

                const float dx = projected_x - ((float)x + 0.5f);
                const float dy = projected_y - ((float)y + 0.5f);
                dst[0] = (unsigned char)std::clamp<int>(
                    (int)std::lround(std::abs(dx)),
                    0,
                    255);
                dst[1] = (unsigned char)std::clamp<int>(
                    (int)std::lround(std::abs(dy)),
                    0,
                    255);
                dst[3] = 255u;
                ++projected_count;
            }
        }

        std::error_code ec;
        const std::filesystem::path renders_dir = "renders";
        std::filesystem::create_directories(renders_dir, ec);
        if (ec)
        {
            fail("could not create renders directory: " + ec.message());
            return;
        }

        const std::filesystem::path output_path =
            renders_dir / "canva_iso_displacement.png";
        if (stbi_write_png(
                output_path.string().c_str(),
                width,
                height,
                4,
                output.data(),
                width * 4) == 0)
        {
            fail("could not write " + output_path.string());
            return;
        }

        state.canva_iso_displacement_path = output_path.string();
        console_add_line(
            state,
            "Canva ISO displacement saved: " + output_path.string()
                + " (projected=" + std::to_string(projected_count)
                + ", threshold_errors=" + std::to_string(threshold_error_count)
                + "/" + std::to_string(foreground_count) + ")");
    }
}
