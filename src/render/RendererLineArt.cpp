#include "render/RendererLineArt.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace poseweaver
{
    namespace
    {
        struct MeshEdge
        {
            int a = -1;
            int b = -1;
            int face_a = -1;
            int face_b = -1;
        };

        struct ScreenVertex
        {
            float x = 0.0f;
            float y = 0.0f;
            float depth = 1.0f;
            bool valid = false;
        };

        static unsigned long long edge_key(int a, int b)
        {
            const unsigned int lo = (unsigned int)std::min(a, b);
            const unsigned int hi = (unsigned int)std::max(a, b);
            return (static_cast<unsigned long long>(lo) << 32u) | hi;
        }

        static std::vector<MeshEdge> build_edges(
            const Eigen::MatrixXi& faces,
            int vertex_count,
            std::vector<std::vector<int>>& adjacency)
        {
            adjacency.assign((size_t)vertex_count, {});
            std::vector<MeshEdge> edges;
            std::unordered_map<unsigned long long, int> edge_indices;
            edge_indices.reserve((size_t)faces.rows() * 3u);

            for (int face = 0; face < faces.rows(); ++face)
            {
                for (int corner = 0; corner < 3; ++corner)
                {
                    const int a = faces(face, corner);
                    const int b = faces(face, (corner + 1) % 3);
                    if (a < 0 || b < 0 || a >= vertex_count || b >= vertex_count || a == b)
                        continue;

                    const unsigned long long key = edge_key(a, b);
                    const auto found = edge_indices.find(key);
                    if (found == edge_indices.end())
                    {
                        MeshEdge edge;
                        edge.a = std::min(a, b);
                        edge.b = std::max(a, b);
                        edge.face_a = face;
                        edge_indices.emplace(key, (int)edges.size());
                        edges.push_back(edge);
                        adjacency[(size_t)a].push_back(b);
                        adjacency[(size_t)b].push_back(a);
                    }
                    else
                    {
                        MeshEdge& edge = edges[(size_t)found->second];
                        if (edge.face_b < 0)
                            edge.face_b = face;
                    }
                }
            }
            return edges;
        }

        static Eigen::MatrixXd smooth_for_line_art(
            const Eigen::MatrixXd& vertices,
            const std::vector<MeshEdge>& edges,
            const std::vector<std::vector<int>>& adjacency)
        {
            std::vector<unsigned char> boundary((size_t)vertices.rows(), 0u);
            for (const MeshEdge& edge : edges)
            {
                if (edge.face_b >= 0)
                    continue;
                boundary[(size_t)edge.a] = 1u;
                boundary[(size_t)edge.b] = 1u;
            }

            Eigen::MatrixXd smoothed = vertices;
            for (int iteration = 0; iteration < 5; ++iteration)
            {
                Eigen::MatrixXd next = smoothed;
                for (int vertex = 0; vertex < smoothed.rows(); ++vertex)
                {
                    const auto& neighbors = adjacency[(size_t)vertex];
                    if (boundary[(size_t)vertex] != 0u || neighbors.empty())
                        continue;
                    Eigen::RowVector3d average = Eigen::RowVector3d::Zero();
                    for (int neighbor : neighbors)
                        average += smoothed.row(neighbor);
                    average /= (double)neighbors.size();
                    next.row(vertex) += 0.42 * (average - smoothed.row(vertex));
                }
                smoothed = std::move(next);
            }
            return smoothed;
        }

        static std::vector<Eigen::RowVector3d> face_normals(
            const Eigen::MatrixXd& vertices,
            const Eigen::MatrixXi& faces)
        {
            std::vector<Eigen::RowVector3d> normals(
                (size_t)faces.rows(),
                Eigen::RowVector3d::Zero());
            for (int face = 0; face < faces.rows(); ++face)
            {
                const Eigen::RowVector3d ab =
                    vertices.row(faces(face, 1)) - vertices.row(faces(face, 0));
                const Eigen::RowVector3d ac =
                    vertices.row(faces(face, 2)) - vertices.row(faces(face, 0));
                Eigen::RowVector3d normal = ab.cross(ac);
                const double length = normal.norm();
                if (length > 1e-12 && std::isfinite(length))
                    normals[(size_t)face] = normal / length;
            }
            return normals;
        }

        static std::vector<ScreenVertex> project_vertices(
            const Eigen::MatrixXd& vertices,
            const Eigen::Matrix4f& view,
            const Eigen::Matrix4f& projection,
            int width,
            int height)
        {
            std::vector<ScreenVertex> projected((size_t)vertices.rows());
            const Eigen::Matrix4f transform = projection * view;
            for (int vertex = 0; vertex < vertices.rows(); ++vertex)
            {
                const Eigen::Vector4f point(
                    (float)vertices(vertex, 0),
                    (float)vertices(vertex, 1),
                    (float)vertices(vertex, 2),
                    1.0f);
                const Eigen::Vector4f clip = transform * point;
                if (!clip.allFinite() || clip.w() <= 1e-6f)
                    continue;
                const Eigen::Vector3f ndc = clip.head<3>() / clip.w();
                ScreenVertex& screen = projected[(size_t)vertex];
                screen.x = (ndc.x() * 0.5f + 0.5f) * (float)(width - 1);
                screen.y = (0.5f - ndc.y() * 0.5f) * (float)(height - 1);
                screen.depth = ndc.z() * 0.5f + 0.5f;
                screen.valid = ndc.z() >= -1.02f && ndc.z() <= 1.02f;
            }
            return projected;
        }

        static float segment_length(const ScreenVertex& a, const ScreenVertex& b)
        {
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            return std::sqrt(dx * dx + dy * dy);
        }

        static std::vector<unsigned char> select_crease_edges(
            const std::vector<MeshEdge>& edges,
            const std::vector<Eigen::RowVector3d>& normals,
            const std::vector<ScreenVertex>& projected,
            float crease_angle_degrees)
        {
            constexpr float kPi = 3.14159265358979323846f;
            const float threshold = std::clamp(crease_angle_degrees, 10.0f, 80.0f) * kPi / 180.0f;
            std::vector<unsigned char> selected(edges.size(), 0u);
            std::vector<std::vector<int>> incident(projected.size());

            for (int edge_index = 0; edge_index < (int)edges.size(); ++edge_index)
            {
                const MeshEdge& edge = edges[(size_t)edge_index];
                if (edge.face_a < 0 || edge.face_b < 0)
                    continue;
                const Eigen::RowVector3d& a = normals[(size_t)edge.face_a];
                const Eigen::RowVector3d& b = normals[(size_t)edge.face_b];
                if (a.squaredNorm() < 0.5 || b.squaredNorm() < 0.5)
                    continue;
                const double cosine = std::clamp(a.dot(b), -1.0, 1.0);
                if ((float)std::acos(cosine) < threshold)
                    continue;
                const ScreenVertex& screen_a = projected[(size_t)edge.a];
                const ScreenVertex& screen_b = projected[(size_t)edge.b];
                if (!screen_a.valid || !screen_b.valid || segment_length(screen_a, screen_b) < 0.75f)
                    continue;
                selected[(size_t)edge_index] = 1u;
                incident[(size_t)edge.a].push_back(edge_index);
                incident[(size_t)edge.b].push_back(edge_index);
            }

            std::vector<unsigned char> visited(edges.size(), 0u);
            std::vector<int> stack;
            std::vector<int> component;
            for (int seed = 0; seed < (int)edges.size(); ++seed)
            {
                if (selected[(size_t)seed] == 0u || visited[(size_t)seed] != 0u)
                    continue;
                stack.clear();
                component.clear();
                stack.push_back(seed);
                visited[(size_t)seed] = 1u;
                float total_length = 0.0f;

                while (!stack.empty())
                {
                    const int edge_index = stack.back();
                    stack.pop_back();
                    component.push_back(edge_index);
                    const MeshEdge& edge = edges[(size_t)edge_index];
                    total_length += segment_length(
                        projected[(size_t)edge.a],
                        projected[(size_t)edge.b]);
                    for (int vertex : { edge.a, edge.b })
                    {
                        for (int next : incident[(size_t)vertex])
                        {
                            if (visited[(size_t)next] == 0u)
                            {
                                visited[(size_t)next] = 1u;
                                stack.push_back(next);
                            }
                        }
                    }
                }

                if (total_length >= 14.0f)
                    continue;
                for (int edge_index : component)
                    selected[(size_t)edge_index] = 0u;
            }
            return selected;
        }

        static float pixel_luminance(const unsigned char* pixel)
        {
            return (
                0.2126f * (float)pixel[0]
                + 0.7152f * (float)pixel[1]
                + 0.0722f * (float)pixel[2]) / 255.0f;
        }

        static std::vector<float> blur_masked_tone(
            const std::vector<float>& source,
            const std::vector<unsigned char>& mask,
            int width,
            int height,
            int radius)
        {
            std::vector<float> horizontal(source.size(), 1.0f);
            std::vector<float> result(source.size(), 1.0f);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t index = (size_t)y * (size_t)width + (size_t)x;
                    if (mask[index] == 0u)
                        continue;
                    float sum = 0.0f;
                    int samples = 0;
                    for (int dx = -radius; dx <= radius; ++dx)
                    {
                        const int sample_x = std::clamp(x + dx, 0, width - 1);
                        const size_t sample = (size_t)y * (size_t)width + (size_t)sample_x;
                        if (mask[sample] != 0u)
                        {
                            sum += source[sample];
                            samples++;
                        }
                    }
                    horizontal[index] = samples > 0 ? sum / (float)samples : source[index];
                }
            }
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t index = (size_t)y * (size_t)width + (size_t)x;
                    if (mask[index] == 0u)
                        continue;
                    float sum = 0.0f;
                    int samples = 0;
                    for (int dy = -radius; dy <= radius; ++dy)
                    {
                        const int sample_y = std::clamp(y + dy, 0, height - 1);
                        const size_t sample = (size_t)sample_y * (size_t)width + (size_t)x;
                        if (mask[sample] != 0u)
                        {
                            sum += horizontal[sample];
                            samples++;
                        }
                    }
                    result[index] = samples > 0 ? sum / (float)samples : horizontal[index];
                }
            }
            return result;
        }

        static std::vector<float> boundary_distance(
            const std::vector<unsigned char>& mask,
            int width,
            int height)
        {
            const float infinity = std::numeric_limits<float>::max() / 4.0f;
            std::vector<float> distance(mask.size(), infinity);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t index = (size_t)y * (size_t)width + (size_t)x;
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        const int sy = y + dy;
                        if (sy < 0 || sy >= height)
                            continue;
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            const int sx = x + dx;
                            if (sx < 0 || sx >= width || (dx == 0 && dy == 0))
                                continue;
                            if (mask[(size_t)sy * (size_t)width + (size_t)sx] != mask[index])
                                distance[index] = 0.0f;
                        }
                    }
                }
            }

            constexpr float diagonal = 1.41421356f;
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t index = (size_t)y * (size_t)width + (size_t)x;
                    if (x > 0)
                        distance[index] = std::min(distance[index], distance[index - 1u] + 1.0f);
                    if (y > 0)
                    {
                        distance[index] = std::min(distance[index], distance[index - (size_t)width] + 1.0f);
                        if (x > 0)
                            distance[index] = std::min(distance[index], distance[index - (size_t)width - 1u] + diagonal);
                        if (x + 1 < width)
                            distance[index] = std::min(distance[index], distance[index - (size_t)width + 1u] + diagonal);
                    }
                }
            }
            for (int y = height - 1; y >= 0; --y)
            {
                for (int x = width - 1; x >= 0; --x)
                {
                    const size_t index = (size_t)y * (size_t)width + (size_t)x;
                    if (x + 1 < width)
                        distance[index] = std::min(distance[index], distance[index + 1u] + 1.0f);
                    if (y + 1 < height)
                    {
                        distance[index] = std::min(distance[index], distance[index + (size_t)width] + 1.0f);
                        if (x > 0)
                            distance[index] = std::min(distance[index], distance[index + (size_t)width - 1u] + diagonal);
                        if (x + 1 < width)
                            distance[index] = std::min(distance[index], distance[index + (size_t)width + 1u] + diagonal);
                    }
                }
            }
            return distance;
        }

        static void blend_ink(
            unsigned char* pixel,
            const std::array<float, 3>& line_color,
            float coverage)
        {
            coverage = std::clamp(coverage, 0.0f, 1.0f);
            for (int channel = 0; channel < 3; ++channel)
            {
                const float ink = 255.0f * std::clamp(line_color[(size_t)channel], 0.0f, 1.0f);
                pixel[channel] = (unsigned char)std::clamp(
                    (int)std::lround((1.0f - coverage) * (float)pixel[channel] + coverage * ink),
                    0,
                    255);
            }
            pixel[3] = std::max(
                pixel[3],
                (unsigned char)std::clamp((int)std::lround(255.0f * coverage), 0, 255));
        }

        static void draw_crease_edge(
            std::vector<unsigned char>& rgba,
            int width,
            int height,
            const std::vector<float>& gl_depth,
            const ScreenVertex& a,
            const ScreenVertex& b,
            float line_width,
            const std::array<float, 3>& line_color)
        {
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            const float length_squared = dx * dx + dy * dy;
            if (length_squared < 0.25f)
                return;
            const float radius = std::max(0.5f, line_width * 0.5f);
            const int min_x = std::max(0, (int)std::floor(std::min(a.x, b.x) - radius - 1.0f));
            const int max_x = std::min(width - 1, (int)std::ceil(std::max(a.x, b.x) + radius + 1.0f));
            const int min_y = std::max(0, (int)std::floor(std::min(a.y, b.y) - radius - 1.0f));
            const int max_y = std::min(height - 1, (int)std::ceil(std::max(a.y, b.y) + radius + 1.0f));

            for (int y = min_y; y <= max_y; ++y)
            {
                for (int x = min_x; x <= max_x; ++x)
                {
                    const float px = (float)x + 0.5f;
                    const float py = (float)y + 0.5f;
                    const float t = std::clamp(((px - a.x) * dx + (py - a.y) * dy) / length_squared, 0.0f, 1.0f);
                    const float closest_x = a.x + t * dx;
                    const float closest_y = a.y + t * dy;
                    const float distance_x = px - closest_x;
                    const float distance_y = py - closest_y;
                    const float distance = std::sqrt(distance_x * distance_x + distance_y * distance_y);
                    const float coverage = std::clamp(radius + 0.75f - distance, 0.0f, 1.0f);
                    if (coverage <= 0.0f)
                        continue;

                    const float line_depth = a.depth + t * (b.depth - a.depth);
                    const size_t gl_index = (size_t)(height - 1 - y) * (size_t)width + (size_t)x;
                    if (gl_depth.size() == (size_t)width * (size_t)height
                        && line_depth > gl_depth[gl_index] + 0.0015f)
                        continue;
                    blend_ink(
                        rgba.data() + ((size_t)y * (size_t)width + (size_t)x) * 4u,
                        line_color,
                        0.88f * coverage);
                }
            }
        }
    }

    void apply_grease_pencil_line_art(
        std::vector<unsigned char>& rgba,
        int width,
        int height,
        bool shaded,
        const GreasePencilRenderSettings& settings,
        const Eigen::MatrixXd& vertices,
        const Eigen::MatrixXi& faces,
        const Eigen::Matrix4f& view,
        const Eigen::Matrix4f& projection,
        const std::vector<float>& gl_depth,
        const std::vector<unsigned char>& shading_rgba)
    {
        const size_t pixel_count = (size_t)width * (size_t)height;
        if (width < 3 || height < 3 || rgba.size() < pixel_count * 4u
            || vertices.rows() <= 0 || vertices.cols() != 3
            || faces.rows() <= 0 || faces.cols() != 3)
            return;

        std::vector<unsigned char> mask(pixel_count, 0u);
        std::vector<float> tone(pixel_count, 1.0f);
        std::array<int, 256> histogram{};
        int opaque_count = 0;
        const bool has_shading_reference = shading_rgba.size() >= pixel_count * 4u;
        for (size_t index = 0; index < pixel_count; ++index)
        {
            unsigned char* pixel = rgba.data() + index * 4u;
            if (pixel[3] < 32u)
                continue;
            mask[index] = 1u;
            tone[index] = pixel_luminance(has_shading_reference
                ? shading_rgba.data() + index * 4u
                : pixel);
            if (pixel[3] >= 128u)
                opaque_count++;
        }
        if (opaque_count == 0)
            return;

        const std::vector<float> shaded_tone = shaded
            ? blur_masked_tone(tone, mask, width, height, 18)
            : tone;
        for (size_t index = 0; index < pixel_count; ++index)
        {
            if (mask[index] != 0u && rgba[index * 4u + 3u] >= 128u)
            {
                histogram[(size_t)std::clamp(
                    (int)std::lround(255.0f * shaded_tone[index]),
                    0,
                    255)]++;
            }
        }

        const auto percentile = [&](float fraction)
        {
            const int target = std::clamp((int)std::lround(fraction * (float)opaque_count), 0, opaque_count - 1);
            int accumulated = 0;
            for (int value = 0; value < 256; ++value)
            {
                accumulated += histogram[(size_t)value];
                if (accumulated > target)
                    return (float)value / 255.0f;
            }
            return 1.0f;
        };
        const float dark_tone = percentile(0.06f);
        const float light_tone = percentile(0.96f);
        const float tone_range = std::max(0.08f, light_tone - dark_tone);
        const float shadow_start = std::clamp(settings.shadow_threshold, 0.35f, 0.95f);
        const float shadow_range = std::clamp(settings.shadow_range, 0.05f, 0.70f);
        const float shadow_strength = std::clamp(settings.shadow_strength, 0.0f, 1.0f);
        std::vector<float> shadow_amount(pixel_count, 0.0f);

        for (size_t index = 0; index < pixel_count; ++index)
        {
            unsigned char* pixel = rgba.data() + index * 4u;
            if (mask[index] == 0u)
            {
                pixel[0] = 255u;
                pixel[1] = 255u;
                pixel[2] = 255u;
                pixel[3] = 0u;
                continue;
            }

            float value = 0.985f;
            if (shaded)
            {
                const float normalized = std::clamp((shaded_tone[index] - dark_tone) / tone_range, 0.0f, 1.0f);
                const float shadow = std::clamp((shadow_start - normalized) / shadow_range, 0.0f, 1.0f);
                shadow_amount[index] = shadow;
                const float cel = shadow > 0.68f ? 0.80f : (shadow > 0.18f ? 0.91f : 0.985f);
                value = 0.985f + (cel - 0.985f) * shadow_strength;
            }
            const float shade = shaded ? value / 0.985f : 1.0f;
            pixel[0] = (unsigned char)std::clamp((int)std::lround((float)pixel[0] * shade), 0, 255);
            pixel[1] = (unsigned char)std::clamp((int)std::lround((float)pixel[1] * shade), 0, 255);
            pixel[2] = (unsigned char)std::clamp((int)std::lround((float)pixel[2] * shade), 0, 255);
            pixel[3] = std::max(pixel[3], (unsigned char)224u);
        }

        const std::vector<float> distance = boundary_distance(mask, width, height);
        const float outline_radius = std::max(0.0f, settings.outline_width) * 0.5f;
        if (shaded)
        {
            const auto line_coverage = [](float coordinate, float spacing, float width)
            {
                float phase = std::fmod(coordinate, spacing);
                if (phase < 0.0f)
                    phase += spacing;
                const float line_distance = std::min(phase, spacing - phase);
                return std::clamp(width + 0.75f - line_distance, 0.0f, 1.0f);
            };
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const size_t index = (size_t)y * (size_t)width + (size_t)x;
                    const float shadow = shadow_amount[index];
                    if (mask[index] == 0u || shadow <= 0.10f
                        || distance[index] <= outline_radius + 1.0f)
                        continue;

                    const float primary = line_coverage((float)x + 0.62f * (float)y, 13.0f, 0.65f);
                    const float primary_strength = std::clamp((shadow - 0.10f) / 0.62f, 0.0f, 1.0f);
                    const float coverage = primary * primary_strength * (0.28f + 0.42f * shadow_strength);
                    if (coverage > 0.0f)
                        blend_ink(rgba.data() + index * 4u, settings.line_color, coverage);
                }
            }
        }
        if (outline_radius > 0.0f)
        {
            for (size_t index = 0; index < pixel_count; ++index)
            {
                const float coverage = std::clamp(outline_radius + 0.75f - distance[index], 0.0f, 1.0f);
                if (coverage > 0.0f)
                    blend_ink(rgba.data() + index * 4u, settings.line_color, coverage);
            }
        }

        if (settings.wrinkle_width <= 0.0f)
            return;
        std::vector<std::vector<int>> adjacency;
        const std::vector<MeshEdge> edges = build_edges(faces, vertices.rows(), adjacency);
        const Eigen::MatrixXd smooth_vertices = smooth_for_line_art(vertices, edges, adjacency);
        const std::vector<Eigen::RowVector3d> normals = face_normals(smooth_vertices, faces);
        const std::vector<ScreenVertex> projected = project_vertices(
            vertices,
            view,
            projection,
            width,
            height);
        const std::vector<unsigned char> selected = select_crease_edges(
            edges,
            normals,
            projected,
            shaded ? 0.55f * settings.crease_angle_degrees : settings.crease_angle_degrees);
        for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index)
        {
            if (selected[edge_index] == 0u)
                continue;
            const MeshEdge& edge = edges[edge_index];
            draw_crease_edge(
                rgba,
                width,
                height,
                gl_depth,
                projected[(size_t)edge.a],
                projected[(size_t)edge.b],
                shaded ? 1.2f * settings.wrinkle_width : settings.wrinkle_width,
                settings.line_color);
        }
    }
}
