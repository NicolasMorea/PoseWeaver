#include "core/GlbGarment.h"

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace poseweaver
{
    namespace
    {
        struct CgltfDeleter
        {
            void operator()(cgltf_data* data) const { cgltf_free(data); }
        };

        using CgltfPtr = std::unique_ptr<cgltf_data, CgltfDeleter>;

        const cgltf_accessor* find_attribute(
            const cgltf_primitive& primitive,
            cgltf_attribute_type type,
            int index)
        {
            for (cgltf_size i = 0; i < primitive.attributes_count; ++i)
            {
                const cgltf_attribute& attribute = primitive.attributes[i];
                if (attribute.type == type && attribute.index == index)
                    return attribute.data;
            }
            return nullptr;
        }

        bool unpack_floats(
            const cgltf_accessor* accessor,
            int components,
            std::vector<float>& values)
        {
            if (!accessor || static_cast<int>(cgltf_num_components(accessor->type)) < components)
                return false;
            const cgltf_size value_count = accessor->count * static_cast<cgltf_size>(components);
            values.resize(static_cast<std::size_t>(value_count));
            return cgltf_accessor_unpack_floats(accessor, values.data(), value_count) == value_count;
        }

        std::vector<std::uint32_t> unpack_indices(const cgltf_primitive& primitive, std::size_t vertex_count)
        {
            const std::size_t index_count = primitive.indices
                ? static_cast<std::size_t>(primitive.indices->count)
                : vertex_count;
            std::vector<std::uint32_t> indices(index_count);
            if (primitive.indices)
            {
                for (std::size_t i = 0; i < index_count; ++i)
                    indices[i] = static_cast<std::uint32_t>(cgltf_accessor_read_index(primitive.indices, i));
            }
            else
            {
                for (std::size_t i = 0; i < index_count; ++i)
                    indices[i] = static_cast<std::uint32_t>(i);
            }
            return indices;
        }

        void append_triangles(
            cgltf_primitive_type type,
            const std::vector<std::uint32_t>& indices,
            int vertex_offset,
            std::vector<Eigen::Vector3i>& faces)
        {
            const auto append = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c)
            {
                if (a != b && b != c && c != a)
                    faces.emplace_back(vertex_offset + static_cast<int>(a),
                        vertex_offset + static_cast<int>(b),
                        vertex_offset + static_cast<int>(c));
            };

            if (type == cgltf_primitive_type_triangles)
            {
                for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
                    append(indices[i], indices[i + 1], indices[i + 2]);
            }
            else if (type == cgltf_primitive_type_triangle_strip)
            {
                for (std::size_t i = 2; i < indices.size(); ++i)
                {
                    if ((i & 1u) == 0u)
                        append(indices[i - 2], indices[i - 1], indices[i]);
                    else
                        append(indices[i - 1], indices[i - 2], indices[i]);
                }
            }
            else if (type == cgltf_primitive_type_triangle_fan)
            {
                for (std::size_t i = 2; i < indices.size(); ++i)
                    append(indices[0], indices[i - 1], indices[i]);
            }
        }

        const cgltf_texture_view* base_color_texture(const cgltf_material* material)
        {
            if (!material)
                return nullptr;
            if (material->has_pbr_metallic_roughness
                && material->pbr_metallic_roughness.base_color_texture.texture)
            {
                return &material->pbr_metallic_roughness.base_color_texture;
            }
            if (material->has_pbr_specular_glossiness
                && material->pbr_specular_glossiness.diffuse_texture.texture)
            {
                return &material->pbr_specular_glossiness.diffuse_texture;
            }
            return nullptr;
        }

        cgltf_image* texture_image(const cgltf_texture_view* texture_view)
        {
            if (!texture_view || !texture_view->texture)
                return nullptr;
            cgltf_texture* texture = texture_view->texture;
            if (texture->has_basisu && texture->basisu_image)
                return texture->basisu_image;
            if (texture->has_webp && texture->webp_image)
                return texture->webp_image;
            return texture->image;
        }

        bool decode_image(
            const std::filesystem::path& glb_path,
            cgltf_image* image,
            std::vector<unsigned char>& rgba,
            int& width,
            int& height)
        {
            if (!image)
                return false;

            int channels = 0;
            unsigned char* pixels = nullptr;
            if (image->buffer_view && image->buffer_view->buffer)
            {
                const cgltf_buffer_view* view = image->buffer_view;
                const unsigned char* bytes = view->data
                    ? static_cast<const unsigned char*>(view->data)
                    : static_cast<const unsigned char*>(view->buffer->data) + view->offset;
                if (!bytes || view->size == 0 || view->size > static_cast<cgltf_size>(INT32_MAX))
                    return false;
                pixels = stbi_load_from_memory(
                    bytes,
                    static_cast<int>(view->size),
                    &width,
                    &height,
                    &channels,
                    STBI_rgb_alpha);
            }
            else if (image->uri && std::strncmp(image->uri, "data:", 5) != 0)
            {
                std::string uri(image->uri);
                cgltf_decode_uri(uri.data());
                const std::filesystem::path image_path = glb_path.parent_path() / uri;
                pixels = stbi_load(image_path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
            }

            if (!pixels || width <= 0 || height <= 0)
            {
                if (pixels)
                    stbi_image_free(pixels);
                return false;
            }

            const std::size_t byte_count = static_cast<std::size_t>(width)
                * static_cast<std::size_t>(height) * 4u;
            rgba.assign(pixels, pixels + byte_count);
            stbi_image_free(pixels);
            return true;
        }

        Eigen::Vector2d transformed_uv(const Eigen::Vector2d& uv, const cgltf_texture_view* texture_view)
        {
            if (!texture_view || !texture_view->has_transform)
                return uv;
            const cgltf_texture_transform& transform = texture_view->transform;
            const double c = std::cos(static_cast<double>(transform.rotation));
            const double s = std::sin(static_cast<double>(transform.rotation));
            const double x = uv.x() * static_cast<double>(transform.scale[0]);
            const double y = uv.y() * static_cast<double>(transform.scale[1]);
            return Eigen::Vector2d(
                static_cast<double>(transform.offset[0]) + c * x - s * y,
                static_cast<double>(transform.offset[1]) + s * x + c * y);
        }

        Eigen::RowVector3d base_color(const cgltf_material* material)
        {
            Eigen::RowVector3d color = Eigen::RowVector3d::Ones();
            if (!material)
                return color;

            const cgltf_float* factor = nullptr;
            if (material->has_pbr_metallic_roughness)
                factor = material->pbr_metallic_roughness.base_color_factor;
            else if (material->has_pbr_specular_glossiness)
                factor = material->pbr_specular_glossiness.diffuse_factor;
            if (!factor)
                return color;
            color << factor[0], factor[1], factor[2];
            return color.cwiseMax(0.0).cwiseMin(1.0);
        }
    }

    bool load_gltf_garment(const std::filesystem::path& path, GltfGarment& out, std::string* error)
    {
        out = GltfGarment{};

        cgltf_options options{};
        cgltf_data* raw_data = nullptr;
        const cgltf_result parse_result = cgltf_parse_file(&options, path.string().c_str(), &raw_data);
        CgltfPtr data(raw_data);
        if (parse_result != cgltf_result_success || !data)
        {
            if (error)
                *error = "Could not parse the glTF garment file.";
            return false;
        }
        if (cgltf_load_buffers(&options, data.get(), path.string().c_str()) != cgltf_result_success)
        {
            if (error)
                *error = "Could not load the glTF buffers.";
            return false;
        }

        cgltf_image* selected_image = nullptr;
        for (cgltf_size ni = 0; ni < data->nodes_count && !selected_image; ++ni)
        {
            const cgltf_node& node = data->nodes[ni];
            if (!node.mesh)
                continue;
            for (cgltf_size pi = 0; pi < node.mesh->primitives_count; ++pi)
            {
                const cgltf_primitive& primitive = node.mesh->primitives[pi];
                const cgltf_texture_view* candidate = base_color_texture(primitive.material);
                cgltf_image* image = texture_image(candidate);
                if (candidate && image)
                {
                    selected_image = image;
                    break;
                }
            }
        }

        if (selected_image)
        {
            if (!decode_image(path, selected_image, out.texture_rgba, out.texture_width, out.texture_height))
            {
                out.texture_warning = "The glTF mesh loaded, but its base-color texture could not be decoded.";
                out.texture_rgba.clear();
                out.texture_width = 0;
                out.texture_height = 0;
            }
        }

        std::vector<Eigen::Vector3d> vertices;
        std::vector<Eigen::Vector2d> uv;
        std::vector<Eigen::Vector3i> faces;
        std::vector<Eigen::Vector3i> uv_faces;
        std::vector<Eigen::RowVector3d> face_colors;
        bool incompatible_texture = false;

        for (cgltf_size ni = 0; ni < data->nodes_count; ++ni)
        {
            const cgltf_node& node = data->nodes[ni];
            if (!node.mesh)
                continue;

            cgltf_float world[16];
            cgltf_node_transform_world(&node, world);

            for (cgltf_size pi = 0; pi < node.mesh->primitives_count; ++pi)
            {
                const cgltf_primitive& primitive = node.mesh->primitives[pi];
                if (primitive.type != cgltf_primitive_type_triangles
                    && primitive.type != cgltf_primitive_type_triangle_strip
                    && primitive.type != cgltf_primitive_type_triangle_fan)
                {
                    continue;
                }

                const cgltf_accessor* position_accessor = find_attribute(primitive, cgltf_attribute_type_position, 0);
                std::vector<float> positions;
                if (!unpack_floats(position_accessor, 3, positions) || !position_accessor)
                    continue;

                const cgltf_texture_view* primitive_texture = base_color_texture(primitive.material);
                const bool uses_selected_texture = selected_image
                    && texture_image(primitive_texture) == selected_image;
                if (selected_image && primitive_texture && !uses_selected_texture)
                    incompatible_texture = true;

                int texcoord_index = primitive_texture ? primitive_texture->texcoord : 0;
                if (primitive_texture && primitive_texture->has_transform
                    && primitive_texture->transform.has_texcoord)
                {
                    texcoord_index = primitive_texture->transform.texcoord;
                }
                const cgltf_accessor* uv_accessor = find_attribute(
                    primitive,
                    cgltf_attribute_type_texcoord,
                    texcoord_index);
                std::vector<float> primitive_uv;
                const bool has_uv = unpack_floats(uv_accessor, 2, primitive_uv)
                    && uv_accessor
                    && uv_accessor->count == position_accessor->count;

                const int vertex_offset = static_cast<int>(vertices.size());
                const std::size_t vertex_count = static_cast<std::size_t>(position_accessor->count);
                vertices.reserve(vertices.size() + vertex_count);
                uv.reserve(uv.size() + vertex_count);
                for (std::size_t vi = 0; vi < vertex_count; ++vi)
                {
                    const double x = positions[vi * 3u];
                    const double y = positions[vi * 3u + 1u];
                    const double z = positions[vi * 3u + 2u];
                    vertices.emplace_back(
                        world[0] * x + world[4] * y + world[8] * z + world[12],
                        world[1] * x + world[5] * y + world[9] * z + world[13],
                        world[2] * x + world[6] * y + world[10] * z + world[14]);

                    const Eigen::Vector2d source_uv = has_uv
                        ? Eigen::Vector2d(primitive_uv[vi * 2u], primitive_uv[vi * 2u + 1u])
                        : Eigen::Vector2d::Zero();
                    uv.push_back(transformed_uv(source_uv, primitive_texture));
                }

                const std::size_t face_begin = faces.size();
                append_triangles(
                    primitive.type,
                    unpack_indices(primitive, vertex_count),
                    vertex_offset,
                    faces);
                for (std::size_t fi = face_begin; fi < faces.size(); ++fi)
                {
                    uv_faces.push_back(faces[fi]);
                    face_colors.push_back(base_color(primitive.material));
                }

                if (primitive.material)
                {
                    out.has_material = true;
                    if (primitive.material->double_sided)
                        out.double_sided = true;
                }
                if (selected_image && (!uses_selected_texture || !has_uv))
                    incompatible_texture = true;
            }
        }

        if (vertices.empty() || faces.empty())
        {
            if (error)
                *error = "The glTF file does not contain a supported triangle mesh.";
            return false;
        }

        out.vertices.resize(static_cast<int>(vertices.size()), 3);
        out.uv.resize(static_cast<int>(uv.size()), 2);
        for (int i = 0; i < out.vertices.rows(); ++i)
        {
            out.vertices.row(i) = vertices[static_cast<std::size_t>(i)];
            out.uv.row(i) = uv[static_cast<std::size_t>(i)];
        }
        out.faces.resize(static_cast<int>(faces.size()), 3);
        out.uv_faces.resize(static_cast<int>(uv_faces.size()), 3);
        out.face_colors.resize(static_cast<int>(face_colors.size()), 3);
        for (int i = 0; i < out.faces.rows(); ++i)
        {
            out.faces.row(i) = faces[static_cast<std::size_t>(i)];
            out.uv_faces.row(i) = uv_faces[static_cast<std::size_t>(i)];
            out.face_colors.row(i) = face_colors[static_cast<std::size_t>(i)];
        }

        if (incompatible_texture && out.has_image_texture())
        {
            out.texture_warning = "The glTF file uses multiple or partially mapped materials; the first base-color texture is used.";
        }
        return true;
    }
}
