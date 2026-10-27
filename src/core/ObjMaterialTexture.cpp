#include "core/ObjMaterialTexture.h"

#include <igl/readOBJ.h>
#include <stb_image.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace poseweaver
{
    namespace
    {
        std::string trim(std::string value)
        {
            const auto first = std::find_if_not(value.begin(), value.end(),
                [](unsigned char c) { return std::isspace(c) != 0; });
            const auto last = std::find_if_not(value.rbegin(), value.rend(),
                [](unsigned char c) { return std::isspace(c) != 0; }).base();
            if (first >= last)
                return {};
            return std::string(first, last);
        }

        std::string directive_value(const std::string& line, const char* directive)
        {
            const std::string clean = trim(line);
            const std::size_t length = std::char_traits<char>::length(directive);
            if (clean.size() <= length || clean.compare(0, length, directive) != 0
                || std::isspace(static_cast<unsigned char>(clean[length])) == 0)
            {
                return {};
            }
            return trim(clean.substr(length));
        }

        std::filesystem::path referenced_mtl(const std::filesystem::path& obj_path)
        {
            std::ifstream input(obj_path);
            std::string line;
            while (std::getline(input, line))
            {
                const std::string value = directive_value(line, "mtllib");
                if (!value.empty())
                    return obj_path.parent_path() / value;
            }
            return {};
        }

        struct Material
        {
            Eigen::RowVector3d diffuse{ 0.8, 0.8, 0.8 };
            std::filesystem::path texture;
        };

        using MaterialMap = std::unordered_map<std::string, Material>;

        bool load_materials(
            const std::filesystem::path& mtl_path,
            MaterialMap& materials,
            std::vector<std::string>& order)
        {
            std::ifstream input(mtl_path);
            if (!input)
                return false;

            Material* current = nullptr;
            std::string line;
            while (std::getline(input, line))
            {
                const std::string material_name = directive_value(line, "newmtl");
                if (!material_name.empty())
                {
                    order.push_back(material_name);
                    current = &materials[material_name];
                    continue;
                }
                if (!current)
                    continue;

                const std::string diffuse = directive_value(line, "Kd");
                if (!diffuse.empty())
                {
                    std::istringstream values(diffuse);
                    values >> current->diffuse.x() >> current->diffuse.y() >> current->diffuse.z();
                    current->diffuse = current->diffuse.cwiseMax(0.0).cwiseMin(1.0);
                    continue;
                }

                std::string texture = directive_value(line, "map_Kd");
                if (texture.empty())
                    continue;
                if (texture.front() == '"' && texture.back() == '"' && texture.size() >= 2)
                    texture = texture.substr(1, texture.size() - 2);
                current->texture = mtl_path.parent_path() / texture;
            }
            return !materials.empty();
        }
    }

    bool load_obj_material_texture(const std::filesystem::path& obj_path, ObjMaterialTexture& out)
    {
        out = ObjMaterialTexture{};

        const std::filesystem::path mtl_path = referenced_mtl(obj_path);
        if (mtl_path.empty() || !std::filesystem::is_regular_file(mtl_path))
            return false;

        MaterialMap materials;
        std::vector<std::string> material_order;
        if (!load_materials(mtl_path, materials, material_order))
            return false;
        out.has_material = true;

        std::vector<std::vector<double>> vertices;
        std::vector<std::vector<double>> texture_coordinates;
        std::vector<std::vector<double>> normals;
        std::vector<std::vector<int>> faces;
        std::vector<std::vector<int>> texture_faces;
        std::vector<std::vector<int>> normal_faces;
        std::vector<std::tuple<std::string, int, int>> material_ranges;
        if (!igl::readOBJ(
                obj_path.string(),
                vertices,
                texture_coordinates,
                normals,
                faces,
                texture_faces,
                normal_faces,
                material_ranges))
        {
            return false;
        }

        const Eigen::RowVector3d default_color = !material_order.empty()
            ? materials[material_order.front()].diffuse
            : Eigen::RowVector3d(0.8, 0.8, 0.8);
        Eigen::MatrixXd polygon_colors(static_cast<int>(faces.size()), 3);
        polygon_colors.rowwise() = default_color;
        for (const auto& range : material_ranges)
        {
            const auto found = materials.find(std::get<0>(range));
            if (found == materials.end())
                continue;
            const int first = std::max(0, std::get<1>(range));
            const int last = std::min(static_cast<int>(faces.size()) - 1, std::get<2>(range));
            for (int face = first; face <= last; ++face)
                polygon_colors.row(face) = found->second.diffuse;
        }

        int triangle_count = 0;
        for (const auto& face : faces)
            triangle_count += std::max(0, static_cast<int>(face.size()) - 2);
        out.face_colors.resize(triangle_count, 3);
        int triangle = 0;
        for (int polygon = 0; polygon < static_cast<int>(faces.size()); ++polygon)
        {
            for (int corner = 1; corner + 1 < static_cast<int>(faces[(size_t)polygon].size()); ++corner)
                out.face_colors.row(triangle++) = polygon_colors.row(polygon);
        }

        bool valid_uv = !texture_coordinates.empty() && texture_faces.size() == faces.size();
        for (const auto& coordinate : texture_coordinates)
            valid_uv = valid_uv && coordinate.size() >= 2;
        for (int polygon = 0; polygon < static_cast<int>(texture_faces.size()); ++polygon)
            valid_uv = valid_uv && texture_faces[(size_t)polygon].size() == faces[(size_t)polygon].size();
        if (valid_uv)
        {
            out.uv.resize(static_cast<int>(texture_coordinates.size()), 2);
            for (int i = 0; i < out.uv.rows(); ++i)
            {
                out.uv(i, 0) = texture_coordinates[static_cast<std::size_t>(i)][0];
                out.uv(i, 1) = texture_coordinates[static_cast<std::size_t>(i)][1];
            }
            out.uv_faces.resize(triangle_count, 3);
            triangle = 0;
            for (const auto& face : texture_faces)
            {
                for (int corner = 1; corner + 1 < static_cast<int>(face.size()); ++corner)
                {
                    out.uv_faces(triangle, 0) = face[0];
                    out.uv_faces(triangle, 1) = face[(size_t)corner];
                    out.uv_faces(triangle, 2) = face[(size_t)corner + 1u];
                    ++triangle;
                }
            }
        }

        std::filesystem::path texture_path;
        for (const auto& range : material_ranges)
        {
            const auto found = materials.find(std::get<0>(range));
            if (found != materials.end() && !found->second.texture.empty())
            {
                texture_path = found->second.texture;
                break;
            }
        }
        if (texture_path.empty())
        {
            for (const std::string& name : material_order)
            {
                if (!materials[name].texture.empty())
                {
                    texture_path = materials[name].texture;
                    break;
                }
            }
        }
        if (texture_path.empty())
            return true;
        if (!valid_uv)
        {
            out.warning = "The OBJ material has an image texture, but the mesh has no usable triangle UVs; using its material color.";
            return true;
        }
        if (!std::filesystem::is_regular_file(texture_path))
        {
            out.warning = "The OBJ texture image is missing; using its material color: " + texture_path.string();
            return true;
        }

        int channels = 0;
        unsigned char* pixels = stbi_load(
            texture_path.string().c_str(),
            &out.width,
            &out.height,
            &channels,
            STBI_rgb_alpha);
        if (!pixels || out.width <= 0 || out.height <= 0)
        {
            if (pixels)
                stbi_image_free(pixels);
            out.warning = "The OBJ diffuse texture could not be decoded: " + texture_path.string();
            out.width = 0;
            out.height = 0;
            return true;
        }

        const std::size_t byte_count = static_cast<std::size_t>(out.width)
            * static_cast<std::size_t>(out.height) * 4u;
        out.rgba.assign(pixels, pixels + byte_count);
        stbi_image_free(pixels);
        return true;
    }
}
