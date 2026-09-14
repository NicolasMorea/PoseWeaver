#pragma once

#include <Eigen/Core>

#include <filesystem>
#include <string>
#include <vector>

namespace poseweaver
{
    struct GltfGarment
    {
        Eigen::MatrixXd vertices;
        Eigen::MatrixXi faces;
        Eigen::MatrixXd uv;
        Eigen::MatrixXi uv_faces;
        Eigen::MatrixXd face_colors;
        std::vector<unsigned char> texture_rgba;
        int texture_width = 0;
        int texture_height = 0;
        bool double_sided = false;
        bool has_material = false;
        std::string texture_warning;

        bool has_image_texture() const
        {
            return texture_width > 0
                && texture_height > 0
                && texture_rgba.size() == static_cast<std::size_t>(texture_width)
                    * static_cast<std::size_t>(texture_height) * 4u
                && uv.rows() > 0
                && uv_faces.rows() == faces.rows();
        }
    };

    bool load_gltf_garment(
        const std::filesystem::path& path,
        GltfGarment& out,
        std::string* error);
}
