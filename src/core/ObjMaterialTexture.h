#pragma once

#include <Eigen/Core>

#include <filesystem>
#include <string>
#include <vector>

namespace poseweaver
{
    struct ObjMaterialTexture
    {
        Eigen::MatrixXd uv;
        Eigen::MatrixXi uv_faces;
        std::vector<unsigned char> rgba;
        int width = 0;
        int height = 0;
        Eigen::MatrixXd face_colors;
        bool has_material = false;
        std::string warning;

        bool has_image_texture() const
        {
            return width > 0
                && height > 0
                && rgba.size() == static_cast<std::size_t>(width)
                    * static_cast<std::size_t>(height) * 4u
                && uv.rows() > 0
                && uv_faces.rows() > 0;
        }
    };

    bool load_obj_material_texture(
        const std::filesystem::path& obj_path,
        ObjMaterialTexture& out);
}
