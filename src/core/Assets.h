#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace poseweaver
{
    struct GarmentTextureVariant
    {
        std::string name;
        std::string file;
    };

    class AssetIndex
    {
    public:
        explicit AssetIndex(std::filesystem::path assets_root);

        void refresh();

        const std::vector<std::string>& poses() const { return m_poses; }
        const std::vector<std::string>& garments() const { return m_garments; }
        const std::vector<GarmentTextureVariant>& garment_textures(const std::string& garment) const;
        const std::vector<std::string>& meshes() const { return m_meshes; }

    private:
        std::filesystem::path m_assets_root;
        std::vector<std::string> m_poses;
        std::vector<std::string> m_garments;
        std::vector<std::vector<GarmentTextureVariant>> m_garment_textures;
        std::vector<std::string> m_meshes;

        void index_garments(const std::filesystem::path& dir);
        static std::vector<std::string> list_pose_files(const std::filesystem::path& dir);
        static std::vector<std::string> list_mesh_files(const std::filesystem::path& dir);
    };
}
