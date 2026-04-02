#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace poseweaver
{
    class AssetIndex
    {
    public:
        explicit AssetIndex(std::filesystem::path assets_root);

        void refresh();

        const std::vector<std::string>& poses() const { return m_poses; }
        const std::vector<std::string>& garments() const { return m_garments; }

    private:
        std::filesystem::path m_assets_root;
        std::vector<std::string> m_poses;
        std::vector<std::string> m_garments;

        static std::vector<std::string> list_entries(const std::filesystem::path& dir);
    };
}
