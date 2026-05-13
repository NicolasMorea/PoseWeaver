#include "core/Assets.h"

#include <algorithm>
#include <cctype>
#include <system_error>

namespace poseweaver
{
    AssetIndex::AssetIndex(std::filesystem::path assets_root) : m_assets_root(std::move(assets_root))
    {
        refresh();
    }

    std::vector<std::string> AssetIndex::list_entries(const std::filesystem::path& dir)
    {
        std::vector<std::string> out;

        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec))
            return out;

        for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
        {
            if (ec)
                break;
            const auto& p = entry.path();
            const std::string name = p.filename().string();
            if (name.find(":Zone.Identifier") != std::string::npos) continue;
            out.push_back(name);
        }

        std::sort(out.begin(), out.end());
        return out;
    }

    std::vector<std::string> AssetIndex::list_pose_files(const std::filesystem::path& dir)
    {
        std::vector<std::string> out;

        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec))
            return out;

        for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
        {
            if (ec)
                break;
            if (!entry.is_regular_file(ec))
                continue;

            const auto& p = entry.path();
            const std::string name = p.filename().string();
            if (name.find(":Zone.Identifier") != std::string::npos)
                continue;

            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            if (ext == ".npz")
                out.push_back(name);
        }

        std::sort(out.begin(), out.end());
        return out;
    }

    std::vector<std::string> AssetIndex::list_mesh_files(const std::filesystem::path& dir)
    {
        std::vector<std::string> out;

        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec))
            return out;

        for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
        {
            if (ec)
                break;

            if (!entry.is_regular_file(ec))
                continue;

            const auto& p = entry.path();
            const std::string name = p.filename().string();
            if (name.find(":Zone.Identifier") != std::string::npos)
                continue;

            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });

            if (ext == ".obj" || ext == ".off" || ext == ".ply" || ext == ".stl" || ext == ".mesh" || ext == ".msh" || ext == ".wrl")
                out.push_back(name);
        }

        std::sort(out.begin(), out.end());
        return out;
    }

    void AssetIndex::refresh()
    {
        m_poses = list_pose_files(m_assets_root / "poses");
        m_garments = list_entries(m_assets_root / "garments");
        m_meshes = list_mesh_files(m_assets_root / "mesh");
    }
}
