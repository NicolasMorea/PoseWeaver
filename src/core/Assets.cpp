#include "core/Assets.h"

#include <algorithm>
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
            out.push_back(p.filename().string());
        }

        std::sort(out.begin(), out.end());
        return out;
    }

    void AssetIndex::refresh()
    {
        m_poses = list_entries(m_assets_root / "poses");
        m_garments = list_entries(m_assets_root / "garments");
    }
}
