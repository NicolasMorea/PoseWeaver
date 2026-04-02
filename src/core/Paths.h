#pragma once

#include <filesystem>

namespace poseweaver
{
    inline std::filesystem::path assets_dir()
    {
    #ifdef POSEWEAVER_ASSETS_DIR
        return std::filesystem::path(POSEWEAVER_ASSETS_DIR);
    #else
        return std::filesystem::path("assets");
    #endif
    }
}