#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace poseweaver::ui_canva
{
    bool save_normalized_model_input_png(
        int width,
        int height,
        const std::vector<unsigned char>& rgba,
        std::filesystem::path* out_path,
        std::string* out_error);
}
