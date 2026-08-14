#pragma once

#include <filesystem>
#include <string>

namespace poseweaver::postproc
{
    bool filter_isomap_prediction(
        const std::filesystem::path& input_npz,
        std::filesystem::path* out_npz,
        std::string* out_error);

    bool filter_depth_prediction(
        const std::filesystem::path& input_json_or_npz,
        std::filesystem::path* out_json,
        std::string* out_error);
}
