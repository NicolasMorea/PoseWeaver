#pragma once

#include <filesystem>
#include <string>

namespace poseweaver::postproc
{
    struct IsoContinuityFilterParams
    {
        float strength = 0.50f;
        float edge_threshold = 0.04f;
        int iterations = 12;
    };

    bool sharpen_isomap_prediction(
        const std::filesystem::path& input_npz,
        std::filesystem::path* out_npz,
        std::string* out_error);

    bool filter_isomap_continuity(
        const std::filesystem::path& input_npz,
        const IsoContinuityFilterParams& params,
        std::filesystem::path* out_npz,
        std::string* out_error);

    bool filter_depth_prediction(
        const std::filesystem::path& input_json_or_npz,
        std::filesystem::path* out_json,
        std::string* out_error);
}
