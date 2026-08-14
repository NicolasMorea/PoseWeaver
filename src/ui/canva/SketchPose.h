#pragma once

#include "app/AppState.h"

#include <filesystem>
#include <string>
#include <vector>

namespace poseweaver::ui_canva
{
    struct SketchPoseBuildResult
    {
        std::filesystem::path json_path;
        std::filesystem::path npz_path;
        std::string gender;
        std::vector<double> betas;
        std::vector<double> pose_axis_angle;
    };

    bool save_sketch_pose_from_outputs(
        const std::filesystem::path& rig_json_path,
        const std::filesystem::path& betas_json_path,
        const std::filesystem::path& out_json_path,
        const std::filesystem::path& out_npz_path,
        SketchPoseBuildResult* out_result,
        std::string* out_error);

    bool save_nlf_pose_from_output(
        const std::filesystem::path& nlf_json_path,
        const std::filesystem::path& out_json_path,
        const std::filesystem::path& out_npz_path,
        SketchPoseBuildResult* out_result,
        std::string* out_error);
}
