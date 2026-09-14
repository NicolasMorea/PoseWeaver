#pragma once

#include <Eigen/Core>

#include <array>
#include <vector>

namespace poseweaver
{
    struct GreasePencilRenderSettings
    {
        float outline_width = 3.0f;
        float wrinkle_width = 1.25f;
        float crease_angle_degrees = 32.0f;
        float shadow_threshold = 0.72f;
        float shadow_range = 0.34f;
        float shadow_strength = 0.72f;
        std::array<float, 3> line_color{ 0.03f, 0.03f, 0.04f };
    };

    void apply_grease_pencil_line_art(
        std::vector<unsigned char>& rgba,
        int width,
        int height,
        bool shaded,
        const GreasePencilRenderSettings& settings,
        const Eigen::MatrixXd& vertices,
        const Eigen::MatrixXi& faces,
        const Eigen::Matrix4f& view,
        const Eigen::Matrix4f& projection,
        const std::vector<float>& gl_depth);
}
