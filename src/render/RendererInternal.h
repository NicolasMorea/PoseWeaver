#pragma once

#include <Eigen/Core>

#include <algorithm>

namespace poseweaver::render_internal
{
    constexpr unsigned char kUvBlueByte = 96;
    constexpr double kUvBlue = (double)kUvBlueByte / 255.0;

    inline Eigen::RowVector3d uv_to_rgb01(double u, double v)
    {
        u = std::clamp(u, 0.0, 1.0);
        v = std::clamp(v, 0.0, 1.0);
        return Eigen::RowVector3d(u, v, kUvBlue);
    }
}
