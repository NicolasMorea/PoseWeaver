#include "render/RendererLineArt.h"

#include <Eigen/Core>

#include <iostream>
#include <vector>

namespace
{
    int rendered_crease_pixels(double folded_z)
    {
        constexpr int size = 64;
        std::vector<unsigned char> rgba((size_t)size * (size_t)size * 4u, 255u);
        std::vector<float> depth((size_t)size * (size_t)size, 1.0f);

        Eigen::MatrixXd vertices(4, 3);
        vertices <<
            -0.8, -0.8, 0.0,
             0.8, -0.8, 0.0,
            -0.8,  0.8, 0.0,
             0.8,  0.8, folded_z;
        Eigen::MatrixXi faces(2, 3);
        faces << 0, 1, 2, 1, 3, 2;

        poseweaver::GreasePencilRenderSettings settings;
        settings.outline_width = 0.0f;
        settings.wrinkle_width = 2.0f;
        settings.crease_angle_degrees = 32.0f;
        poseweaver::apply_grease_pencil_line_art(
            rgba,
            size,
            size,
            false,
            settings,
            vertices,
            faces,
            Eigen::Matrix4f::Identity(),
            Eigen::Matrix4f::Identity(),
            depth);

        int dark_pixels = 0;
        for (size_t index = 0; index < rgba.size(); index += 4u)
            if (rgba[index] < 220u)
                dark_pixels++;
        return dark_pixels;
    }
}

int main()
{
    const int flat_pixels = rendered_crease_pixels(0.0);
    const int folded_pixels = rendered_crease_pixels(1.0);
    if (flat_pixels != 0)
    {
        std::cerr << "Flat triangulation produced " << flat_pixels << " false crease pixels\n";
        return 1;
    }
    if (folded_pixels < 40)
    {
        std::cerr << "Folded surface produced only " << folded_pixels << " crease pixels\n";
        return 1;
    }
    std::cout << "Grease Pencil line art smoke passed: " << folded_pixels << " crease pixels\n";
    return 0;
}
