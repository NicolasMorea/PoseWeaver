#include "render/RendererLineArt.h"

#include <Eigen/Core>

#include <algorithm>
#include <iostream>
#include <vector>

namespace
{
    bool preserves_surface_color(bool shaded)
    {
        constexpr int size = 16;
        constexpr unsigned char red = 48u;
        constexpr unsigned char green = 132u;
        constexpr unsigned char blue = 211u;
        std::vector<unsigned char> rgba((size_t)size * (size_t)size * 4u, 255u);
        for (size_t index = 0; index < rgba.size(); index += 4u)
        {
            rgba[index] = red;
            rgba[index + 1u] = green;
            rgba[index + 2u] = blue;
        }
        std::vector<float> depth((size_t)size * (size_t)size, 1.0f);
        Eigen::MatrixXd vertices(4, 3);
        vertices <<
            -0.8, -0.8, 0.0,
             0.8, -0.8, 0.0,
            -0.8,  0.8, 0.0,
             0.8,  0.8, 0.0;
        Eigen::MatrixXi faces(2, 3);
        faces << 0, 1, 2, 1, 3, 2;

        poseweaver::GreasePencilRenderSettings settings;
        settings.outline_width = 0.0f;
        settings.wrinkle_width = 0.0f;
        poseweaver::apply_grease_pencil_line_art(
            rgba,
            size,
            size,
            shaded,
            settings,
            vertices,
            faces,
            Eigen::Matrix4f::Identity(),
            Eigen::Matrix4f::Identity(),
            depth,
            {});

        if (!shaded)
            return rgba[0] == red && rgba[1] == green && rgba[2] == blue;
        int darkest = 255;
        int lightest = 0;
        for (size_t index = 0; index < rgba.size(); index += 4u)
        {
            const int luminance = (int)rgba[index] + (int)rgba[index + 1u] + (int)rgba[index + 2u];
            darkest = std::min(darkest, luminance / 3);
            lightest = std::max(lightest, luminance / 3);
        }
        return lightest - darkest >= 12;
    }

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
            depth,
            {});

        int dark_pixels = 0;
        for (size_t index = 0; index < rgba.size(); index += 4u)
            if (rgba[index] < 220u)
                dark_pixels++;
        return dark_pixels;
    }
}

int main()
{
    if (!preserves_surface_color(false) || !preserves_surface_color(true))
    {
        std::cerr << "Grease Pencil surface color was replaced or desaturated\n";
        return 1;
    }
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
