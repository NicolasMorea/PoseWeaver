#include "postproc/PredictionFilters.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <cnpy.h>

int main()
{
    constexpr int width = 9;
    constexpr int height = 7;
    constexpr int channels = 4;
    const size_t pixel_count = (size_t)width * (size_t)height;
    std::vector<float> isomap(pixel_count * (size_t)channels, 0.0f);
    std::vector<std::uint8_t> mask(pixel_count, 1u);
    std::vector<float> probability(pixel_count, 1.0f);
    std::vector<float> low((size_t)channels, 0.0f);
    std::vector<float> high((size_t)channels, 1.0f);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const size_t pixel = (size_t)y * (size_t)width + (size_t)x;
            const float region = x < 5 ? 0.20f : 0.80f;
            for (int c = 0; c < channels; ++c)
                isomap[pixel * (size_t)channels + (size_t)c] = region + 0.002f * (float)y;
        }
    }
    const size_t noisy_pixel = (size_t)3 * (size_t)width + 2u;
    for (int c = 0; c < channels; ++c)
        isomap[noisy_pixel * (size_t)channels + (size_t)c] = 0.27f;

    const std::filesystem::path input =
        std::filesystem::temp_directory_path() / "poseweaver_iso_filter_smoke_sharpened.npz";
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "poseweaver_iso_filter_smoke_filtered_iso.npz";
    std::error_code remove_error;
    std::filesystem::remove(input, remove_error);
    std::filesystem::remove(output, remove_error);

    cnpy::npz_save(
        input.string(), "isomap", isomap.data(),
        { (size_t)height, (size_t)width, (size_t)channels }, "w");
    cnpy::npz_save(input.string(), "mask", mask.data(), { (size_t)height, (size_t)width }, "a");
    cnpy::npz_save(
        input.string(), "mask_probability", probability.data(),
        { (size_t)height, (size_t)width }, "a");
    cnpy::npz_save(input.string(), "descriptor_min", low.data(), { (size_t)channels }, "a");
    cnpy::npz_save(input.string(), "descriptor_max", high.data(), { (size_t)channels }, "a");

    poseweaver::postproc::IsoContinuityFilterParams params;
    params.strength = 0.5f;
    params.edge_threshold = 0.04f;
    params.iterations = 12;
    std::filesystem::path actual_output;
    std::string error;
    if (!poseweaver::postproc::filter_isomap_continuity(
            input, params, &actual_output, &error))
    {
        std::cerr << "continuity filter failed: " << error << '\n';
        return 1;
    }

    const cnpy::NpyArray filtered = cnpy::npz_load(actual_output.string(), "isomap");
    const float* values = filtered.data<float>();
    const float smoothed_noise = values[noisy_pixel * (size_t)channels];
    const size_t left_edge = ((size_t)3 * (size_t)width + 4u) * (size_t)channels;
    const size_t right_edge = ((size_t)3 * (size_t)width + 5u) * (size_t)channels;
    const float retained_edge = values[right_edge] - values[left_edge];

    std::filesystem::remove(input, remove_error);
    std::filesystem::remove(actual_output, remove_error);

    if (!(smoothed_noise < 0.235f))
    {
        std::cerr << "isolated descriptor noise was not sufficiently reduced: "
                  << smoothed_noise << '\n';
        return 2;
    }
    if (!(retained_edge > 0.50f))
    {
        std::cerr << "hard descriptor transition was blurred: " << retained_edge << '\n';
        return 3;
    }
    return 0;
}
