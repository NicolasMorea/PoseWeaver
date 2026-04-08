#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace poseweaver
{
    struct ViewportRect
    {
        // (x, y) is bottom-left.
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
        bool valid = false;
    };

    enum class SmplType
    {
        Male = 0,
        Female = 1,
        Neutral = 2,
    };

    struct SimulationControls
    {
        bool simulating = false;
        bool playing = false;
        int current_frame = 0;
        int max_frames = 300;
        float fps = 30.0f;
    };

    struct SimulationParams
    {
        float dt = 1.0f / 60.0f;
        int solver_iterations = 10;
        float gravity = -9.81f;
        float stretch_stiffness = 1.0f;
        float bend_stiffness = 0.5f;
        float damping = 0.01f;
    };

    struct SelectionState
    {
        SmplType gender = SmplType::Male;

        std::vector<std::string> poses;
        int pose_index = -1;

        std::vector<std::string> garments;
        int garment_index = -1;

        std::vector<std::string> meshes;
        int mesh_index = -1;
    };

    struct DebugConsoleState
    {
        bool auto_scroll = true;
        std::vector<std::string> lines;
    };

    struct UiLayoutState // resizable panes
    {
        float left_panel_width = 360.0f;
        float console_height = 240.0f;
    };

    inline void console_add_line(DebugConsoleState& console, std::string line)
    {
        constexpr size_t kMaxLines = 2000;
        console.lines.emplace_back(std::move(line));
        if (console.lines.size() > kMaxLines)
        {
            const size_t extra = console.lines.size() - kMaxLines;
            console.lines.erase(console.lines.begin(), console.lines.begin() + (std::ptrdiff_t)extra);
        }
    }

    struct SmplShapeState
    {
        // SMPLH shape parameters
        std::array<float, 16> betas{};
    };

    struct AppState
    {
        SimulationControls controls;
        SimulationParams params;
        SelectionState selection;
        ViewportRect cloth_view;

        // Commands : Incremented by UI for calling
        std::uint64_t garment_pushout_request_id = 0;
        std::uint64_t garment_ease_request_id = 0;
        std::uint64_t garment_prefit_request_id = 0;
        std::uint64_t garment_relax_request_id = 0;
        std::uint64_t uv_image_request_id = 0;

        // Commands parameters
        float garment_min_cloth_skin_distance = 0.003f; // in model units
        float garment_ease_amount = 0.20f;
        int garment_prefit_steps = 10;

        // UV view controls
        float uv_view_zoom = 1.0f;
        float uv_view_pan_x = 0.0f;
        float uv_view_pan_y = 0.0f;

        SmplShapeState smpl_shape;
        UiLayoutState layout;

        DebugConsoleState console;

        bool show_cloth_view = true;
        bool show_simulation = true;
        bool show_smpl = true;
        bool show_smpl_character = true;
        bool smpl_color_by_uv = false;
        bool show_garments = true;
        bool show_object_mesh = false;
        bool show_garment_mesh = true;
        bool garment_color_by_smpl_uv = false;
        bool show_console = true;
    };

    inline void console_add_line(AppState& state, std::string line)
    {
        console_add_line(state.console, std::move(line));
    }
}
