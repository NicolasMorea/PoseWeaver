#pragma once

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

    struct AppState
    {
        SimulationControls controls;
        SimulationParams params;
        SelectionState selection;
        ViewportRect cloth_view;

        DebugConsoleState console;

        bool show_cloth_view = true;
        bool show_simulation = true;
        bool show_smpl = true;
        bool show_garments = true;
		bool show_console = true;
    };
}
