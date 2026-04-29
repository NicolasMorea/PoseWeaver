#pragma once

namespace poseweaver
{
    struct AppState;

    void install_imgui_state_settings_handler(AppState& state);
    void save_imgui_settings_to_disk();
}
