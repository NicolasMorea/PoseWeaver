#pragma once

#include "app/AppState.h"

#include <imgui.h>

namespace poseweaver::ui_internal
{
    inline ImGuiWindowFlags fixed_panel_flags()
    {
        return ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
    }

    inline ImGuiWindowFlags splitter_window_flags()
    {
        return ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
               ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoNav;
    }

    inline const char* smpl_type_label(SmplType g)
    {
        return (g == SmplType::Male) ? "Male" : "Female";
    }
}
