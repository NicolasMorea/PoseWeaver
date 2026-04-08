#include "ui/Ui.h"

#include <imgui.h>

namespace poseweaver
{
    void Ui::draw_main_menu(AppState& state, const AssetIndex& assets)
    {
        (void)assets;

        if (!ImGui::BeginMainMenuBar())
            return;

        if (ImGui::BeginMenu("Windows"))
        {
            ImGui::MenuItem("Scene View", nullptr, &state.show_cloth_view);
            ImGui::MenuItem("Simulation", nullptr, &state.show_simulation);
            ImGui::MenuItem("SMPL", nullptr, &state.show_smpl);
            ImGui::MenuItem("Garments", nullptr, &state.show_garments);
            ImGui::MenuItem("Debug Console", nullptr, &state.show_console);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}
