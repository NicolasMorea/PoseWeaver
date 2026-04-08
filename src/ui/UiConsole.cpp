#include "ui/Ui.h"

#include "ui/UiInternal.h"

#include <imgui.h>
#include <string>

namespace poseweaver
{
    void Ui::draw_console(AppState& state)
    {
        ImGui::Begin("Debug Console", &state.show_console, ui_internal::fixed_panel_flags());

        if (ImGui::Button("Clear"))
            state.console.lines.clear();
        ImGui::SameLine();
        ImGui::Checkbox("Auto-scroll", &state.console.auto_scroll);

        ImGui::Separator();

        ImGui::BeginChild("##console_scrollback", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_HorizontalScrollbar);
        const bool should_scroll = state.console.auto_scroll && (ImGui::GetScrollY() >= ImGui::GetScrollMaxY());

        for (const std::string& line : state.console.lines)
            ImGui::TextUnformatted(line.c_str());

        if (should_scroll)
            ImGui::SetScrollHereY(1.0f);

        ImGui::EndChild();
        ImGui::End();
    }
}
