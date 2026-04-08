#include "ui/Ui.h"

#include "render/Renderer.h"
#include "ui/UiInternal.h"

#include <algorithm>
#include <imgui.h>

namespace poseweaver
{
    void Ui::draw(AppState& state, const AssetIndex& assets, Renderer& renderer, const UiFrameInfo& frame)
    {
        draw_main_menu(state, assets);

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImVec2 work_pos = vp->WorkPos;
        ImVec2 work_size = vp->WorkSize;

        const float dy = work_pos.y - vp->Pos.y;
        if (dy < 1.0f)
        {
            const float menu_h = ImGui::GetFrameHeight();
            work_pos.y += menu_h;
            work_size.y = std::max(0.0f, work_size.y - menu_h);
        }

        const float splitter_h = 6.0f;
        const float min_console_h = 120.0f;
        const float max_console_h = std::max(min_console_h, work_size.y - 140.0f);
        if (state.layout.console_height <= 0.0f)
            state.layout.console_height = std::clamp(work_size.y * 0.25f, 160.0f, 320.0f);

        const float console_h = state.show_console ? std::clamp(state.layout.console_height, min_console_h, max_console_h) : 0.0f;
        const float top_h = std::max(0.0f, work_size.y - console_h - (state.show_console ? splitter_h : 0.0f));

        const bool show_controls = state.show_simulation || state.show_smpl || state.show_garments;

        const float splitter_w = 6.0f;
        const float min_left_w = 260.0f;
        const float min_center_w = 260.0f;
        const float max_left_w = std::max(min_left_w, work_size.x - splitter_w - min_center_w);

        const float default_left = std::clamp(work_size.x * 0.28f, 300.0f, 420.0f);
        if (state.layout.left_panel_width <= 0.0f)
            state.layout.left_panel_width = default_left;

        float left_w = show_controls ? std::clamp(state.layout.left_panel_width, min_left_w, max_left_w) : 0.0f;
        const float center_w = std::max(0.0f, work_size.x - left_w - (show_controls ? splitter_w : 0.0f));

        const ImVec2 left_pos = work_pos;
        const ImVec2 left_size = ImVec2(left_w, top_h);

        const ImVec2 splitter_pos = ImVec2(work_pos.x + left_w, work_pos.y);
        const ImVec2 splitter_size = ImVec2(splitter_w, top_h);

        const ImVec2 center_pos = ImVec2(work_pos.x + left_w + (show_controls ? splitter_w : 0.0f), work_pos.y);
        const ImVec2 center_size = ImVec2(center_w, top_h);

        const ImVec2 splitter_h_pos = ImVec2(work_pos.x, work_pos.y + top_h);
        const ImVec2 splitter_h_size = ImVec2(work_size.x, state.show_console ? splitter_h : 0.0f);

        const ImVec2 console_pos = ImVec2(work_pos.x, work_pos.y + top_h + (state.show_console ? splitter_h : 0.0f));
        const ImVec2 console_size = ImVec2(work_size.x, console_h);

        if (show_controls && left_size.x > 1.0f && left_size.y > 1.0f)
        {
            ImGui::SetNextWindowPos(left_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(left_size, ImGuiCond_Always);

            draw_controls(state, assets, renderer);
        }

        if (show_controls && center_size.x > 1.0f)
        {
            ImGui::SetNextWindowPos(splitter_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(splitter_size, ImGuiCond_Always);

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
            ImGui::Begin("##controls_splitter", nullptr, ui_internal::splitter_window_flags());

            ImGui::InvisibleButton("##splitter", splitter_size);
            const bool hovered = ImGui::IsItemHovered();
            const bool active = ImGui::IsItemActive();
            if (hovered || active)
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 p0 = ImGui::GetWindowPos();
            const ImVec2 p1 = ImVec2(p0.x + splitter_size.x, p0.y + splitter_size.y);
            dl->AddRectFilled(p0, p1, IM_COL32(120, 120, 120, (hovered || active) ? 180 : 90));

            if (active)
            {
                const float new_left = std::clamp(left_w + ImGui::GetIO().MouseDelta.x, min_left_w, max_left_w);
                state.layout.left_panel_width = new_left;
            }

            ImGui::End();
            ImGui::PopStyleColor();
        }

        if (state.show_cloth_view)
        {
            ImGui::SetNextWindowPos(center_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(center_size, ImGuiCond_Always);
            draw_cloth_view(state, renderer, frame);
        }
        else
        {
            state.cloth_view.valid = false;
        }

        if (state.show_console)
        {
            ImGui::SetNextWindowPos(splitter_h_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(splitter_h_size, ImGuiCond_Always);

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
            ImGui::Begin("##console_splitter", nullptr, ui_internal::splitter_window_flags());

            ImGui::InvisibleButton("##hsplitter", splitter_h_size);
            const bool hovered = ImGui::IsItemHovered();
            const bool active = ImGui::IsItemActive();
            if (hovered || active)
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);

            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 p0 = ImGui::GetWindowPos();
            const ImVec2 p1 = ImVec2(p0.x + splitter_h_size.x, p0.y + splitter_h_size.y);
            dl->AddRectFilled(p0, p1, IM_COL32(120, 120, 120, (hovered || active) ? 180 : 90));

            if (active)
            {
                const float new_console = std::clamp(console_h - ImGui::GetIO().MouseDelta.y, min_console_h, max_console_h);
                state.layout.console_height = new_console;
            }

            ImGui::End();
            ImGui::PopStyleColor();

            ImGui::SetNextWindowPos(console_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(console_size, ImGuiCond_Always);
            draw_console(state);
        }
    }
}
