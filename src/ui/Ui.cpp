#include "ui/Ui.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <imgui.h>

namespace poseweaver
{
    static ImGuiWindowFlags fixed_panel_flags()
    {
        return ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
    }

    static const char* to_string(SmplType g)
    {
        return (g == SmplType::Male) ? "Male" : "Female"; // might add more in the future for different human proportions
    }

    void Ui::draw(AppState& state, const AssetIndex& assets, const UiFrameInfo& frame)
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

        const float left_w = std::clamp(work_size.x * 0.28f, 300.0f, 420.0f);
        const float bottom_h = std::clamp(work_size.y * 0.25f, 160.0f, 320.0f);
        const float top_h = std::max(0.0f, work_size.y - bottom_h);
        const float center_w = std::max(0.0f, work_size.x - left_w);

        const ImVec2 left_pos = work_pos;
        const ImVec2 left_size = ImVec2(left_w, top_h);

        const ImVec2 center_pos = ImVec2(work_pos.x + left_w, work_pos.y);
        const ImVec2 center_size = ImVec2(center_w, top_h);

        const ImVec2 console_pos = ImVec2(work_pos.x, work_pos.y + top_h);
        const ImVec2 console_size = ImVec2(work_size.x, bottom_h);

        // We put the options on the left side
        const bool left_sim = state.show_simulation;
        const bool left_smpl = state.show_smpl;
        const bool left_garments = state.show_garments;
        const int left_count = (left_sim ? 1 : 0) + (left_smpl ? 1 : 0) + (left_garments ? 1 : 0);

        float y_cursor = left_pos.y;
        const float slot_h = (left_count > 0) ? (left_size.y / (float)left_count) : 0.0f;

        if (left_sim)
        {
            ImGui::SetNextWindowPos(ImVec2(left_pos.x, y_cursor), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(left_size.x, slot_h), ImGuiCond_Always);
            draw_simulation(state);
            y_cursor += slot_h;
        }
        if (left_smpl)
        {
            ImGui::SetNextWindowPos(ImVec2(left_pos.x, y_cursor), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(left_size.x, slot_h), ImGuiCond_Always);
            draw_smpl(state, assets);
            y_cursor += slot_h;
        }
        if (left_garments)
        {
            ImGui::SetNextWindowPos(ImVec2(left_pos.x, y_cursor), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(left_size.x, slot_h), ImGuiCond_Always);
            draw_garments(state, assets);
            y_cursor += slot_h;
        }

        // Center view
        if (state.show_cloth_view)
        {
            ImGui::SetNextWindowPos(center_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(center_size, ImGuiCond_Always);
            draw_cloth_view(state, frame);
        }
        else
        {
            state.cloth_view.valid = false;
        }

        // Bottom console
        if (state.show_console)
        {
            ImGui::SetNextWindowPos(console_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(console_size, ImGuiCond_Always);
            draw_console(state);
        }

    }

    void Ui::draw_main_menu(AppState& state, const AssetIndex& assets)
    {
        (void)assets;

        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("Windows"))
            {
                ImGui::MenuItem("Cloth View", nullptr, &state.show_cloth_view);
                ImGui::MenuItem("Simulation", nullptr, &state.show_simulation);
                ImGui::MenuItem("SMPL", nullptr, &state.show_smpl);
                ImGui::MenuItem("Garments", nullptr, &state.show_garments);
				ImGui::MenuItem("Debug Console", nullptr, &state.show_console);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
    }

    void Ui::draw_cloth_view(AppState& state, const UiFrameInfo& frame)
    {
        ImGui::Begin("Cloth View", &state.show_cloth_view, fixed_panel_flags());

        const ImVec2 top_left = ImGui::GetCursorScreenPos();
        const ImVec2 avail = ImGui::GetContentRegionAvail();

        ImGui::InvisibleButton("cloth_view_canvas", avail);

        const ImGuiIO& io = ImGui::GetIO();
        const ImVec2 scale = io.DisplayFramebufferScale;

        const int x = static_cast<int>(top_left.x * scale.x);
        const int w = static_cast<int>(avail.x * scale.x);

        // ImGui Y is top-left, OpenGL expects bottom-left.
        const int y_top = static_cast<int>(top_left.y * scale.y);
        const int h = static_cast<int>(avail.y * scale.y);
        const int y = frame.fb_height - (y_top + h);

        state.cloth_view = ViewportRect{ x, y, w, h, (w > 0 && h > 0) };

        ImGui::Separator();
        ImGui::Text("SMPL: %s", to_string(state.selection.gender));
        ImGui::Text("Pose: %s", (state.selection.pose_index >= 0 && state.selection.pose_index < (int)state.selection.poses.size())
                                ? state.selection.poses[state.selection.pose_index].c_str()
                                : "(none)");
        ImGui::Text("Garment: %s", (state.selection.garment_index >= 0 && state.selection.garment_index < (int)state.selection.garments.size())
                                ? state.selection.garments[state.selection.garment_index].c_str()
                                : "(none)");

        ImGui::End();
    }

    void Ui::draw_simulation(AppState& state)
    {
        ImGui::Begin("Simulation", &state.show_simulation, fixed_panel_flags());

        if (ImGui::Button(state.controls.simulating ? "Stop Sim" : "Simulate"))
            state.controls.simulating = !state.controls.simulating;

        ImGui::SameLine();
        if (ImGui::Button(state.controls.playing ? "Pause" : "Play"))
            state.controls.playing = !state.controls.playing;

        ImGui::SameLine();
        if (ImGui::Button("Step"))
            state.controls.current_frame = (state.controls.current_frame + 1) % std::max(1, state.controls.max_frames);

        ImGui::SameLine();
        if (ImGui::Button("Reset"))
            state.controls.current_frame = 0;

        ImGui::Separator();
        if (state.controls.simulating)
            ImGui::TextUnformatted("Simulating: ON (placeholder; no physics implemented)");
        else
            ImGui::TextUnformatted("Simulating: OFF");

        ImGui::SliderInt("Frame", &state.controls.current_frame, 0, std::max(0, state.controls.max_frames - 1));
        ImGui::SliderInt("Max Frames", &state.controls.max_frames, 1, 10000);
        ImGui::SliderFloat("FPS", &state.controls.fps, 1.0f, 240.0f, "%.1f");

        ImGui::Separator();
        ImGui::TextUnformatted("Parameters (placeholders)");
        ImGui::SliderFloat("dt", &state.params.dt, 1.0f / 240.0f, 1.0f / 10.0f, "%.5f");
        ImGui::SliderInt("Iterations", &state.params.solver_iterations, 1, 100);
        ImGui::SliderFloat("Gravity", &state.params.gravity, -50.0f, 50.0f, "%.2f");
        ImGui::SliderFloat("Stretch", &state.params.stretch_stiffness, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Bend", &state.params.bend_stiffness, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Damping", &state.params.damping, 0.0f, 0.5f, "%.4f");

        ImGui::End();
    }

    void Ui::draw_smpl(AppState& state, const AssetIndex& assets)
    {
        ImGui::Begin("SMPL", &state.show_smpl, fixed_panel_flags());

        int gender = (state.selection.gender == SmplType::Male) ? 0 : 1;
        if (ImGui::RadioButton("Male", gender == 0))
            gender = 0;
        ImGui::SameLine();
        if (ImGui::RadioButton("Female", gender == 1))
            gender = 1;
        state.selection.gender = (gender == 0) ? SmplType::Male : SmplType::Female;

        ImGui::Separator();
        ImGui::Text("Poses directory: %s", (assets.poses().empty() ? "assets/poses (empty)" : "assets/poses"));

        // Copy the pose list to keeps UI state decoupled from the index
        state.selection.poses = assets.poses();

        if (state.selection.poses.empty())
        {
            ImGui::TextUnformatted("No poses found. Add files under assets/poses/");
            state.selection.pose_index = -1;
        }
        else
        {
            if (state.selection.pose_index < 0)
                state.selection.pose_index = 0;
            if (state.selection.pose_index >= (int)state.selection.poses.size())
                state.selection.pose_index = (int)state.selection.poses.size() - 1;

            if (ImGui::BeginListBox("##poses", ImVec2(-1.0f, 220.0f)))
            {
                for (int i = 0; i < (int)state.selection.poses.size(); ++i)
                {
                    const bool selected = (i == state.selection.pose_index);
                    if (ImGui::Selectable(state.selection.poses[i].c_str(), selected))
                        state.selection.pose_index = i;
                    if (selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndListBox();
            }
        }

        ImGui::End();
    }

    void Ui::draw_garments(AppState& state, const AssetIndex& assets)
    {
        ImGui::Begin("Garments", &state.show_garments, fixed_panel_flags());

        state.selection.garments = assets.garments();

        if (state.selection.garments.empty())
        {
            ImGui::TextUnformatted("No garments found. Add folders/files under assets/garments/");
            state.selection.garment_index = -1;
        }
        else
        {
            if (state.selection.garment_index < 0)
                state.selection.garment_index = 0;
            if (state.selection.garment_index >= (int)state.selection.garments.size())
                state.selection.garment_index = (int)state.selection.garments.size() - 1;

            if (ImGui::BeginListBox("##garments", ImVec2(-1.0f, 220.0f)))
            {
                for (int i = 0; i < (int)state.selection.garments.size(); ++i)
                {
                    const bool selected = (i == state.selection.garment_index);
                    if (ImGui::Selectable(state.selection.garments[i].c_str(), selected))
                        state.selection.garment_index = i;
                    if (selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndListBox();
            }
        }

        ImGui::End();
    }

    void Ui::draw_console(AppState& state)
    {
        ImGui::Begin("Debug Console", &state.show_console, fixed_panel_flags());

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
