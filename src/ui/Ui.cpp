#include "ui/Ui.h"

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <string>
#include <imgui.h>

namespace poseweaver
{
    static ImGuiWindowFlags fixed_panel_flags()
    {
        return ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
    }

    static ImGuiWindowFlags splitter_window_flags()
    {
        return ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
               ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoNav;
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

        const float bottom_h = std::clamp(work_size.y * 0.25f, 160.0f, 320.0f);
        const float top_h = std::max(0.0f, work_size.y - bottom_h);

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

        const ImVec2 console_pos = ImVec2(work_pos.x, work_pos.y + top_h);
        const ImVec2 console_size = ImVec2(work_size.x, bottom_h);

        // Left controls
        if (show_controls && left_size.x > 1.0f && left_size.y > 1.0f)
        {
            ImGui::SetNextWindowPos(left_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(left_size, ImGuiCond_Always);
            draw_controls(state, assets);
        }

        // Vertical splitter between Controls and Cloth View
        if (show_controls && center_size.x > 1.0f)
        {
            ImGui::SetNextWindowPos(splitter_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(splitter_size, ImGuiCond_Always);

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
            ImGui::Begin("##controls_splitter", nullptr, splitter_window_flags());

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
        ImGui::Begin("Cloth View", &state.show_cloth_view, fixed_panel_flags() | ImGuiWindowFlags_NoBackground);

        const ImVec2 top_left = ImGui::GetCursorScreenPos();
        const ImVec2 avail = ImGui::GetContentRegionAvail();

        ImGui::InvisibleButton("cloth_view_canvas", avail);

        const ImGuiIO& io = ImGui::GetIO();
        const ImVec2 scale = io.DisplayFramebufferScale;

        // Convert ImGui screen-space (absolute) coordinates into window-local framebuffer coordinates.
        // ImGui::GetCursorScreenPos() is in screen space relative to the platform viewport.
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const ImVec2 vp_pos = vp ? vp->Pos : ImVec2(0.0f, 0.0f);
        const ImVec2 local = ImVec2(top_left.x - vp_pos.x, top_left.y - vp_pos.y);

        const int x = static_cast<int>(local.x * scale.x);
        const int w = static_cast<int>(avail.x * scale.x);

        // ImGui Y is top-left, OpenGL expects bottom-left.
        const int y_top = static_cast<int>(local.y * scale.y);
        const int h = static_cast<int>(avail.y * scale.y);
        const int y = frame.fb_height - (y_top + h);

        state.cloth_view = ViewportRect{ x, y, w, h, (w > 0 && h > 0) };

        ImGui::End();
    }

    void Ui::draw_controls(AppState& state, const AssetIndex& assets)
    {
        ImGui::Begin("Controls", nullptr, fixed_panel_flags());

        if (state.show_simulation && ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::TreeNode("Playback"))
            {
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

                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Parameters"))
            {
                ImGui::SliderFloat("dt", &state.params.dt, 1.0f / 240.0f, 1.0f / 10.0f, "%.5f");
                ImGui::SliderInt("Iterations", &state.params.solver_iterations, 1, 100);
                ImGui::SliderFloat("Gravity", &state.params.gravity, -50.0f, 50.0f, "%.2f");
                ImGui::SliderFloat("Stretch", &state.params.stretch_stiffness, 0.0f, 1.0f, "%.3f");
                ImGui::SliderFloat("Bend", &state.params.bend_stiffness, 0.0f, 1.0f, "%.3f");
                ImGui::SliderFloat("Damping", &state.params.damping, 0.0f, 0.5f, "%.4f");
                ImGui::TreePop();
            }
        }

        if (state.show_smpl && ImGui::CollapsingHeader("SMPL", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Show character", &state.show_smpl_character);

            if (ImGui::TreeNode("Gender"))
            {
                int gender = (state.selection.gender == SmplType::Male) ? 0 : 1;
                if (ImGui::RadioButton("Male", gender == 0))
                    gender = 0;
                ImGui::SameLine();
                if (ImGui::RadioButton("Female", gender == 1))
                    gender = 1;
                state.selection.gender = (gender == 0) ? SmplType::Male : SmplType::Female;
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Poses"))
            {
                ImGui::Text("Poses directory: %s", (assets.poses().empty() ? "assets/poses (empty)" : "assets/poses"));
                state.selection.poses = assets.poses();

                if (state.selection.poses.empty())
                {
                    ImGui::TextUnformatted("No poses found. Add files under assets/poses/");
                    state.selection.pose_index = -1;
                }
                else
                {
                    if (state.selection.pose_index < -1)
                        state.selection.pose_index = -1;
                    if (state.selection.pose_index >= (int)state.selection.poses.size())
                        state.selection.pose_index = (int)state.selection.poses.size() - 1;

                    if (ImGui::BeginListBox("##poses", ImVec2(-1.0f, 220.0f)))
                    {
                        {
                            const bool selected = (state.selection.pose_index == -1);
                            if (ImGui::Selectable("(Rest / T-pose)", selected))
                                state.selection.pose_index = -1;
                            if (selected)
                                ImGui::SetItemDefaultFocus();
                        }

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
                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Shape (betas)"))
            {
                ImGui::TextUnformatted("Betas");
                for (int i = 0; i < (int)state.smpl_shape.betas.size(); ++i)
                {
                    char label[32];
                    std::snprintf(label, sizeof(label), "beta[%d]", i);
                    ImGui::SliderFloat(label, &state.smpl_shape.betas[(size_t)i], -3.0f, 3.0f, "%.2f");
                }
                ImGui::TreePop();
            }
        }

        if (state.show_garments && ImGui::CollapsingHeader("Garments", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::TreeNode("Garments"))
            {
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

                ImGui::TreePop();
            }

            if (ImGui::TreeNode("Meshes (assets/mesh)"))
            {
                state.selection.meshes = assets.meshes();

                if (state.selection.meshes.empty())
                {
                    ImGui::TextUnformatted("No meshes found. Add files under assets/mesh/");
                    state.selection.mesh_index = -1;
                }
                else
                {
                    if (state.selection.mesh_index < 0)
                        state.selection.mesh_index = 0;
                    if (state.selection.mesh_index >= (int)state.selection.meshes.size())
                        state.selection.mesh_index = (int)state.selection.meshes.size() - 1;

                    if (ImGui::BeginListBox("##meshes", ImVec2(-1.0f, 220.0f)))
                    {
                        for (int i = 0; i < (int)state.selection.meshes.size(); ++i)
                        {
                            const bool selected = (i == state.selection.mesh_index);
                            if (ImGui::Selectable(state.selection.meshes[i].c_str(), selected))
                                state.selection.mesh_index = i;
                            if (selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndListBox();
                    }
                }

                ImGui::TreePop();
            }
        }

        ImGui::Separator();
        ImGui::Text("SMPL: %s", to_string(state.selection.gender));
        ImGui::Text("Pose: %s", (state.selection.pose_index >= 0 && state.selection.pose_index < (int)state.selection.poses.size())
                                ? state.selection.poses[state.selection.pose_index].c_str()
                                : "(none)");
        ImGui::Text("Garment: %s", (state.selection.garment_index >= 0 && state.selection.garment_index < (int)state.selection.garments.size())
                                ? state.selection.garments[state.selection.garment_index].c_str()
                                : "(none)");
        ImGui::Text("Mesh: %s", (state.selection.mesh_index >= 0 && state.selection.mesh_index < (int)state.selection.meshes.size())
                                ? state.selection.meshes[state.selection.mesh_index].c_str()
                                : "(none)");

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
