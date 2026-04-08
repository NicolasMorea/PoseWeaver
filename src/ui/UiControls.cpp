#include "ui/Ui.h"

#include "render/Renderer.h"
#include "ui/UiInternal.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <imgui.h>
#include <string>

namespace poseweaver
{
    static void log_line(AppState& state, const char* fmt, ...)
    {
        char buf[2048];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        state.console.lines.emplace_back(buf);
    }

    void Ui::draw_controls(AppState& state, const AssetIndex& assets, Renderer& renderer)
    {
        ImGui::Begin("Controls", nullptr, ui_internal::fixed_panel_flags());

        if (ImGui::BeginTabBar("##controls_tabs"))
        {
            if (ImGui::BeginTabItem("Setup"))
            {
                if (state.show_simulation && ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::TextUnformatted(state.controls.simulating
                        ? "Simulating: ON (placeholder; no physics implemented)"
                        : "Simulating: OFF");

                    ImGui::SliderInt("Frame", &state.controls.current_frame, 0, std::max(0, state.controls.max_frames - 1));
                    ImGui::SliderInt("Max Frames", &state.controls.max_frames, 1, 10000);
                    ImGui::SliderFloat("FPS", &state.controls.fps, 1.0f, 240.0f, "%.1f");

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
                    ImGui::Checkbox("Color by UV", &state.smpl_color_by_uv);

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

                    if (ImGui::TreeNodeEx("Poses", ImGuiTreeNodeFlags_DefaultOpen))
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
                            state.selection.pose_index = std::clamp(state.selection.pose_index, -1, (int)state.selection.poses.size() - 1);

                            if (ImGui::BeginListBox("##poses", ImVec2(-1.0f, 220.0f)))
                            {
                                const bool rest_selected = (state.selection.pose_index == -1);
                                if (ImGui::Selectable("(Rest / T-pose)", rest_selected))
                                    state.selection.pose_index = -1;
                                if (rest_selected)
                                    ImGui::SetItemDefaultFocus();

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
                    ImGui::Checkbox("Show object mesh", &state.show_object_mesh);
                    ImGui::Checkbox("Show garment mesh", &state.show_garment_mesh);

                    if (renderer.garment_to_smpl_mapping_valid())
                        ImGui::Checkbox("Color cloth by SMPL UV", &state.garment_color_by_smpl_uv);

                    if (ImGui::TreeNodeEx("Garments##garment_selection", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        state.selection.garments = assets.garments();

                        if (state.selection.garments.empty())
                        {
                            ImGui::TextUnformatted("No garments found. Add folders/files under assets/garments/");
                            state.selection.garment_index = -1;
                        }
                        else
                        {
                            state.selection.garment_index = std::clamp(state.selection.garment_index, -1, (int)state.selection.garments.size() - 1);

                            if (ImGui::BeginListBox("##garments", ImVec2(-1.0f, 220.0f)))
                            {
                                const bool none_selected = (state.selection.garment_index == -1);
                                if (ImGui::Selectable("(None)", none_selected))
                                    state.selection.garment_index = -1;
                                if (none_selected)
                                    ImGui::SetItemDefaultFocus();

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
                            state.selection.mesh_index = std::clamp(state.selection.mesh_index, 0, (int)state.selection.meshes.size() - 1);

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

                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Prefit"))
            {
                if (state.show_simulation && ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::Button(state.controls.simulating ? "Stop" : "Simulate"))
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
                }

                if (state.show_smpl && ImGui::CollapsingHeader("SMPL", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::Button("UV image"))
                    {
                        state.uv_image_request_id++;
                        log_line(state, "[SMPL] Generate UV Image requested");
                    }
                }

                if (state.show_garments && ImGui::CollapsingHeader("Garments", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::SliderFloat("Min cloth-skin distance", &state.garment_min_cloth_skin_distance, 0.0f, 0.02f, "%.4f");

                    if (ImGui::Button("Push-out"))
                    {
                        state.garment_pushout_request_id++;
                        log_line(state, "[Garment] Push-out requested");
                    }

                    ImGui::Separator();

                    ImGui::SliderFloat("amount##ease_cloth", &state.garment_ease_amount, 0.0f, 1.0f, "%.3f");
                    if (ImGui::Button("Ease"))
                    {
                        state.garment_ease_request_id++;
                        log_line(state, "[Garment] Ease cloth requested (amount=%.3f)", state.garment_ease_amount);
                    }

                    ImGui::SliderInt("number of steps##prefit", &state.garment_prefit_steps, 1, 200);
                    if (ImGui::Button("Prefit"))
                    {
                        state.garment_prefit_request_id++;
                        log_line(state, "[Garment] Prefit requested (steps=%d)", state.garment_prefit_steps);
                    }

                    if (ImGui::Button("Relax (mild)"))
                    {
                        state.garment_relax_request_id++;
                        log_line(state, "[Garment] Mild relax requested");
                    }
                }

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::Separator();
        ImGui::Text("SMPL: %s", ui_internal::smpl_type_label(state.selection.gender));
        ImGui::Text("Pose: %s",
            (state.selection.pose_index >= 0 && state.selection.pose_index < (int)state.selection.poses.size())
                ? state.selection.poses[state.selection.pose_index].c_str()
                : "(none)");
        ImGui::Text("Garment: %s",
            (state.selection.garment_index >= 0 && state.selection.garment_index < (int)state.selection.garments.size())
                ? state.selection.garments[state.selection.garment_index].c_str()
                : "(none)");
        ImGui::Text("Mesh: %s",
            (state.selection.mesh_index >= 0 && state.selection.mesh_index < (int)state.selection.meshes.size())
                ? state.selection.meshes[state.selection.mesh_index].c_str()
                : "(none)");

        ImGui::End();
    }
}
