#include "ui/Ui.h"

#include "core/Paths.h"
#include "render/Renderer.h"
#include "smpl/Smplh.h"
#include "ui/UiInternal.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <imgui.h>
#include <optional>
#include <string>

namespace poseweaver
{
    namespace
    {
        static void apply_pose_npz_metadata(AppState& state, const std::string& pose_name)
        {
            if (pose_name.empty())
                return;

            std::optional<Eigen::VectorXd> betas;
            std::optional<std::string> gender;
            std::string err;
            if (!smplh_load_pose_npz_metadata(assets_dir() / "poses" / pose_name, &betas, &gender, &err))
            {
                if (!err.empty())
                    console_add_line(state, "Pose metadata ignored: " + err);
                return;
            }

            if (gender)
            {
                if (*gender == "male")
                    state.selection.gender = SmplType::Male;
                else if (*gender == "female")
                    state.selection.gender = SmplType::Female;
                else if (*gender == "neutral")
                    state.selection.gender = SmplType::Neutral;
            }

            if (betas)
            {
                const int n = std::min<int>((int)state.smpl_shape.betas.size(), (int)betas->size());
                for (int i = 0; i < n; ++i)
                    state.smpl_shape.betas[(size_t)i] = (float)(*betas)[i];
                for (int i = n; i < (int)state.smpl_shape.betas.size(); ++i)
                    state.smpl_shape.betas[(size_t)i] = 0.0f;
            }
        }
    }

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
            static bool s_default_setup_tab = true;
            const ImGuiTabItemFlags setup_flags = s_default_setup_tab ? ImGuiTabItemFlags_SetSelected : 0;
            if (ImGui::BeginTabItem("Setup", nullptr, setup_flags))
            {
                s_default_setup_tab = false;
                if (state.show_smpl && ImGui::CollapsingHeader("SMPL", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Checkbox("Show character", &state.show_smpl_character);
                    ImGui::Checkbox("Color by UV", &state.smpl_color_by_uv);

                    if (ImGui::TreeNode("Gender"))
                    {
                        int gender = (state.selection.gender == SmplType::Male)
                            ? 0
                            : (state.selection.gender == SmplType::Female) ? 1 : 2;
                        if (ImGui::RadioButton("Male", gender == 0))
                            gender = 0;
                        ImGui::SameLine();
                        if (ImGui::RadioButton("Female", gender == 1))
                            gender = 1;
                        ImGui::SameLine();
                        if (ImGui::RadioButton("Neutral", gender == 2))
                            gender = 2;
                        state.selection.gender = (gender == 0)
                            ? SmplType::Male
                            : (gender == 1) ? SmplType::Female : SmplType::Neutral;
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
                                    {
                                        state.selection.pose_index = i;
                                        apply_pose_npz_metadata(state, state.selection.poses[(size_t)i]);
                                    }
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

                    ImGui::Separator();

                    ImGui::Checkbox("Merge cloth vertices", &state.garment_vertex_merge_enabled);
                    ImGui::BeginDisabled(!state.garment_vertex_merge_enabled);
                    {
                        static bool s_merge_dist_init = false;
                        static float s_merge_dist = 0.0025f;
                        if (!s_merge_dist_init)
                        {
                            s_merge_dist = state.garment_vertex_merge_distance;
                            s_merge_dist_init = true;
                        }

                        ImGui::SliderFloat("Merge distance", &s_merge_dist, 0.0f, 0.02f, "%.6f");
                        if (ImGui::IsItemDeactivatedAfterEdit())
                            state.garment_vertex_merge_distance = std::clamp(s_merge_dist, 0.0f, 0.05f);
                        else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited())
                            s_merge_dist = state.garment_vertex_merge_distance;
                    }
                    ImGui::EndDisabled();

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
            s_default_setup_tab = false;

            if (ImGui::BeginTabItem("Prefit"))
            {
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

            if (ImGui::BeginTabItem("Simulate"))
            {
                if (state.show_simulation && ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    const int frame_max = (state.controls.frame_count > 0) ? (state.controls.frame_count - 1) : 0;
                    state.controls.current_frame = std::clamp(state.controls.current_frame, 0, std::max(0, frame_max));

                    if (ImGui::TreeNodeEx("Parameters", ImGuiTreeNodeFlags_DefaultOpen))
                    {
                        ImGui::Checkbox("Smooth sequence", &state.controls.smooth_sequence);
                        ImGui::BeginDisabled(!state.controls.smooth_sequence);
                        {
                            float pct = state.controls.smooth_sequence_length * 100.0f;
                            if (ImGui::SliderFloat("Sequence length", &pct, 0.0f, 100.0f, "%.0f%%"))
                                state.controls.smooth_sequence_length = std::clamp(pct / 100.0f, 0.0f, 1.0f);
                        }
                        ImGui::EndDisabled();

                        ImGui::Separator();

                        // Solver selection (kept in SimulationControls so playback UX stays identical).
                        {
                            int solver_i = (int)state.controls.solver;
                            const char* solver_items[] = { "Projective Dynamics", "XPBD" };
                            if (ImGui::Combo("Solver", &solver_i, solver_items, IM_ARRAYSIZE(solver_items)))
                                state.controls.solver = (SimulationSolver)std::clamp(solver_i, 0, 1);
                        }

                        if (state.controls.solver == SimulationSolver::ProjectiveDynamics)
                        {
                            ImGui::InputFloat("dt", &state.params.dt, 0.001f, 0.01f, "%.6f");
                            state.params.dt = std::max(state.params.dt, 1e-6f);

                            ImGui::InputFloat("Simulation time (s)", &state.params.duration_seconds, 0.1f, 1.0f, "%.3f");
                            state.params.duration_seconds = std::max(state.params.duration_seconds, 0.0f);

                            ImGui::InputInt("Iterations", &state.params.solver_iterations);
                            state.params.solver_iterations = std::max(state.params.solver_iterations, 1);

                            ImGui::InputFloat("Gravity", &state.params.gravity, 0.5f, 5.0f, "%.3f");

                            if (ImGui::TreeNodeEx("Constraints", ImGuiTreeNodeFlags_DefaultOpen))
                            {
                                ImGui::Checkbox("Stretch", &state.params.stretch_enabled);
                                ImGui::BeginDisabled(!state.params.stretch_enabled);
                                ImGui::InputFloat("Stretch stiffness", &state.params.stretch_stiffness, 0.1f, 1.0f, "%.6f");
                                ImGui::EndDisabled();
                                state.params.stretch_stiffness = std::max(state.params.stretch_stiffness, 0.0f);

                                ImGui::BeginDisabled(!state.params.stretch_enabled);
                                ImGui::SliderFloat("Base stretch mult", &state.params.base_stretch_mult, 0.0f, 2.0f, "%.3f");
                                ImGui::EndDisabled();
                                state.params.base_stretch_mult = std::clamp(state.params.base_stretch_mult, 0.0f, 2.0f);

                                ImGui::Checkbox("Bend (dihedral)", &state.params.bend_enabled);
                                ImGui::BeginDisabled(!state.params.bend_enabled);
                                ImGui::InputFloat("Dihedral stiffness", &state.params.bend_stiffness, 0.1f, 1.0f, "%.6f");
                                state.params.bend_stiffness = std::max(state.params.bend_stiffness, 0.0f);

                                ImGui::Checkbox("Iso bend", &state.params.iso_bend_enabled);
                                ImGui::BeginDisabled(!state.params.iso_bend_enabled);
                                ImGui::InputFloat("Iso bend stiffness", &state.params.iso_bend_stiffness, 0.1f, 1.0f, "%.6f");
                                ImGui::EndDisabled();
                                state.params.iso_bend_stiffness = std::max(state.params.iso_bend_stiffness, 0.0f);
                                ImGui::EndDisabled();

                                ImGui::Checkbox("Collision", &state.params.collision_enabled);
                                ImGui::BeginDisabled(!state.params.collision_enabled);
                                ImGui::InputFloat("Collision stiffness", &state.params.collision_stiffness, 0.1f, 1.0f, "%.6f");
                                ImGui::InputFloat("Thickness", &state.params.collision_thickness, 0.0005f, 0.002f, "%.6f");
                                ImGui::EndDisabled();
                                state.params.collision_stiffness = std::max(state.params.collision_stiffness, 0.0f);
                                state.params.collision_thickness = std::max(state.params.collision_thickness, 0.0f);

                                ImGui::TreePop();
                            }

                            if (ImGui::TreeNodeEx("Wind", ImGuiTreeNodeFlags_DefaultOpen))
                            {
                                ImGui::InputFloat("Wind force", &state.params.wind_force, 0.1f, 1.0f, "%.6f");
                                state.params.wind_force = std::max(state.params.wind_force, 0.0f);

                                ImGui::InputFloat("Wind yaw (deg)", &state.params.wind_yaw_deg, 1.0f, 5.0f, "%.3f");
                                ImGui::InputFloat("Wind pitch (deg)", &state.params.wind_pitch_deg, 1.0f, 5.0f, "%.3f");

                                ImGui::InputFloat("Random space", &state.params.wind_random_space, 0.05f, 0.25f, "%.6f");
                                ImGui::InputFloat("Random time", &state.params.wind_random_time, 0.05f, 0.25f, "%.6f");
                                state.params.wind_random_space = std::max(state.params.wind_random_space, 0.0f);
                                state.params.wind_random_time = std::max(state.params.wind_random_time, 0.0f);

                                ImGui::TreePop();
                            }

                            ImGui::Checkbox("Friction", &state.params.enable_friction);
                            if (state.params.enable_friction)
                            {
                                ImGui::InputFloat("mu", &state.params.friction_mu, 0.01f, 0.1f, "%.6f");
                                state.params.friction_mu = std::max(state.params.friction_mu, 0.0f);
                            }

                            ImGui::InputFloat("Damping", &state.params.damping, 0.001f, 0.01f, "%.6f");
                            state.params.damping = std::clamp(state.params.damping, 0.0f, 0.99f);
                        }
                        else
                        {
                            auto& p = state.xpbd_params;

                            ImGui::InputFloat("dt", &p.dt, 0.001f, 0.01f, "%.6f");
                            p.dt = std::max(p.dt, 1e-6f);

                            ImGui::InputFloat("Simulation time (s)", &p.duration_seconds, 0.1f, 1.0f, "%.3f");
                            p.duration_seconds = std::max(p.duration_seconds, 0.0f);

                            ImGui::InputInt("Iterations", &p.solver_iterations);
                            p.solver_iterations = std::max(p.solver_iterations, 1);

                            ImGui::InputInt("Substeps", &p.substeps);
                            p.substeps = std::clamp(p.substeps, 1, 16);

                            ImGui::InputFloat("Gravity", &p.gravity, 0.5f, 5.0f, "%.3f");

                            ImGui::SliderFloat("SMPL geometry kept", &p.smpl_geometry_percent, 1.0f, 100.0f, "%.0f%%");
                            p.smpl_geometry_percent = std::clamp(p.smpl_geometry_percent, 1.0f, 100.0f);

                            if (ImGui::TreeNodeEx("Constraints", ImGuiTreeNodeFlags_DefaultOpen))
                            {
                                ImGui::Checkbox("Stretch", &p.stretch_enabled);
                                ImGui::BeginDisabled(!p.stretch_enabled);
                                ImGui::InputFloat("Stretch compliance", &p.stretch_compliance, 1e-7f, 1e-6f, "%.8f");
                                ImGui::EndDisabled();
                                p.stretch_compliance = std::max(p.stretch_compliance, 0.0f);

                                ImGui::BeginDisabled(!p.stretch_enabled);
                                ImGui::SliderFloat("Base stretch mult", &p.base_stretch_mult, 0.0f, 2.0f, "%.3f");
                                ImGui::EndDisabled();
                                p.base_stretch_mult = std::clamp(p.base_stretch_mult, 0.0f, 2.0f);

                                ImGui::Checkbox("Bend", &p.bend_enabled);
                                ImGui::BeginDisabled(!p.bend_enabled);
                                ImGui::InputFloat("Bend compliance", &p.bend_compliance, 1e-7f, 1e-6f, "%.8f");
                                ImGui::EndDisabled();
                                p.bend_compliance = std::max(p.bend_compliance, 0.0f);

                                ImGui::Checkbox("Collision", &p.collision_enabled);
                                ImGui::BeginDisabled(!p.collision_enabled);
                                ImGui::InputFloat("Collision compliance", &p.collision_compliance, 1e-7f, 1e-6f, "%.8f");
                                ImGui::InputFloat("Thickness", &p.collision_thickness, 0.0005f, 0.002f, "%.6f");
                                ImGui::EndDisabled();
                                p.collision_compliance = std::max(p.collision_compliance, 0.0f);
                                p.collision_thickness = std::max(p.collision_thickness, 0.0f);

                                ImGui::Checkbox("Self collision", &p.self_collision_enabled);
                                ImGui::BeginDisabled(!p.self_collision_enabled);
                                ImGui::InputFloat("Self collision compliance", &p.self_collision_compliance, 1e-7f, 1e-6f, "%.8f");
                                ImGui::InputFloat("Self collision thickness", &p.self_collision_thickness, 0.0005f, 0.002f, "%.6f");
                                ImGui::EndDisabled();
                                p.self_collision_compliance = std::max(p.self_collision_compliance, 0.0f);
                                p.self_collision_thickness = std::max(p.self_collision_thickness, 0.0f);

                                ImGui::TreePop();
                            }

                            if (ImGui::TreeNodeEx("Wind", ImGuiTreeNodeFlags_DefaultOpen))
                            {
                                ImGui::InputFloat("Wind force", &p.wind_force, 0.1f, 1.0f, "%.6f");
                                p.wind_force = std::max(p.wind_force, 0.0f);

                                ImGui::InputFloat("Wind yaw (deg)", &p.wind_yaw_deg, 1.0f, 5.0f, "%.3f");
                                ImGui::InputFloat("Wind pitch (deg)", &p.wind_pitch_deg, 1.0f, 5.0f, "%.3f");

                                ImGui::InputFloat("Random space", &p.wind_random_space, 0.05f, 0.25f, "%.6f");
                                ImGui::InputFloat("Random time", &p.wind_random_time, 0.05f, 0.25f, "%.6f");
                                p.wind_random_space = std::max(p.wind_random_space, 0.0f);
                                p.wind_random_time = std::max(p.wind_random_time, 0.0f);

                                ImGui::TreePop();
                            }

                            ImGui::Checkbox("Friction", &p.enable_friction);
                            if (p.enable_friction)
                            {
                                ImGui::InputFloat("mu", &p.friction_mu, 0.01f, 0.1f, "%.6f");
                                p.friction_mu = std::max(p.friction_mu, 0.0f);
                            }

                            ImGui::InputFloat("Damping", &p.damping, 0.001f, 0.01f, "%.6f");
                            p.damping = std::clamp(p.damping, 0.0f, 0.99f);
                        }

                        ImGui::InputInt("Max Frames (cap)", &state.controls.max_frames);
                        state.controls.max_frames = std::clamp(state.controls.max_frames, 1, 1000000);
                        ImGui::TreePop();
                    }

                    ImGui::Separator();

                    ImGui::Checkbox("Preview", &state.controls.preview);
                    if (!state.controls.simulating)
                    {
                        const bool start_button = ImGui::Button("Start");
                        const bool start_enter =
                            ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)
                            && !ImGui::GetIO().WantTextInput
                            && !ImGui::IsAnyItemActive()
                            && (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeyPadEnter));

                        if (start_button || start_enter)
                        {
                            state.cloth_simulate_request_id++;
                            state.controls.simulating = true;
                            state.controls.playing = false;
                            state.controls.current_frame = 0;
                        }
                    }
                    else
                    {
                        ImGui::TextUnformatted("Simulating...");
                        const int done = std::max(0, state.controls.sim_steps_done);
                        const int total = std::max(0, state.controls.sim_steps_total);
                        const float frac = (total > 0) ? (float)done / (float)total : 0.0f;

                        const ImGuiStyle& style = ImGui::GetStyle();
                        const float stop_w = ImGui::CalcTextSize("Stop").x + 2.0f * style.FramePadding.x;
                        const float avail_w = ImGui::GetContentRegionAvail().x;
                        const float bar_w = std::max(0.0f, avail_w - stop_w - style.ItemSpacing.x);
                        ImGui::ProgressBar(std::clamp(frac, 0.0f, 1.0f), ImVec2(bar_w, 0.0f));
                        ImGui::SameLine();
                        if (ImGui::Button("Stop"))
                        {
                            state.cloth_cancel_simulation_request_id++;
                        }
                        if (total > 0)
                            ImGui::Text("%d / %d steps", done, total);
                        if (!state.controls.sim_stage.empty())
                            ImGui::Text("Step: %s", state.controls.sim_stage.c_str());
                    }

                    ImGui::Separator();

                    ImGui::Text("Status: %s", state.controls.simulating ? "Simulating" : (state.controls.frame_count > 0 ? "Ready" : "No simulation"));
                    if (state.controls.frame_count > 0)
                        ImGui::Text("Frames: %d", state.controls.frame_count);

                    const bool have_frames = (state.controls.frame_count > 0);
                    if (have_frames)
                    {
                        const bool frame_changed = ImGui::SliderInt("Frame", &state.controls.current_frame, 0, std::max(0, frame_max));
                        state.controls.current_frame = std::clamp(state.controls.current_frame, 0, std::max(0, frame_max));
                        if (frame_changed)
                            state.controls.playing = false;

                        ImGui::InputFloat("FPS", &state.controls.fps, 1.0f, 10.0f, "%.3f");
                        state.controls.fps = std::clamp(state.controls.fps, 0.1f, 10000.0f);

                        ImGui::BeginDisabled(state.controls.simulating);

                        if (ImGui::Button("<<"))
                        {
                            state.controls.playing = false;
                            state.controls.current_frame = 0;
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("<"))
                        {
                            state.controls.playing = false;
                            state.controls.current_frame = std::max(0, state.controls.current_frame - 1);
                        }
                        ImGui::SameLine();

                        const char* play_label = state.controls.playing ? "II##playpause" : u8"▶##playpause";
                        if (ImGui::Button(play_label))
                            state.controls.playing = !state.controls.playing;

                        ImGui::SameLine();
                        if (ImGui::Button(">"))
                        {
                            state.controls.playing = false;
                            state.controls.current_frame = std::min(std::max(0, frame_max), state.controls.current_frame + 1);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button(">>"))
                        {
                            state.controls.playing = false;
                            state.controls.current_frame = std::max(0, frame_max);
                        }

                        ImGui::EndDisabled();
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
