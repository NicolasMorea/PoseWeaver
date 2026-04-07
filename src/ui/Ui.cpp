#include "ui/Ui.h"

#include "render/Renderer.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstddef>
#include <string>
#include <imgui.h>

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

        // Left controls
        if (show_controls && left_size.x > 1.0f && left_size.y > 1.0f)
        {
            ImGui::SetNextWindowPos(left_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(left_size, ImGuiCond_Always);
			draw_controls(state, assets, renderer);
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
            draw_cloth_view(state, renderer, frame);
        }
        else
        {
            state.cloth_view.valid = false;
        }

        // Bottom console
        if (state.show_console)
        {
            // Horizontal splitter between top content and console
            ImGui::SetNextWindowPos(splitter_h_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(splitter_h_size, ImGuiCond_Always);

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
            ImGui::Begin("##console_splitter", nullptr, splitter_window_flags());

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

    void Ui::draw_main_menu(AppState& state, const AssetIndex& assets)
    {
        (void)assets;

        if (ImGui::BeginMainMenuBar())
        {
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

    void Ui::draw_cloth_view(AppState& state, Renderer& renderer, const UiFrameInfo& frame)
    {
        // This window owns the main 3D viewport *and* the UV projection view via tabs.
        // The renderer will scissor its draw to state.cloth_view when valid.
        ImGui::Begin("Scene View", &state.show_cloth_view, fixed_panel_flags() | ImGuiWindowFlags_NoBackground);

        const ImVec2 avail_outer = ImGui::GetContentRegionAvail();
        if (avail_outer.x <= 1.0f || avail_outer.y <= 1.0f)
        {
            state.cloth_view.valid = false;
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("##center_tabs"))
        {
            // Tab 1: 3D scene rendering
            if (ImGui::BeginTabItem("Scene View"))
            {
                const ImVec2 top_left = ImGui::GetCursorScreenPos();
                const ImVec2 avail = ImGui::GetContentRegionAvail();

                ImGui::InvisibleButton("scene_view_canvas", avail);

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

                ImGui::EndTabItem();
            }

            // Tab 2: UV projection image
            if (ImGui::BeginTabItem("Uv projection"))
            {
                // Hide 3D rendering while in this tab by scissoring it to a zero-sized viewport.
                // (Renderer clears the framebuffer before drawing; ImGui will draw the UV image after.)
                state.cloth_view = ViewportRect{ 0, 0, 0, 0, true };

                const auto info = renderer.uv_debug_texture_info();
                if (!info.valid)
                {
                    ImGui::TextUnformatted("UV image not generated yet.");
                    ImGui::TextUnformatted("Generate via: Controls -> SMPL -> Generate UV Image.");
                }
                else
                {
                    if (renderer.smpl_uv_layout_loaded())
                        ImGui::Text("UV layout: %s", renderer.smpl_uv_layout_source().c_str());
                    else
                        ImGui::TextUnformatted("UV layout: (missing) using fallback planar projection");

                    // Interactive canvas: wheel zoom + left-drag pan.
                    const ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                    const ImVec2 canvas_size = ImGui::GetContentRegionAvail();

                    if (canvas_size.x > 1.0f && canvas_size.y > 1.0f)
                    {
                        ImGui::InvisibleButton("uv_canvas", canvas_size,
                            ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);

                        const bool hovered = ImGui::IsItemHovered();
                        const bool active = ImGui::IsItemActive();
                        ImDrawList* dl = ImGui::GetWindowDrawList();

                        const float tex_w = (float)info.width;
                        const float tex_h = (float)info.height;
                        const float fit_scale = (tex_w > 0.0f && tex_h > 0.0f)
                            ? std::min(canvas_size.x / tex_w, canvas_size.y / tex_h)
                            : 1.0f;

                        // Pan with left-drag.
                        if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                        {
                            const ImVec2 d = ImGui::GetIO().MouseDelta;
                            state.uv_view_pan_x += d.x;
                            state.uv_view_pan_y += d.y;
                        }

                        // Zoom with mouse wheel (zoom about cursor).
                        if (hovered)
                        {
                            const float wheel = ImGui::GetIO().MouseWheel;
                            if (wheel != 0.0f)
                            {
                                const float zoom_before = state.uv_view_zoom;
                                const float zoom_mul = (wheel > 0.0f) ? 1.12f : (1.0f / 1.12f);
                                state.uv_view_zoom = std::clamp(state.uv_view_zoom * zoom_mul, 0.05f, 50.0f);
                                const float zoom_after = state.uv_view_zoom;

                                // Keep the point under the cursor stable.
                                const ImVec2 mouse = ImGui::GetIO().MousePos;
                                const ImVec2 canvas_center = ImVec2(canvas_pos.x + canvas_size.x * 0.5f, canvas_pos.y + canvas_size.y * 0.5f);
                                const ImVec2 centered_origin = ImVec2(
                                    canvas_center.x - 0.5f * tex_w * fit_scale * zoom_before,
                                    canvas_center.y - 0.5f * tex_h * fit_scale * zoom_before);
                                const ImVec2 origin_before = ImVec2(centered_origin.x + state.uv_view_pan_x, centered_origin.y + state.uv_view_pan_y);
                                const ImVec2 p_img = ImVec2(mouse.x - origin_before.x, mouse.y - origin_before.y);

                                // p_img scales with zoom; adjust pan to compensate.
                                const float s = (zoom_before > 0.0f) ? (zoom_after / zoom_before) : 1.0f;
                                const ImVec2 origin_after_centered = ImVec2(
                                    canvas_center.x - 0.5f * tex_w * fit_scale * zoom_after,
                                    canvas_center.y - 0.5f * tex_h * fit_scale * zoom_after);
                                const ImVec2 origin_after = ImVec2(mouse.x - p_img.x * s, mouse.y - p_img.y * s);

                                state.uv_view_pan_x = origin_after.x - origin_after_centered.x;
                                state.uv_view_pan_y = origin_after.y - origin_after_centered.y;
                            }
                        }

                        const float zoom = state.uv_view_zoom;
                        const ImVec2 canvas_center = ImVec2(canvas_pos.x + canvas_size.x * 0.5f, canvas_pos.y + canvas_size.y * 0.5f);
                        const ImVec2 img_size = ImVec2(tex_w * fit_scale * zoom, tex_h * fit_scale * zoom);
                        const ImVec2 origin = ImVec2(
                            canvas_center.x - img_size.x * 0.5f + state.uv_view_pan_x,
                            canvas_center.y - img_size.y * 0.5f + state.uv_view_pan_y);

                        // Draw image.
                        ImTextureID tid = (ImTextureID)(intptr_t)info.texture_id;
                        dl->AddImage(tid, origin, ImVec2(origin.x + img_size.x, origin.y + img_size.y));

                        // If a cloth is loaded and we have a garment→SMPL mapping, draw the cloth in UV space too.
                        // (Semi-transparent faces + distinct wireframe color.)
                        const auto& cloth_tris = renderer.garment_uv_triangles();
                        if (!cloth_tris.empty())
                        {
                            // Filled triangles can be expensive for very dense meshes; keep it bounded.
                            constexpr size_t kMaxFillTris = 20000;
                            if (cloth_tris.size() <= kMaxFillTris)
                            {
                                const ImU32 cloth_fill_col = IM_COL32(0, 128, 255, 45);
                                for (const auto& t : cloth_tris)
                                {
                                    const float u0 = std::clamp(t.u0, 0.0f, 1.0f);
                                    const float v0 = std::clamp(t.v0, 0.0f, 1.0f);
                                    const float u1 = std::clamp(t.u1, 0.0f, 1.0f);
                                    const float v1 = std::clamp(t.v1, 0.0f, 1.0f);
                                    const float u2 = std::clamp(t.u2, 0.0f, 1.0f);
                                    const float v2 = std::clamp(t.v2, 0.0f, 1.0f);

                                    const ImVec2 a = ImVec2(origin.x + u0 * img_size.x, origin.y + (1.0f - v0) * img_size.y);
                                    const ImVec2 b = ImVec2(origin.x + u1 * img_size.x, origin.y + (1.0f - v1) * img_size.y);
                                    const ImVec2 c = ImVec2(origin.x + u2 * img_size.x, origin.y + (1.0f - v2) * img_size.y);
                                    dl->AddTriangleFilled(a, b, c, cloth_fill_col);
                                }
                            }
                        }

                        // Overlay SMPL UV wireframe in UV space (u,v in [0,1]).
                        const auto& segs = renderer.smpl_uv_wireframe_segments();
                        const ImU32 col = IM_COL32(0, 0, 0, 255);
                        constexpr float kThickness = 0.75f;
                        for (const auto& s : segs)
                        {
                            const float u0 = std::clamp(s.u0, 0.0f, 1.0f);
                            const float v0 = std::clamp(s.v0, 0.0f, 1.0f);
                            const float u1 = std::clamp(s.u1, 0.0f, 1.0f);
                            const float v1 = std::clamp(s.v1, 0.0f, 1.0f);

                            const ImVec2 a = ImVec2(origin.x + u0 * img_size.x, origin.y + (1.0f - v0) * img_size.y);
                            const ImVec2 b = ImVec2(origin.x + u1 * img_size.x, origin.y + (1.0f - v1) * img_size.y);
                            dl->AddLine(a, b, col, kThickness);
                        }

                        // Cloth UV wireframe on top for readability.
                        const auto& cloth_segs = renderer.garment_uv_wireframe_segments();
                        if (!cloth_segs.empty())
                        {
                            const ImU32 cloth_line_col = IM_COL32(0, 90, 220, 230);
                            constexpr float kClothThickness = 1.15f;
                            for (const auto& s : cloth_segs)
                            {
                                const float u0 = std::clamp(s.u0, 0.0f, 1.0f);
                                const float v0 = std::clamp(s.v0, 0.0f, 1.0f);
                                const float u1 = std::clamp(s.u1, 0.0f, 1.0f);
                                const float v1 = std::clamp(s.v1, 0.0f, 1.0f);

                                const ImVec2 a = ImVec2(origin.x + u0 * img_size.x, origin.y + (1.0f - v0) * img_size.y);
                                const ImVec2 b = ImVec2(origin.x + u1 * img_size.x, origin.y + (1.0f - v1) * img_size.y);
                                dl->AddLine(a, b, cloth_line_col, kClothThickness);
                            }
                        }

                        // Border for readability.
                        dl->AddRect(origin, ImVec2(origin.x + img_size.x, origin.y + img_size.y), IM_COL32(0, 0, 0, 100));
                    }
                }

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        else
        {
            state.cloth_view.valid = false;
        }

        ImGui::End();
    }

    void Ui::draw_controls(AppState& state, const AssetIndex& assets, Renderer& renderer)
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
            ImGui::Checkbox("Color by UV", &state.smpl_color_by_uv);

            if (ImGui::Button("Generate UV Image"))
            {
                state.uv_image_request_id++;
                log_line(state, "[SMPL] Generate UV Image requested");
            }

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
            ImGui::Checkbox("Show object mesh", &state.show_object_mesh);
            ImGui::Checkbox("Show garment mesh", &state.show_garment_mesh);

            // Only meaningful when we have a garment→SMPL surface mapping.
            if (renderer.garment_to_smpl_mapping_valid())
                ImGui::Checkbox("Color cloth by SMPL UV", &state.garment_color_by_smpl_uv);

            if (ImGui::Button("Process cloth (push out of SMPL)"))
            {
                state.garment_pushout_request_id++;
                log_line(state, "[Garment] Push-out requested");
            }

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
