#include "ui/Ui.h"

#include "render/Renderer.h"
#include "ui/UiInternal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <imgui.h>

namespace poseweaver
{
    void Ui::draw_cloth_view(AppState& state, Renderer& renderer, const UiFrameInfo& frame)
    {
        ImGui::Begin("Scene View", &state.show_cloth_view,
            ui_internal::fixed_panel_flags() | ImGuiWindowFlags_NoBackground);

        const ImVec2 avail_outer = ImGui::GetContentRegionAvail();
        if (avail_outer.x <= 1.0f || avail_outer.y <= 1.0f)
        {
            state.cloth_view.valid = false;
            ImGui::End();
            return;
        }

        if (!ImGui::BeginTabBar("##center_tabs"))
        {
            state.cloth_view.valid = false;
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabItem("Scene View"))
        {
            const ImVec2 top_left = ImGui::GetCursorScreenPos();
            const ImVec2 avail = ImGui::GetContentRegionAvail();

            ImGui::InvisibleButton("scene_view_canvas", avail);

            const ImGuiIO& io = ImGui::GetIO();
            const ImVec2 scale = io.DisplayFramebufferScale;

            const ImGuiViewport* vp = ImGui::GetMainViewport();
            const ImVec2 vp_pos = vp ? vp->Pos : ImVec2(0.0f, 0.0f);
            const ImVec2 local = ImVec2(top_left.x - vp_pos.x, top_left.y - vp_pos.y);

            const int x = static_cast<int>(local.x * scale.x);
            const int w = static_cast<int>(avail.x * scale.x);

            const int y_top = static_cast<int>(local.y * scale.y);
            const int h = static_cast<int>(avail.y * scale.y);
            const int y = frame.fb_height - (y_top + h);

            state.cloth_view = ViewportRect{ x, y, w, h, (w > 0 && h > 0) };

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Uv projection"))
        {
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

                    if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                    {
                        const ImVec2 d = ImGui::GetIO().MouseDelta;
                        state.uv_view_pan_x += d.x;
                        state.uv_view_pan_y += d.y;
                    }

                    if (hovered)
                    {
                        const float wheel = ImGui::GetIO().MouseWheel;
                        if (wheel != 0.0f)
                        {
                            const float zoom_before = state.uv_view_zoom;
                            const float zoom_mul = (wheel > 0.0f) ? 1.12f : (1.0f / 1.12f);
                            state.uv_view_zoom = std::clamp(state.uv_view_zoom * zoom_mul, 0.05f, 50.0f);
                            const float zoom_after = state.uv_view_zoom;

                            const ImVec2 mouse = ImGui::GetIO().MousePos;
                            const ImVec2 canvas_center =
                                ImVec2(canvas_pos.x + canvas_size.x * 0.5f, canvas_pos.y + canvas_size.y * 0.5f);
                            const ImVec2 centered_origin = ImVec2(
                                canvas_center.x - 0.5f * tex_w * fit_scale * zoom_before,
                                canvas_center.y - 0.5f * tex_h * fit_scale * zoom_before);
                            const ImVec2 origin_before =
                                ImVec2(centered_origin.x + state.uv_view_pan_x, centered_origin.y + state.uv_view_pan_y);
                            const ImVec2 p_img = ImVec2(mouse.x - origin_before.x, mouse.y - origin_before.y);

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
                    const ImVec2 canvas_center =
                        ImVec2(canvas_pos.x + canvas_size.x * 0.5f, canvas_pos.y + canvas_size.y * 0.5f);
                    const ImVec2 img_size = ImVec2(tex_w * fit_scale * zoom, tex_h * fit_scale * zoom);
                    const ImVec2 origin = ImVec2(
                        canvas_center.x - img_size.x * 0.5f + state.uv_view_pan_x,
                        canvas_center.y - img_size.y * 0.5f + state.uv_view_pan_y);

                    ImTextureID tid = (ImTextureID)(intptr_t)info.texture_id;
                    dl->AddImage(tid, origin, ImVec2(origin.x + img_size.x, origin.y + img_size.y));

                    const auto& cloth_tris = renderer.garment_uv_triangles();
                    if (!cloth_tris.empty())
                    {
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

                                const ImVec2 a =
                                    ImVec2(origin.x + u0 * img_size.x, origin.y + (1.0f - v0) * img_size.y);
                                const ImVec2 b =
                                    ImVec2(origin.x + u1 * img_size.x, origin.y + (1.0f - v1) * img_size.y);
                                const ImVec2 c =
                                    ImVec2(origin.x + u2 * img_size.x, origin.y + (1.0f - v2) * img_size.y);
                                dl->AddTriangleFilled(a, b, c, cloth_fill_col);
                            }
                        }
                    }

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

                            const ImVec2 a =
                                ImVec2(origin.x + u0 * img_size.x, origin.y + (1.0f - v0) * img_size.y);
                            const ImVec2 b =
                                ImVec2(origin.x + u1 * img_size.x, origin.y + (1.0f - v1) * img_size.y);
                            dl->AddLine(a, b, cloth_line_col, kClothThickness);
                        }
                    }

                    dl->AddRect(origin, ImVec2(origin.x + img_size.x, origin.y + img_size.y), IM_COL32(0, 0, 0, 100));
                }
            }

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
        ImGui::End();
    }
}
