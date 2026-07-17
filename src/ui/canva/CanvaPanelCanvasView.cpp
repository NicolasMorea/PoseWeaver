#include "ui/canva/CanvaPanelInternal.h"

#include "ui/canva/CanvaPreprocess.h"
#include "core/Paths.h"
#include "render/Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cerrno>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tinyfiledialogs.h>
#include <cnpy.h>

namespace poseweaver::ui_canva
{
    void draw_canva_canvas_view(AppState& state, CanvaState& canva, Renderer& renderer, float left_w, bool show_generate_cloth_steps)
    {
    // Left: drawing canvas view.
    ImGui::BeginChild("##canva_view", ImVec2(left_w, 0.0f), false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        const ImVec2 view_pos = ImGui::GetCursorScreenPos();
        const ImVec2 view_size = ImGui::GetContentRegionAvail();

        if (view_size.x > 1.0f && view_size.y > 1.0f)
        {
            ImGui::InvisibleButton("##canva_invisible", view_size,
                ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);

            const bool hovered = ImGui::IsItemHovered();
            ImGuiIO& io = ImGui::GetIO();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const bool shift = io.KeyShift;
            const bool sequence_preview = canva.sequence_preview_active;
            auto& draw_layers = sequence_preview ? canva.sequence_preview_layers : canva.layers;
            for (auto& layer : draw_layers)
            {
                if (layer.dirty)
                    canva_upload_layer(layer, canva.width, canva.height);
            }
            const std::vector<CanvaLayer>& display_layers =
                draw_layers;

            const ImVec2 view_center = ImVec2(view_pos.x + view_size.x * 0.5f, view_pos.y + view_size.y * 0.5f);

            // Zoom (mouse wheel) around cursor.
            if (hovered)
            {
                const float wheel = io.MouseWheel;
                if (wheel != 0.0f)
                {
                    const float zoom_before = canva.zoom;
                    const float zoom_mul = (wheel > 0.0f) ? 1.12f : (1.0f / 1.12f);
                    canva.zoom = std::clamp(canva.zoom * zoom_mul, 0.10f, 20.0f);
                    const float zoom_after = canva.zoom;

                    const ImVec2 mouse = io.MousePos;
                    const ImVec2 canvas_size_before = ImVec2((float)canva.width * zoom_before, (float)canva.height * zoom_before);
                    const ImVec2 origin_centered_before = ImVec2(
                        view_center.x - canvas_size_before.x * 0.5f,
                        view_center.y - canvas_size_before.y * 0.5f);
                    const ImVec2 origin_before = ImVec2(origin_centered_before.x + canva.pan.x, origin_centered_before.y + canva.pan.y);
                    const ImVec2 p_canvas = ImVec2(
                        (zoom_before > 1e-6f) ? ((mouse.x - origin_before.x) / zoom_before) : 0.0f,
                        (zoom_before > 1e-6f) ? ((mouse.y - origin_before.y) / zoom_before) : 0.0f);

                    const ImVec2 canvas_size_after = ImVec2((float)canva.width * zoom_after, (float)canva.height * zoom_after);
                    const ImVec2 origin_centered_after = ImVec2(
                        view_center.x - canvas_size_after.x * 0.5f,
                        view_center.y - canvas_size_after.y * 0.5f);
                    const ImVec2 origin_after = ImVec2(mouse.x - p_canvas.x * zoom_after, mouse.y - p_canvas.y * zoom_after);

                    canva.pan.x = origin_after.x - origin_centered_after.x;
                    canva.pan.y = origin_after.y - origin_centered_after.y;
                }
            }

            // Pan (middle mouse drag).
            if (hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
            {
                const ImVec2 d = io.MouseDelta;
                canva.pan.x += d.x;
                canva.pan.y += d.y;
            }

            const ImVec2 canvas_size = ImVec2((float)canva.width * canva.zoom, (float)canva.height * canva.zoom);
            const ImVec2 origin_centered = ImVec2(view_center.x - canvas_size.x * 0.5f, view_center.y - canvas_size.y * 0.5f);
            const ImVec2 origin = ImVec2(origin_centered.x + canva.pan.x, origin_centered.y + canva.pan.y);
            const ImVec2 p0 = origin;
            const ImVec2 p1 = ImVec2(origin.x + canvas_size.x, origin.y + canvas_size.y);

            const ImVec2 mouse = io.MousePos;
            const ImVec2 local = ImVec2((mouse.x - origin.x) / canva.zoom, (mouse.y - origin.y) / canva.zoom);
            const bool in_canvas = (local.x >= 0.0f && local.y >= 0.0f && local.x < (float)canva.width && local.y < (float)canva.height);

            dl->PushClipRect(view_pos, ImVec2(view_pos.x + view_size.x, view_pos.y + view_size.y), true);

            // Canvas background (white).
            dl->AddRectFilled(p0, p1, IM_COL32(255, 255, 255, 255));

            // Clip layers to canvas bounds.
            dl->PushClipRect(p0, p1, true);

            // Draw layers bottom->top (vector is top->bottom).
            for (int i = (int)display_layers.size() - 1; i >= 0; --i)
            {
                const auto& layer = display_layers[(size_t)i];
                if (!layer.enabled || layer.texture_id == 0)
                    continue;
                ImTextureID tid = (ImTextureID)(intptr_t)layer.texture_id;

                const ImVec2 lp0 = ImVec2(p0.x + layer.offset.x * canva.zoom, p0.y + layer.offset.y * canva.zoom);
                const float s = std::max(1e-6f, layer.scale);
                const ImVec2 lp1 = ImVec2(lp0.x + canvas_size.x * s, lp0.y + canvas_size.y * s);
                dl->AddImage(tid, lp0, lp1);
            }

            dl->PopClipRect();

            dl->AddRect(p0, p1, IM_COL32(0, 0, 0, 140));

            if (show_generate_cloth_steps)
            {
                const float steps_w = 360.0f;
                const float min_x = view_pos.x + 8.0f;
                const float max_x = std::max(min_x, view_pos.x + view_size.x - steps_w - 8.0f);
                const ImVec2 steps_anchor(
                    std::clamp(
                        p1.x + 28.0f * canva.zoom,
                        min_x,
                        max_x),
                    p0.y + 16.0f * canva.zoom);
                draw_generate_cloth_steps_in_view(state, canva, dl, steps_anchor);
            }

            if (sequence_preview)
            {
                canva.drawing = false;
                canva.sizing = false;
                canva.transforming = false;
                canva.scaling = false;
                canva.transform_layer = -1;
            }

            // Transform tool: pick + drag active layer.
            if (!sequence_preview && canva.tool == CanvaTool::Transform)
            {
                int hovered_handle = -1;
                ImVec2 handle_pos[4] = { ImVec2(0, 0), ImVec2(0, 0), ImVec2(0, 0), ImVec2(0, 0) };
                CanvaBounds active_bounds;

                if (!canva.layers.empty())
                {
                    // Draw active layer bounds.
                    CanvaLayer& layer = canva.layers[(size_t)canva.active_layer];
                    active_bounds = canva_layer_content_bounds(layer, canva.width, canva.height);

                    const float s = std::max(1e-6f, layer.scale);
                    const ImVec2 tl_canvas = ImVec2(layer.offset.x + (float)active_bounds.min_x * s, layer.offset.y + (float)active_bounds.min_y * s);
                    const ImVec2 br_canvas = ImVec2(layer.offset.x + (float)active_bounds.max_x_excl * s, layer.offset.y + (float)active_bounds.max_y_excl * s);
                    const ImVec2 lp0 = ImVec2(p0.x + tl_canvas.x * canva.zoom, p0.y + tl_canvas.y * canva.zoom);
                    const ImVec2 lp1 = ImVec2(p0.x + br_canvas.x * canva.zoom, p0.y + br_canvas.y * canva.zoom);
                    dl->AddRect(lp0, lp1, IM_COL32(40, 40, 40, 200), 0.0f, 0, 1.5f);

                    // Corner scale handles.
                    handle_pos[0] = lp0;                  // TL
                    handle_pos[1] = ImVec2(lp1.x, lp0.y); // TR
                    handle_pos[2] = lp1;                  // BR
                    handle_pos[3] = ImVec2(lp0.x, lp1.y); // BL

                    const float hr = 6.0f;
                    const float hr2 = hr * hr;
                    for (int i = 0; i < 4; ++i)
                    {
                        const float dx = mouse.x - handle_pos[i].x;
                        const float dy = mouse.y - handle_pos[i].y;
                        if (dx * dx + dy * dy <= hr2)
                            hovered_handle = i;
                    }

                    for (int i = 0; i < 4; ++i)
                    {
                        const bool hi = (i == hovered_handle);
                        const ImU32 fill = hi ? IM_COL32(30, 30, 30, 255) : IM_COL32(50, 50, 50, 230);
                        const ImU32 outl = IM_COL32(220, 220, 220, 230);
                        const ImVec2 a = ImVec2(handle_pos[i].x - hr, handle_pos[i].y - hr);
                        const ImVec2 b = ImVec2(handle_pos[i].x + hr, handle_pos[i].y + hr);
                        dl->AddRectFilled(a, b, fill, 2.0f);
                        dl->AddRect(a, b, outl, 2.0f, 0, 1.0f);
                    }
                }

                // Pick layer on click, but don't steal clicks intended for the scale handles.
                if (hovered && in_canvas && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hovered_handle < 0)
                {
                    const int picked = canva_pick_layer_at(canva, local);
                    if (picked >= 0)
                        canva.active_layer = picked;
                }

                if (hovered && in_canvas && ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDown(ImGuiMouseButton_Middle))
                {
                    if (!canva.transforming)
                    {
                        // If starting from a handle, transform the already-active layer.
                        if (hovered_handle < 0)
                        {
                            const int picked = canva_pick_layer_at(canva, local);
                            if (picked >= 0)
                                canva.active_layer = picked;
                        }

                        canva.transform_layer = canva.active_layer;
                        canva.transforming = true;
                        canva.scaling = (hovered_handle >= 0);
                        canva.transform_start_mouse = mouse;

                        CanvaLayer& layer = canva.layers[(size_t)canva.transform_layer];
                        canva.transform_start_offset = layer.offset;
                        canva.transform_start_scale = layer.scale;
                        canva_push_undo(canva, layer.id, layer.pixels, layer.offset, layer.scale);

                        if (canva.scaling)
                        {
                            const ImVec2 min_px = ImVec2((float)active_bounds.min_x, (float)active_bounds.min_y);
                            const ImVec2 max_px = ImVec2((float)active_bounds.max_x_excl, (float)active_bounds.max_y_excl);

                            // dragged corner (handle) and opposite anchor corner in *layer pixel* coords.
                            if (hovered_handle == 0) // TL
                            {
                                canva.transform_drag_px = min_px;
                                canva.transform_anchor_px = max_px;
                            }
                            else if (hovered_handle == 1) // TR
                            {
                                canva.transform_drag_px = ImVec2(max_px.x, min_px.y);
                                canva.transform_anchor_px = ImVec2(min_px.x, max_px.y);
                            }
                            else if (hovered_handle == 2) // BR
                            {
                                canva.transform_drag_px = max_px;
                                canva.transform_anchor_px = min_px;
                            }
                            else // BL
                            {
                                canva.transform_drag_px = ImVec2(min_px.x, max_px.y);
                                canva.transform_anchor_px = ImVec2(max_px.x, min_px.y);
                            }

                            const float ss = std::max(1e-6f, canva.transform_start_scale);
                            canva.transform_anchor_canvas = ImVec2(
                                canva.transform_start_offset.x + canva.transform_anchor_px.x * ss,
                                canva.transform_start_offset.y + canva.transform_anchor_px.y * ss);
                        }
                    }

                    if (canva.transform_layer >= 0 && canva.transform_layer < (int)canva.layers.size())
                    {
                        CanvaLayer& layer = canva.layers[(size_t)canva.transform_layer];
                        if (!canva.scaling)
                        {
                            const ImVec2 d_screen = ImVec2(mouse.x - canva.transform_start_mouse.x, mouse.y - canva.transform_start_mouse.y);
                            const ImVec2 d_canvas = ImVec2(d_screen.x / canva.zoom, d_screen.y / canva.zoom);
                            layer.offset = ImVec2(canva.transform_start_offset.x + d_canvas.x, canva.transform_start_offset.y + d_canvas.y);
                        }
                        else
                        {
                            // Scale uniformly by dragging a corner handle. The opposite corner stays fixed.
                            const ImVec2 mouse_canvas = local;
                            const ImVec2 dp = ImVec2(canva.transform_drag_px.x - canva.transform_anchor_px.x,
                                canva.transform_drag_px.y - canva.transform_anchor_px.y);
                            const ImVec2 dc = ImVec2(mouse_canvas.x - canva.transform_anchor_canvas.x,
                                mouse_canvas.y - canva.transform_anchor_canvas.y);

                            float s_new = canva.transform_start_scale;
                            const bool have_x = std::abs(dp.x) > 1e-6f;
                            const bool have_y = std::abs(dp.y) > 1e-6f;
                            float sx = 0.0f;
                            float sy = 0.0f;
                            if (have_x)
                                sx = dc.x / dp.x;
                            if (have_y)
                                sy = dc.y / dp.y;

                            if (have_x && have_y)
                                s_new = (std::abs(sx) > std::abs(sy)) ? sx : sy;
                            else if (have_x)
                                s_new = sx;
                            else if (have_y)
                                s_new = sy;

                            s_new = std::clamp(std::abs(s_new), 0.05f, 20.0f);

                            layer.scale = s_new;
                            layer.offset = ImVec2(
                                canva.transform_anchor_canvas.x - canva.transform_anchor_px.x * s_new,
                                canva.transform_anchor_canvas.y - canva.transform_anchor_px.y * s_new);
                        }
                    }
                }
                else
                {
                    canva.transforming = false;
                    canva.scaling = false;
                    canva.transform_layer = -1;
                }

                // No drawing in transform tool.
                canva.drawing = false;
                canva.sizing = false;
            }

            // Brush tool: Shift+drag changes brush size.
            if (!sequence_preview && canva.tool == CanvaTool::Brush)
            {
                const bool left_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);
                const bool middle_down = ImGui::IsMouseDown(ImGuiMouseButton_Middle);

                if (hovered && in_canvas && shift && left_down && !middle_down)
                {
                    if (!canva.sizing)
                    {
                        canva.sizing = true;
                        canva.size_start_mouse = mouse;
                        canva.size_start_radius = canva.brush_radius;
                    }

                    const float dx_canvas = (mouse.x - canva.size_start_mouse.x) / std::max(1e-6f, canva.zoom);
                    const float new_radius = canva.size_start_radius + dx_canvas;
                    canva.brush_radius = std::clamp(new_radius, 0.5f, 250.0f);

                    // While sizing, do not draw.
                    canva.drawing = false;
                }
                else
                {
                    canva.sizing = false;
                }

                // Drawing (left mouse).
                if (hovered && in_canvas && left_down && !middle_down && !shift)
                {
                    CanvaLayer& layer = canva.layers[(size_t)canva.active_layer];
                    const float s = std::max(1e-6f, layer.scale);
                    const ImVec2 layer_local = ImVec2((local.x - layer.offset.x) / s, (local.y - layer.offset.y) / s);
                    const bool in_layer = (layer_local.x >= 0.0f && layer_local.y >= 0.0f
                        && layer_local.x < (float)canva.width && layer_local.y < (float)canva.height);

                    if (in_layer)
                    {
                        if (!canva.drawing)
                        {
                            // Snapshot for undo at the start of a stroke.
                            canva_push_undo(canva, layer.id, layer.pixels, layer.offset, layer.scale);

                            canva.drawing = true;
                            canva.last_canvas_pos = layer_local;
                            const float r_layer = canva.brush_radius / s;
                            canva_stamp_circle(layer, canva.width, canva.height, layer_local.x, layer_local.y, r_layer, canva.brush_color);
                        }
                        else
                        {
                            const float r_layer = canva.brush_radius / s;
                            canva_draw_segment(layer, canva.width, canva.height, canva.last_canvas_pos, layer_local, r_layer, canva.brush_color);
                            canva.last_canvas_pos = layer_local;
                        }
                    }
                    else
                    {
                        canva.drawing = false;
                    }
                }
                else
                {
                    canva.drawing = false;
                }

                // Custom cursor: brush size preview circle.
                if (hovered && in_canvas)
                {
                    const float r_px = std::max(0.5f, canva.brush_radius * canva.zoom);
                    const ImU32 outline = IM_COL32(0, 0, 0, 210);

                    ImVec4 solid = canva.brush_color;
                    solid.w = 1.0f;
                    const ImU32 fill = ImGui::ColorConvertFloat4ToU32(solid);

                    dl->AddCircle(mouse, r_px, outline, 0, 1.25f);
                    dl->AddCircleFilled(mouse, 1.5f, outline);
                    dl->AddCircleFilled(mouse, 1.0f, fill);
                }
            }

            dl->PopClipRect();

            // Upload textures if needed (after painting).
            auto& upload_layers = sequence_preview ? canva.sequence_preview_layers : canva.layers;
            for (auto& layer : upload_layers)
            {
                if (layer.dirty)
                    canva_upload_layer(layer, canva.width, canva.height);
            }
        }
    }
    ImGui::EndChild();

    }
}
