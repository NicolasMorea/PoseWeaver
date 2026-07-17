#include "ui/canva/CanvaPanel.h"
#include "ui/canva/CanvaPanelInternal.h"

#include "render/Renderer.h"

#include <algorithm>
#include <imgui.h>

namespace poseweaver::ui_canva
{
    CanvaState& canva_state()
    {
        static CanvaState canva;
        return canva;
    }

    void canva_ensure_initialized(AppState& state, CanvaState& canva)
    {
        if (!canva.layers.empty())
            return;

        canva.brush_radius = state.canva_brush_radius;
        canva.brush_color = ImVec4(
            state.canva_brush_color[0],
            state.canva_brush_color[1],
            state.canva_brush_color[2],
            state.canva_brush_color[3]);

        canva.next_layer_id = 1;
        canva.layers.emplace_back(canva_make_layer(canva.width, canva.height, canva.next_layer_id++, "Layer 1"));
        canva.active_layer = 0;
        canva.zoom = 1.0f;
        canva.pan = ImVec2(0.0f, 0.0f);
        canva.tool = CanvaTool::Brush;
        canva.last_undo_request_id = state.canva_undo_request_id;
    }

    void draw_canva_tab(AppState& state, Renderer& renderer)
    {
        CanvaState& canva = canva_state();
        canva_ensure_initialized(state, canva);
        canva.active_layer = std::clamp(canva.active_layer, 0, std::max(0, (int)canva.layers.size() - 1));

        update_canva_frame(state, canva, renderer);

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float spacing_x = ImGui::GetStyle().ItemSpacing.x;
        const bool show_generate_cloth_steps =
            canva.generate_cloth_active
            || canva.generate_cloth_status == "Done"
            || canva.generate_cloth_status == "Failed";
        float right_w = 420.0f;
        if (avail.x < right_w + 220.0f)
            right_w = std::max(200.0f, avail.x * 0.35f);
        const float left_w = std::max(1.0f, avail.x - right_w - spacing_x);

        draw_canva_canvas_view(state, canva, renderer, left_w, show_generate_cloth_steps);
        ImGui::SameLine();
        draw_canva_controls(state, canva, renderer);
    }
}
