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


        // Left: drawing canvas view.
#include "ui/canva/CanvaPanelCanvasView.inc"

        ImGui::SameLine();

        // Right: brush + layers controls.
#include "ui/canva/CanvaPanelControls.inc"
