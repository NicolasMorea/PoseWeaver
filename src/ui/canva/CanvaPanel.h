#pragma once

#include "app/AppState.h"

namespace poseweaver { class Renderer; }

namespace poseweaver::ui_canva
{
    // Draws the "canva" tab (brush + layers + canvas view).
    // Keeps its own internal persistent state.
    void draw_canva_tab(AppState& state, Renderer& renderer);

    // Draws the lower "Sequence" tab, sharing the canva state for frame previews.
    void draw_sequence_tab(AppState& state, Renderer& renderer);
}
