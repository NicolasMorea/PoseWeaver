#pragma once

#include "app/AppState.h"

namespace poseweaver { class Renderer; }

namespace poseweaver::ui_canva
{
    // Advances canvas pipelines once per application frame, independently of tab visibility.
    void update_canva(AppState& state, Renderer& renderer);

    // Draws the "canva" tab (brush + layers + canvas view).
    // Keeps its own internal persistent state.
    void draw_canva_tab(AppState& state, Renderer& renderer);

    // Draws the Canva controls in the left-side "Fitting" tab.
    void draw_canva_fitting_controls(AppState& state, Renderer& renderer);

    // Draws the lower "Sequence" tab, sharing the canva state for frame previews.
    void draw_sequence_tab(AppState& state, Renderer& renderer);
}
