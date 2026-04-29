#pragma once

#include "app/AppState.h"

namespace poseweaver::ui_canva
{
    // Draws the "canva" tab (brush + layers + canvas view).
    // Keeps its own internal persistent state.
    void draw_canva_tab(AppState& state);
}
