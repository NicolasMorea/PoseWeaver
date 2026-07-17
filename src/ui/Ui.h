#pragma once

#include "app/AppState.h"
#include "core/Assets.h"

namespace poseweaver { class Renderer; }

namespace poseweaver
{
    struct UiFrameInfo
    {
        int fb_width = 0;
        int fb_height = 0;
    };

    class Ui
    {
    public:
        void draw(AppState& state, const AssetIndex& assets, Renderer& renderer, const UiFrameInfo& frame);

    private:
        void draw_main_menu(AppState& state, const AssetIndex& assets);
		void draw_cloth_view(AppState& state, Renderer& renderer, const UiFrameInfo& frame);
        void draw_controls(AppState& state, const AssetIndex& assets, Renderer& renderer);
		void draw_console(AppState& state, Renderer& renderer);
    };
}
