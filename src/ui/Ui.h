#pragma once

#include "app/AppState.h"
#include "core/Assets.h"

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
        void draw(AppState& state, const AssetIndex& assets, const UiFrameInfo& frame);

    private:
        void draw_main_menu(AppState& state, const AssetIndex& assets);
        void draw_cloth_view(AppState& state, const UiFrameInfo& frame);
        void draw_simulation(AppState& state);
        void draw_smpl(AppState& state, const AssetIndex& assets);
        void draw_garments(AppState& state, const AssetIndex& assets);
		void draw_console(AppState& state);
    };
}
