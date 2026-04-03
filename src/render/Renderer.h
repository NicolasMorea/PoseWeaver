#pragma once

#include "app/AppState.h"

namespace poseweaver
{
    class Renderer
    {
    public:
        void render(const AppState& state, int fb_width, int fb_height);

        void shutdown();

    private:
        void render_cloth_view(const AppState& state);
        bool ensure_initialized();

        unsigned int m_program = 0;
        unsigned int m_vao = 0;
        unsigned int m_vbo = 0;
        int m_u_mvp = -1;
    };
}
