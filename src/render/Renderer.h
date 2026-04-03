#pragma once

#include "app/AppState.h"

#include <igl/opengl/glfw/Viewer.h>

struct GLFWwindow;

namespace poseweaver
{
    class Renderer
    {
    public:
        void init(GLFWwindow* window);
        void render(const AppState& state, int fb_width, int fb_height);

        void shutdown();

        void on_framebuffer_resize(int fb_width, int fb_height);

        void on_cursor_pos(double x, double y);
        void on_mouse_button(int button, int action, int mods);
        void on_scroll(double xoffset, double yoffset);
        void on_key(int key, int scancode, int action, int mods);

    private:
        void ensure_initialized();
        void ensure_grid_built();
        void ensure_selected_mesh_loaded(const AppState& state);

        GLFWwindow* m_window = nullptr;
        bool m_initialized = false;

        igl::opengl::glfw::Viewer m_viewer;
        int m_grid_mesh_index = 0;
        int m_loaded_mesh_index = 1;
        std::string m_loaded_mesh_name;
        int m_last_fb_w = 0;
        int m_last_fb_h = 0;
    };
}
