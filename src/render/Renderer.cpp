#include "render/Renderer.h"
#include "core/Paths.h"

#include <Eigen/Core>
#include <glad/glad.h>
#include <igl/read_triangle_mesh.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <filesystem>
#include <cstdio>
#include <vector>

namespace poseweaver
{
    namespace
    {
        static void append_segment(
            std::vector<Eigen::RowVector3d>& p1,
            std::vector<Eigen::RowVector3d>& p2,
            std::vector<Eigen::RowVector3d>& c,
            const Eigen::RowVector3d& a,
            const Eigen::RowVector3d& b,
            const Eigen::RowVector3d& col)
        {
            p1.push_back(a);
            p2.push_back(b);
            c.push_back(col);
        }

        static void build_scene_edges(Eigen::MatrixXd& P1, Eigen::MatrixXd& P2, Eigen::MatrixXd& C)
        {
            std::vector<Eigen::RowVector3d> p1;
            std::vector<Eigen::RowVector3d> p2;
            std::vector<Eigen::RowVector3d> c;

            p1.reserve(8192);
            p2.reserve(8192);
            c.reserve(8192);

            // Axes
            append_segment(p1, p2, c, { -1, -1, -1 }, { -0.25, -1, -1 }, { 1.0, 0.2, 0.2 });
            append_segment(p1, p2, c, { -1, -1, -1 }, { -1, -0.25, -1 }, { 0.2, 1.0, 0.2 });
            append_segment(p1, p2, c, { -1, -1, -1 }, { -1, -1, -0.25 }, { 0.2, 0.4, 1.0 });

            const int count = (int)p1.size();
            P1.resize(count, 3);
            P2.resize(count, 3);
            C.resize(count, 3);
            for (int i = 0; i < count; ++i)
            {
                P1.row(i) = p1[(size_t)i];
                P2.row(i) = p2[(size_t)i];
                C.row(i) = c[(size_t)i];
            }
        }

        static int mouse_button_to_igl(int glfw_button)
        {
            if (glfw_button == GLFW_MOUSE_BUTTON_LEFT)
                return (int)igl::opengl::glfw::Viewer::MouseButton::Left;
            if (glfw_button == GLFW_MOUSE_BUTTON_RIGHT)
                return (int)igl::opengl::glfw::Viewer::MouseButton::Right;
            return (int)igl::opengl::glfw::Viewer::MouseButton::Middle;
        }
    }

    void Renderer::init(GLFWwindow* window)
    {
        m_window = window;
        ensure_initialized();
    }

    void Renderer::ensure_initialized()
    {
        if (m_initialized)
            return;
        if (m_window == nullptr)
            return;

        m_viewer.window = m_window;
        m_viewer.init();

        m_grid_mesh_index = 0;
        m_object_mesh_index = m_viewer.append_mesh(true);
        m_garment_mesh_index = m_viewer.append_mesh(true);
        m_smplh_mesh_index = m_viewer.append_mesh(true);

        // Keep meshes hidden until a selection is made / toggles are enabled.
        m_viewer.data(m_object_mesh_index).set_visible(false);
        m_viewer.data(m_garment_mesh_index).set_visible(false);
        m_viewer.data(m_smplh_mesh_index).set_visible(false);

        ensure_grid_built();

        m_initialized = true;
    }

    void Renderer::ensure_grid_built()
    {
        Eigen::MatrixXd P1, P2, C;
        build_scene_edges(P1, P2, C);

        auto& grid = m_viewer.data(m_grid_mesh_index);
        grid.clear();
        grid.show_faces = 0;
        grid.show_lines = 0;
        grid.show_overlay = 1;
        grid.line_width = 2.5f;
        grid.add_edges(P1, P2, C);
        grid.set_visible(true);
    }

    void Renderer::ensure_object_mesh_loaded(const AppState& state)
    {
        std::string desired;
        if (state.selection.mesh_index >= 0 && state.selection.mesh_index < (int)state.selection.meshes.size())
            desired = state.selection.meshes[(size_t)state.selection.mesh_index];

        if (desired == m_object_mesh_name)
        {
            m_viewer.data(m_object_mesh_index).set_visible(!desired.empty());
            return;
        }

        auto& data = m_viewer.data(m_object_mesh_index);

        if (desired.empty())
        {
            data.clear();
            data.set_visible(false);
            m_object_mesh_name.clear();
            return;
        }

        const std::filesystem::path mesh_path = assets_dir() / "mesh" / desired;

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        const bool ok = igl::read_triangle_mesh(mesh_path.string(), V, F);
        if (!ok)
        {
            data.clear();
            data.set_visible(false);
            m_object_mesh_name.clear();
            return;
        }

        data.clear();
        data.set_mesh(V, F);
        data.show_faces = 1;
        data.show_lines = 0;
        data.show_overlay = 0;
        data.set_colors(Eigen::RowVector3d(0.85, 0.86, 0.88));
        data.set_visible(true);

        m_viewer.core().align_camera_center(V, F);

        m_object_mesh_name = std::move(desired);
    }

    void Renderer::on_framebuffer_resize(int fb_width, int fb_height)
    {
        m_last_fb_w = fb_width;
        m_last_fb_h = fb_height;

        m_viewer.post_resize(fb_width, fb_height);
    }

    void Renderer::on_cursor_pos(double x, double y)
    {
        if (!m_window)
            return;

        int win_w = 0, win_h = 0;
        int fb_w = 0, fb_h = 0;
        glfwGetWindowSize(m_window, &win_w, &win_h);
        glfwGetFramebufferSize(m_window, &fb_w, &fb_h);

        const double sx = (win_w > 0) ? (double)fb_w / (double)win_w : 1.0;
        const double sy = (win_h > 0) ? (double)fb_h / (double)win_h : 1.0;

        m_viewer.mouse_move((int)(x * sx), (int)(y * sy));
    }

    void Renderer::on_mouse_button(int button, int action, int mods)
    {
        const int mb = mouse_button_to_igl(button);
        if (action == GLFW_PRESS)
            m_viewer.mouse_down((igl::opengl::glfw::Viewer::MouseButton)mb, mods);
        else if (action == GLFW_RELEASE)
            m_viewer.mouse_up((igl::opengl::glfw::Viewer::MouseButton)mb, mods);
    }

    void Renderer::on_scroll(double /*xoffset*/, double yoffset)
    {
        m_viewer.mouse_scroll((float)yoffset);
    }

    void Renderer::on_key(int key, int /*scancode*/, int action, int mods)
    {
        if (action == GLFW_PRESS || action == GLFW_REPEAT)
            m_viewer.key_down(key, mods);
        else if (action == GLFW_RELEASE)
            m_viewer.key_up(key, mods);
    }

    void Renderer::on_char(unsigned int codepoint, int mods)
    {
        m_viewer.key_pressed(codepoint, mods);
    }

    void Renderer::render(AppState& state, int fb_width, int fb_height)
    {
        ensure_initialized();

		m_smplh_vertices_recomputed_this_frame = false;

        if (fb_width != m_last_fb_w || fb_height != m_last_fb_h)
            on_framebuffer_resize(fb_width, fb_height);

        glViewport(0, 0, fb_width, fb_height);
        glDisable(GL_SCISSOR_TEST);
        glClearColor(0.45f, 0.45f, 0.47f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ViewportRect view = state.cloth_view;
        if (!view.valid)
        {
            // fall back to rendering into the full framebuffer so the scene is never blank.
            view = ViewportRect{ 0, 0, fb_width, fb_height, (fb_width > 0 && fb_height > 0) };
        }
        
        // SMPL mesh (also needed to pose the garment when a pose is selected)
        const bool need_body_for_garment = state.show_garment_mesh && (state.selection.pose_index >= 0);
        if (state.show_smpl || need_body_for_garment)
            ensure_smplh_mesh_loaded(state);
        else
            m_viewer.data(m_smplh_mesh_index).set_visible(false);

        // Object mesh (assets/mesh)
        if (state.show_object_mesh)
            ensure_object_mesh_loaded(state);
        else
            m_viewer.data(m_object_mesh_index).set_visible(false);

        // Garment mesh (assets/garments)
        if (state.show_garment_mesh)
            ensure_garment_mesh_loaded(state);
        else
            m_viewer.data(m_garment_mesh_index).set_visible(false);

        // If a pose is selected for SMPL, update the garment vertices using the precomputed mapping.
        // Version-gated: runs once per pose/garment change (not every frame).
        handle_garment_posing(state, false);

        call_commands_from_state(state);

        const auto& r = view;
        auto& core = m_viewer.core();
        core.viewport = Eigen::Vector4f((float)r.x, (float)r.y, (float)r.width, (float)r.height);
        core.background_color = Eigen::Vector4f(0.45f, 0.45f, 0.47f, 1.0f);

        // Allow UI tabs to intentionally hide the 3D view by providing a valid-but-empty viewport.
        if (r.width <= 0 || r.height <= 0)
        {
            glDisable(GL_SCISSOR_TEST);
            return;
        }

        // Clip rendering to the cloth view when available.
        if (state.cloth_view.valid)
        {
            glEnable(GL_SCISSOR_TEST);
            glScissor(r.x, r.y, r.width, r.height);
        }
        else
        {
            glDisable(GL_SCISSOR_TEST);
        }

        m_viewer.draw();

        glDisable(GL_SCISSOR_TEST);
    }

    Renderer::UvDebugTextureInfo Renderer::uv_debug_texture_info() const
    {
        UvDebugTextureInfo info;
        info.texture_id = m_uv_debug_texture_id;
        info.width = m_uv_debug_texture_w;
        info.height = m_uv_debug_texture_h;
        info.valid = (m_uv_debug_texture_id != 0 && m_uv_debug_texture_w > 0 && m_uv_debug_texture_h > 0);
        return info;
    }

    const std::vector<Renderer::UvSegment>& Renderer::smpl_uv_wireframe_segments() const
    {
        return m_smpl_uv_segments;
    }

    void Renderer::shutdown()
    {
        m_initialized = false;
        m_window = nullptr;
        m_object_mesh_name.clear();
        m_garment_mesh_name.clear();
        m_smplh_mesh_name.clear();

        if (m_uv_debug_texture_id != 0)
        {
            glDeleteTextures(1, &m_uv_debug_texture_id);
            m_uv_debug_texture_id = 0;
        }
        m_uv_debug_texture_w = 0;
        m_uv_debug_texture_h = 0;
        m_smpl_uv_segments.clear();

        for (auto& data : m_viewer.data_list)
            data.meshgl.free();
        for (auto& core : m_viewer.core_list)
            core.shut();
        m_viewer.shutdown_plugins();
    }
}
