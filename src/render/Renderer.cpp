#include "render/Renderer.h"
#include "core/Paths.h"
#include "smpl/Smplh.h"

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
        m_loaded_mesh_index = m_viewer.append_mesh(true);

        // Keep the loaded mesh hidden until a selection is made.
        m_viewer.data(m_loaded_mesh_index).set_visible(false);

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

    void Renderer::ensure_selected_mesh_loaded(const AppState& state)
    {
        std::string desired;
        if (state.selection.mesh_index >= 0 && state.selection.mesh_index < (int)state.selection.meshes.size())
            desired = state.selection.meshes[(size_t)state.selection.mesh_index];

        if (desired == m_loaded_mesh_name)
            return;

        auto& data = m_viewer.data(m_loaded_mesh_index);

        if (desired.empty())
        {
            data.clear();
            data.set_visible(false);
            m_loaded_mesh_name.clear();
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
            m_loaded_mesh_name.clear();
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

        m_loaded_mesh_name = std::move(desired);
    }

    void Renderer::ensure_smplh_mesh_loaded(const AppState& state)
    {
        const SmplType gender = state.selection.gender;
        const std::string desired = (gender == SmplType::Male) ? "SMPLH:Male" : "SMPLH:Female";
        auto& data = m_viewer.data(m_loaded_mesh_index);

        // Allow toggling the SMPL character visibility without disabling the SMPL UI.
        if (!state.show_smpl_character)
        {
            data.set_visible(false);
            m_smplh_last_show_character = false;
            return;
        }

        if (!m_smplh_last_show_character && state.show_smpl_character)
        {
            // Coming back from hidden -> ensure we update the view.
            m_smplh_deform_dirty = true;
            m_smplh_last_show_character = true;
        }

        if (!m_smplh_loaded || m_smplh_loaded_gender != gender)
        {
            const std::filesystem::path model_path = assets_dir() / "smplh" / ((gender == SmplType::Male) ? "male" : "female") / "model.npz";
            std::string err;
            if (!m_smplh_model.load_from_npz(model_path, &err))
            {
                std::fprintf(stderr, "[PoseWeaver] Failed to load SMPLH model: %s\n", err.c_str());
                data.clear();
                data.set_visible(false);
                m_loaded_mesh_name.clear();
                m_smplh_loaded = false;
                return;
            }
            m_smplh_loaded = true;
            m_smplh_loaded_gender = gender;

            m_smplh_deform_dirty = true;
            m_smplh_loaded_pose_name.clear();
            m_smplh_loaded_pose_axis_angle.resize(0);

            m_smplh_params.betas = Eigen::VectorXd::Zero(m_smplh_model.beta_count());
            // Default to the model's rest pose (T-pose).
            m_smplh_params.pose_axis_angle = Eigen::VectorXd::Zero(3 * m_smplh_model.joint_count());
        }

        // Apply shape (betas) from UI state.
        {
            const int B = m_smplh_model.beta_count();
            if (m_smplh_params.betas.size() != B)
            {
                m_smplh_params.betas = Eigen::VectorXd::Zero(B);
				m_smplh_deform_dirty = true;
			}

            const int n = std::min<int>((int)state.smpl_shape.betas.size(), B);
            for (int i = 0; i < n; ++i)
            {
                const double v = (double)state.smpl_shape.betas[(size_t)i];
                if (m_smplh_params.betas[i] != v)
                {
                    m_smplh_params.betas[i] = v;
                    m_smplh_deform_dirty = true;
                }
            }
            for (int i = n; i < B; ++i)
            {
                if (m_smplh_params.betas[i] != 0.0)
                {
                    m_smplh_params.betas[i] = 0.0;
                    m_smplh_deform_dirty = true;
                }
            }
        }

        // Apply pose selection.
        {
            std::string pose_name;
            if (state.selection.pose_index >= 0 && state.selection.pose_index < (int)state.selection.poses.size())
                pose_name = state.selection.poses[(size_t)state.selection.pose_index];

            if (pose_name != m_smplh_loaded_pose_name)
            {
                m_smplh_loaded_pose_name = pose_name;
                m_smplh_loaded_pose_axis_angle.resize(0);

                m_smplh_deform_dirty = true;

                if (!pose_name.empty())
                {
                    const std::filesystem::path pose_path = assets_dir() / "poses" / pose_name;
                    std::string pose_err;
                    Eigen::VectorXd pose_axis_angle;
                    if (!smplh_load_pose_npz(pose_path, pose_axis_angle, nullptr, &pose_err))
                    {
                        std::fprintf(stderr, "[PoseWeaver] Failed to load pose '%s': %s\n", pose_name.c_str(), pose_err.c_str());
                    }
                    else
                    {
                        m_smplh_loaded_pose_axis_angle = std::move(pose_axis_angle);

                        if (m_smplh_loaded_pose_axis_angle.size() >= 3)
                        {
                            std::fprintf(
                                stderr,
                                "[PoseWeaver] Loaded pose '%s' (len=%d). root=(%.3f, %.3f, %.3f)\n",
                                pose_name.c_str(),
                                (int)m_smplh_loaded_pose_axis_angle.size(),
                                m_smplh_loaded_pose_axis_angle[0],
                                m_smplh_loaded_pose_axis_angle[1],
                                m_smplh_loaded_pose_axis_angle[2]);
                        }
                    }
                }
            }

            const int expected = 3 * m_smplh_model.joint_count();
            if (m_smplh_params.pose_axis_angle.size() != expected)
            {
                m_smplh_params.pose_axis_angle = Eigen::VectorXd::Zero(expected);
				m_smplh_deform_dirty = true;
			}

            if ((int)m_smplh_loaded_pose_axis_angle.size() == expected)
            {
                if (!m_smplh_params.pose_axis_angle.isApprox(m_smplh_loaded_pose_axis_angle, 0.0))
				{
					m_smplh_params.pose_axis_angle = m_smplh_loaded_pose_axis_angle;
					m_smplh_deform_dirty = true;
				}
            }
            else
            {
                // No pose selected or failed to load -> rest pose.
				if (!m_smplh_params.pose_axis_angle.isZero())
				{
					m_smplh_params.pose_axis_angle.setZero();
					m_smplh_deform_dirty = true;
				}
            }
        }

                bool recomputed_vertices = false;
                if (m_smplh_deform_dirty || m_smplh_vertices.rows() == 0)
        {
            MatrixX3dRM V_skinned;
            std::string eval_err;
            if (!smplh_evaluate_lbs(m_smplh_model, m_smplh_params, V_skinned, &eval_err))
            {
                std::fprintf(stderr, "[PoseWeaver] SMPLH LBS eval failed: %s\n", eval_err.c_str());
                data.clear();
                data.set_visible(false);
                m_loaded_mesh_name.clear();
                return;
            }
            m_smplh_vertices = V_skinned;
            recomputed_vertices = true;
            m_smplh_deform_dirty = false;
        }

        const Eigen::MatrixXd& V = m_smplh_vertices;
        const Eigen::MatrixXi F = m_smplh_model.F;

        // If this is a fresh SMPLH selection, set the mesh topology once and align the camera.
        // Otherwise, only update skinned vertex positions.
        if (m_loaded_mesh_name != desired)
        {
            data.clear();
            data.set_mesh(V, F);
            data.show_faces = 1;
            data.show_lines = 0;
            data.show_overlay = 0;
            data.set_colors(Eigen::RowVector3d(0.85, 0.86, 0.88));
            data.set_visible(true);

            m_viewer.core().align_camera_center(V, F);
            m_loaded_mesh_name = desired;
        }
        else
        {
            // Only push vertices when we've recomputed skinning.
            // (ViewerData copies V internally, so avoid doing this every frame.)
            if (recomputed_vertices)
            {
                data.set_vertices(V);
                data.compute_normals();
            }
            data.set_visible(true);
        }
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

    void Renderer::render(const AppState& state, int fb_width, int fb_height)
    {
        ensure_initialized();

        if (fb_width != m_last_fb_w || fb_height != m_last_fb_h)
            on_framebuffer_resize(fb_width, fb_height);

        glViewport(0, 0, fb_width, fb_height);
        glDisable(GL_SCISSOR_TEST);
        glClearColor(0.45f, 0.45f, 0.47f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // If the UI hasn't provided a valid viewport yet (or the Cloth View is hidden),
        // fall back to rendering into the full framebuffer so the scene is never blank.
        ViewportRect view = state.cloth_view;
        if (!view.valid)
            view = ViewportRect{ 0, 0, fb_width, fb_height, (fb_width > 0 && fb_height > 0) };

        if (state.show_smpl)
            ensure_smplh_mesh_loaded(state);
        else
            ensure_selected_mesh_loaded(state);

        const auto& r = view;
        auto& core = m_viewer.core();
        core.viewport = Eigen::Vector4f((float)r.x, (float)r.y, (float)r.width, (float)r.height);
        core.background_color = Eigen::Vector4f(0.45f, 0.45f, 0.47f, 1.0f);

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

    void Renderer::shutdown()
    {
        m_initialized = false;
        m_window = nullptr;
        m_loaded_mesh_name.clear();

        for (auto& data : m_viewer.data_list)
            data.meshgl.free();
        for (auto& core : m_viewer.core_list)
            core.shut();
        m_viewer.shutdown_plugins();
    }
}
