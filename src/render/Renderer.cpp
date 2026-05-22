#include "render/Renderer.h"
#include "core/Paths.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glad/glad.h>
#include <igl/frustum.h>
#include <igl/look_at.h>
#include <igl/ortho.h>
#include <igl/read_triangle_mesh.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <vector>

#include <stb_image_write.h>

namespace poseweaver
{
    namespace
    {
        constexpr float kSketch2RigVerticalFovRad = 0.7348438501358032f;
        constexpr float kSketch2RigHorizontalFovRad = kSketch2RigVerticalFovRad;
        constexpr float kSketch2RigVerticalFovDeg =
            kSketch2RigVerticalFovRad * 180.0f / 3.14159265358979323846f;

        static void update_core_matrices(igl::opengl::ViewerCore& core)
        {
            core.view = Eigen::Matrix4f::Identity();
            core.proj = Eigen::Matrix4f::Identity();
            core.norm = Eigen::Matrix4f::Identity();

            igl::look_at(core.camera_eye, core.camera_center, core.camera_up, core.view);
            core.view = core.view
                * (core.trackball_angle
                    * Eigen::Scaling(core.camera_zoom * core.camera_base_zoom)
                    * Eigen::Translation3f(core.camera_translation + core.camera_base_translation))
                      .matrix();

            core.norm = core.view.inverse().transpose();

            const float width = core.viewport(2);
            const float height = std::max(1.0f, core.viewport(3));

            constexpr float kPi = 3.14159265358979323846f;
            if (core.orthographic)
            {
                const float length = (core.camera_eye - core.camera_center).norm();
                const float h = std::tan(core.camera_view_angle / 360.0f * kPi) * length;
                igl::ortho(
                    -h * width / height,
                    h * width / height,
                    -h,
                    h,
                    core.camera_dnear,
                    core.camera_dfar,
                    core.proj);
            }
            else
            {
                const float fH = std::tan(core.camera_view_angle / 360.0f * kPi) * core.camera_dnear;
                const float fW = fH * width / height;
                igl::frustum(-fW, fW, -fH, fH, core.camera_dnear, core.camera_dfar, core.proj);
            }
        }

        static GLuint compile_shader(GLenum type, const char* src)
        {
            if (!src)
                return 0;
            const GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);

            GLint ok = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
            if (ok == GL_TRUE)
                return shader;

            GLint log_len = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
            std::string log;
            if (log_len > 1)
            {
                log.resize((size_t)log_len);
                GLsizei written = 0;
                glGetShaderInfoLog(shader, log_len, &written, log.data());
            }

            const char* stage = (type == GL_VERTEX_SHADER) ? "vertex" : (type == GL_FRAGMENT_SHADER) ? "fragment" : "shader";
            std::fprintf(
                stderr,
                "Two-light %s shader compile failed: %s\n",
                stage,
                log.empty() ? "(no log)" : log.c_str());
            glDeleteShader(shader);
            return 0;
        }

        static std::string read_text_file(const std::filesystem::path& path)
        {
            std::ifstream f(path, std::ios::binary);
            if (!f)
                return {};

            std::string s;
            f.seekg(0, std::ios::end);
            const std::streamsize len = f.tellg();
            if (len <= 0)
                return {};
            s.resize((size_t)len);
            f.seekg(0, std::ios::beg);
            f.read(s.data(), len);
            return s;
        }

        static GLuint link_program(GLuint vs, GLuint fs)
        {
            if (!vs || !fs)
                return 0;

            const GLuint prog = glCreateProgram();
            glAttachShader(prog, vs);
            glAttachShader(prog, fs);
            glLinkProgram(prog);

            GLint ok = 0;
            glGetProgramiv(prog, GL_LINK_STATUS, &ok);
            if (ok == GL_TRUE)
            {
                glDetachShader(prog, vs);
                glDetachShader(prog, fs);
                glDeleteShader(vs);
                glDeleteShader(fs);
                return prog;
            }

            GLint log_len = 0;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &log_len);
            std::string log;
            if (log_len > 1)
            {
                log.resize((size_t)log_len);
                GLsizei written = 0;
                glGetProgramInfoLog(prog, log_len, &written, log.data());
            }

            std::fprintf(stderr, "Two-light shader link failed: %s\n", log.empty() ? "(no log)" : log.c_str());
            glDeleteProgram(prog);
            glDetachShader(prog, vs);
            glDetachShader(prog, fs);
            glDeleteShader(vs);
            glDeleteShader(fs);
            return 0;
        }

        static GLuint build_two_light_program()
        {
            // Same attribute/uniform names as libigl's MeshGL so bind_mesh keeps working.
            const std::filesystem::path vs_path = assets_dir() / "shaders" / "two_light.vert.glsl";
            const std::filesystem::path fs_path = assets_dir() / "shaders" / "two_light.frag.glsl";

            const std::string vs_src = read_text_file(vs_path);
            const std::string fs_src = read_text_file(fs_path);
            if (vs_src.empty() || fs_src.empty())
            {
                std::fprintf(
                    stderr,
                    "Two-light shader load failed (vs=%s, fs=%s)\n",
                    vs_path.string().c_str(),
                    fs_path.string().c_str());
                return 0;
            }

            const GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src.c_str());
            const GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src.c_str());
            return link_program(vs, fs);
        }

        static Eigen::Vector3d wind_dir_from_angles_deg(double yaw_deg, double pitch_deg)
        {
            constexpr double kPi = 3.14159265358979323846264338327950288;
            const double yaw = yaw_deg * (kPi / 180.0);
            const double pitch = pitch_deg * (kPi / 180.0);
            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            const double cp = std::cos(pitch);
            const double sp = std::sin(pitch);
            // Yaw about +Y, pitch about +X' (after yaw). Base forward is +Z.
            return Eigen::Vector3d(sy * cp, -sp, cy * cp);
        }

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

        static void apply_target_camera(
            igl::opengl::ViewerCore& core,
            const AppState& state,
            std::uint64_t& last_request_id)
        {
            if (state.scene_view_snap_to_target_request_id == last_request_id || !state.scene_view_target_valid)
                return;

            Eigen::Vector3f eye(
                state.scene_view_target_eye[0],
                state.scene_view_target_eye[1],
                state.scene_view_target_eye[2]);
            Eigen::Vector3f dir(
                state.scene_view_target_direction[0],
                state.scene_view_target_direction[1],
                state.scene_view_target_direction[2]);
            Eigen::Vector3f up(
                state.scene_view_target_up[0],
                state.scene_view_target_up[1],
                state.scene_view_target_up[2]);

            if (!eye.allFinite() || !dir.allFinite() || !up.allFinite())
                return;
            if (dir.squaredNorm() < 1e-10f)
                dir = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
            dir.normalize();

            up -= dir * up.dot(dir);
            if (up.squaredNorm() < 1e-10f)
                up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
            up.normalize();

            core.orthographic = false;
            core.camera_eye = eye;
            core.camera_center = eye + dir;
            core.camera_up = up;
            core.camera_zoom = 1.0f;
            core.camera_base_zoom = 1.0f;
            core.camera_translation.setZero();
            core.camera_base_translation.setZero();
            core.trackball_angle = Eigen::Quaternionf::Identity();

            core.camera_view_angle = kSketch2RigVerticalFovDeg;
            last_request_id = state.scene_view_snap_to_target_request_id;
        }

        static ViewportRect sketch2rig_projection_rect(const ViewportRect& view)
        {
            if (!view.valid || view.width <= 0 || view.height <= 0)
                return view;

            const double target_aspect =
                std::tan(0.5 * (double)kSketch2RigHorizontalFovRad)
                / std::tan(0.5 * (double)kSketch2RigVerticalFovRad);
            int w = view.width;
            int h = (int)std::lround((double)w / target_aspect);
            if (h > view.height)
            {
                h = view.height;
                w = (int)std::lround((double)h * target_aspect);
            }
            w = std::clamp(w, 1, view.width);
            h = std::clamp(h, 1, view.height);

            return ViewportRect{
                view.x + (view.width - w) / 2,
                view.y + (view.height - h) / 2,
                w,
                h,
                true
            };
        }

        static std::vector<unsigned char> resize_gl_rgba_to_top_left_512(
            const std::vector<unsigned char>& src,
            int src_w,
            int src_h)
        {
            constexpr int kOut = 512;
            std::vector<unsigned char> dst((size_t)kOut * (size_t)kOut * 4u, 255u);
            if (src.empty() || src_w <= 0 || src_h <= 0)
                return dst;

            const double scale_x = (double)src_w / (double)kOut;
            const double scale_y = (double)src_h / (double)kOut;

            for (int y = 0; y < kOut; ++y)
            {
                const double sy_top = ((double)y + 0.5) * scale_y - 0.5;
                int y0_top = (int)std::floor(sy_top);
                int y1_top = y0_top + 1;
                const double ty = sy_top - (double)y0_top;
                y0_top = std::clamp(y0_top, 0, src_h - 1);
                y1_top = std::clamp(y1_top, 0, src_h - 1);

                const int y0_gl = src_h - 1 - y0_top;
                const int y1_gl = src_h - 1 - y1_top;

                for (int x = 0; x < kOut; ++x)
                {
                    const double sx = ((double)x + 0.5) * scale_x - 0.5;
                    int x0 = (int)std::floor(sx);
                    int x1 = x0 + 1;
                    const double tx = sx - (double)x0;
                    x0 = std::clamp(x0, 0, src_w - 1);
                    x1 = std::clamp(x1, 0, src_w - 1);

                    const unsigned char* p00 = src.data() + (((size_t)y0_gl * (size_t)src_w + (size_t)x0) * 4u);
                    const unsigned char* p10 = src.data() + (((size_t)y0_gl * (size_t)src_w + (size_t)x1) * 4u);
                    const unsigned char* p01 = src.data() + (((size_t)y1_gl * (size_t)src_w + (size_t)x0) * 4u);
                    const unsigned char* p11 = src.data() + (((size_t)y1_gl * (size_t)src_w + (size_t)x1) * 4u);
                    unsigned char* out = dst.data() + (((size_t)y * (size_t)kOut + (size_t)x) * 4u);

                    for (int c = 0; c < 4; ++c)
                    {
                        const double a = (double)p00[c] + ((double)p10[c] - (double)p00[c]) * tx;
                        const double b = (double)p01[c] + ((double)p11[c] - (double)p01[c]) * tx;
                        out[c] = (unsigned char)std::clamp<int>((int)std::lround(a + (b - a) * ty), 0, 255);
                    }
                }
            }

            return dst;
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
        m_wind_arrow_mesh_index = m_viewer.append_mesh(true);
        m_sketch2rig_rig_mesh_index = m_viewer.append_mesh(true);

        m_viewer.data(m_object_mesh_index).set_visible(false);
        m_viewer.data(m_garment_mesh_index).set_visible(false);
        m_viewer.data(m_smplh_mesh_index).set_visible(false);
        m_viewer.data(m_sketch2rig_rig_mesh_index).set_visible(false);

        {
            auto& w = m_viewer.data(m_wind_arrow_mesh_index);
            w.clear();
            w.show_faces = 0;
            w.show_lines = 0;
            w.show_overlay = 1;
            w.line_width = 3.0f;
            w.set_visible(true);
        }

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
            view = ViewportRect{ 0, 0, fb_width, fb_height, (fb_width > 0 && fb_height > 0) };

        ensure_sketch2rig_rig_loaded(state);

        const bool need_body_for_garment = state.show_garment_mesh && (state.selection.pose_index >= 0);
        if (state.show_smpl || need_body_for_garment)
            ensure_smplh_mesh_loaded(state);
        else
            m_viewer.data(m_smplh_mesh_index).set_visible(false);

        if (state.show_object_mesh)
            ensure_object_mesh_loaded(state);
        else
            m_viewer.data(m_object_mesh_index).set_visible(false);

        if (state.show_garment_mesh)
            ensure_garment_mesh_loaded(state);
        else
        {
            m_viewer.data(m_garment_mesh_index).set_visible(false);
            m_garment_last_show_mesh = false;
        }

        handle_garment_posing(state, false);

        call_commands_from_state(state);

        update_cloth_simulation_job(state);
        apply_cloth_simulation_frame(state);

        Eigen::Vector3d scene_center(0.0, 0.0, 0.0);
        double scene_diag = 1.0;
        {

            // Prefer garment for positioning/scale, fall back to SMPL.
            if (state.show_garment_mesh && !m_garment_mesh_name.empty())
            {
                const auto& gd = m_viewer.data(m_garment_mesh_index);
                if (gd.V.rows() > 0)
                {
                    scene_center = gd.V.colwise().mean().transpose();
                    const Eigen::RowVector3d mn = gd.V.colwise().minCoeff();
                    const Eigen::RowVector3d mx = gd.V.colwise().maxCoeff();
                    scene_diag = (mx - mn).norm();
                }
            }
            else if (m_smplh_vertices.rows() > 0)
            {
                scene_center = m_smplh_vertices.colwise().mean().transpose();
                const Eigen::RowVector3d mn = m_smplh_vertices.colwise().minCoeff();
                const Eigen::RowVector3d mx = m_smplh_vertices.colwise().maxCoeff();
                scene_diag = (mx - mn).norm();
            }

            if (!(scene_diag > 1e-9) || !std::isfinite(scene_diag))
                scene_diag = 1.0;

            if (!state.wind_arrow_pivot_initialized)
            {
                // Offset up + forward + side so the arrow starts outside the body and not centered on it.
                const Eigen::Vector3d offset(0.22 * scene_diag, 0.25 * scene_diag, 0.20 * scene_diag);
                const Eigen::Vector3d p = scene_center + offset;
                state.wind_arrow_pivot = { (float)p.x(), (float)p.y(), (float)p.z() };
                state.wind_arrow_pivot_initialized = true;
            }
            if (!state.light_position_initialized)
            {
                // Offset up + front-right so the light starts outside the body.
                const Eigen::Vector3d offset(0.35 * scene_diag, 0.45 * scene_diag, 0.55 * scene_diag);
                const Eigen::Vector3d p = scene_center + offset;
                state.light_position = { (float)p.x(), (float)p.y(), (float)p.z() };
                state.light_position_initialized = true;
            }
        }

        {
            auto& w = m_viewer.data(m_wind_arrow_mesh_index);
            double diag = 1.0;
            if (state.show_garment_mesh && !m_garment_mesh_name.empty())
            {
                const auto& gd = m_viewer.data(m_garment_mesh_index);
                if (gd.V.rows() > 0)
                {
                    const Eigen::RowVector3d mn = gd.V.colwise().minCoeff();
                    const Eigen::RowVector3d mx = gd.V.colwise().maxCoeff();
                    diag = (mx - mn).norm();
                }
            }
            else if (m_smplh_vertices.rows() > 0)
            {
                const Eigen::RowVector3d mn = m_smplh_vertices.colwise().minCoeff();
                const Eigen::RowVector3d mx = m_smplh_vertices.colwise().maxCoeff();
                diag = (mx - mn).norm();
            }
            if (!(diag > 1e-9) || !std::isfinite(diag))
                diag = 1.0;
            const double len = std::clamp(0.25 * diag, 0.05, 1.0);

            const float wind_yaw_deg = (state.controls.solver == SimulationSolver::ProjectiveDynamics)
                ? state.params.wind_yaw_deg
                : state.xpbd_params.wind_yaw_deg;
            const float wind_pitch_deg = (state.controls.solver == SimulationSolver::ProjectiveDynamics)
                ? state.params.wind_pitch_deg
                : state.xpbd_params.wind_pitch_deg;
            const float wind_force = (state.controls.solver == SimulationSolver::ProjectiveDynamics)
                ? state.params.wind_force
                : state.xpbd_params.wind_force;

            Eigen::Vector3d dir = wind_dir_from_angles_deg((double)wind_yaw_deg, (double)wind_pitch_deg);
            const double dn = dir.norm();
            if (!(dn > 1e-12) || !std::isfinite(dn))
                dir = Eigen::Vector3d(0.0, 0.0, 1.0);
            else
                dir /= dn;

            Eigen::Vector3d up(0.0, 1.0, 0.0);
            Eigen::Vector3d side = dir.cross(up);
            if (side.squaredNorm() < 1e-10)
                side = dir.cross(Eigen::Vector3d(1.0, 0.0, 0.0));
            if (side.squaredNorm() < 1e-10)
                side = Eigen::Vector3d(1.0, 0.0, 0.0);
            side.normalize();

            const Eigen::Vector3d tail(
                (double)state.wind_arrow_pivot[0],
                (double)state.wind_arrow_pivot[1],
                (double)state.wind_arrow_pivot[2]);
            const Eigen::Vector3d tip = tail + len * dir;
            const Eigen::Vector3d head_base = tip - 0.20 * len * dir;
            const Eigen::Vector3d head_l = head_base + 0.08 * len * side;
            const Eigen::Vector3d head_r = head_base - 0.08 * len * side;

            Eigen::MatrixXd P1(3, 3), P2(3, 3), C(3, 3);
            P1.row(0) = tail.transpose();
            P2.row(0) = tip.transpose();
            P1.row(1) = tip.transpose();
            P2.row(1) = head_l.transpose();
            P1.row(2) = tip.transpose();
            P2.row(2) = head_r.transpose();

            const bool active = (wind_force > 0.0f);
            const Eigen::RowVector3d col = active ? Eigen::RowVector3d(1.0, 0.9, 0.2) : Eigen::RowVector3d(0.65, 0.65, 0.65);
            C.row(0) = col;
            C.row(1) = col;
            C.row(2) = col;

            w.clear();
            w.show_faces = 0;
            w.show_lines = 0;
            w.show_overlay = 1;
            w.line_width = 3.0f;
            w.add_edges(P1, P2, C);
            w.set_visible(true);
        }

        const auto& r = view;
        auto& core = m_viewer.core();
        core.viewport = Eigen::Vector4f((float)r.x, (float)r.y, (float)r.width, (float)r.height);
        core.background_color = Eigen::Vector4f(0.45f, 0.45f, 0.47f, 1.0f);

        if (r.width <= 0 || r.height <= 0)
        {
            capture_generated_cloth_render_if_requested(state);
            glDisable(GL_SCISSOR_TEST);
            return;
        }

        apply_target_camera(core, state, m_last_scene_view_snap_to_target_request_id);
        update_core_matrices(core);

        // primary light: user-controlled; secondary: mirrored in X/Z around scene center.
        const Eigen::Vector3d light1_world_d(
            (double)state.light_position[0],
            (double)state.light_position[1],
            (double)state.light_position[2]);
        Eigen::Vector3d d = light1_world_d - scene_center;
        if (!d.allFinite() || d.squaredNorm() < 1e-12)
            d = Eigen::Vector3d(0.35 * scene_diag, 0.45 * scene_diag, 0.55 * scene_diag);
        const Eigen::Vector3d light2_world_d = scene_center + Eigen::Vector3d(-d.x(), d.y(), -d.z());

        const Eigen::Vector4f light1_world(
            (float)light1_world_d.x(),
            (float)light1_world_d.y(),
            (float)light1_world_d.z(),
            1.0f);
        const Eigen::Vector4f light2_world(
            (float)light2_world_d.x(),
            (float)light2_world_d.y(),
            (float)light2_world_d.z(),
            1.0f);
        const Eigen::Vector3f light1_eye = (core.view * light1_world).head<3>();
        const Eigen::Vector3f light2_eye = (core.view * light2_world).head<3>();

        core.light_position = light1_eye;
        core.is_directional_light = false;
        core.is_shadow_mapping = false;

        core.clear_framebuffers();

        const auto ensure_two_light_shader_installed = [&](int mesh_index)
        {
            const size_t i = (size_t)mesh_index;
            if (i >= m_two_light_shader_installed.size())
                return;
            if (m_two_light_shader_installed[i])
                return;

            auto& data = m_viewer.data(mesh_index);
            auto& meshgl = data.meshgl;
            if (!meshgl.is_initialized)
                return;

            const GLuint prog = build_two_light_program();
            if (prog == 0)
                return;

            const GLuint old = meshgl.shader_mesh;
            meshgl.shader_mesh = prog;
            if (old != 0)
                glDeleteProgram(old);

            m_two_light_shader_installed[i] = true;
        };

        const auto draw_mesh_with_two_lights = [&](int mesh_index)
        {
            auto& data = m_viewer.data(mesh_index);
            if (!(data.is_visible & core.id))
                return;
            if (data.V.rows() <= 0)
                return;

            if (core.depth_test)
                glEnable(GL_DEPTH_TEST);
            else
                glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            if (data.dirty)
            {
                data.updateGL(data, data.invert_normals, data.meshgl);
                data.dirty = igl::opengl::MeshGL::DIRTY_NONE;
            }

            ensure_two_light_shader_installed(mesh_index);

            data.meshgl.bind_mesh();
            glViewport((int)core.viewport(0), (int)core.viewport(1), (int)core.viewport(2), (int)core.viewport(3));

            const GLuint prog = data.meshgl.shader_mesh;

            const GLint viewi = glGetUniformLocation(prog, "view");
            const GLint proji = glGetUniformLocation(prog, "proj");
            const GLint normi = glGetUniformLocation(prog, "normal_matrix");
            glUniformMatrix4fv(viewi, 1, GL_FALSE, core.view.data());
            glUniformMatrix4fv(proji, 1, GL_FALSE, core.proj.data());
            glUniformMatrix4fv(normi, 1, GL_FALSE, core.norm.data());

            glUniform1f(glGetUniformLocation(prog, "specular_exponent"), data.shininess);
            glUniform3fv(glGetUniformLocation(prog, "light_position_eye"), 1, light1_eye.data());
            glUniform3fv(glGetUniformLocation(prog, "light_position_eye2"), 1, light2_eye.data());
            glUniform1f(glGetUniformLocation(prog, "light2_factor"), 0.55f);
            glUniform1f(glGetUniformLocation(prog, "lighting_factor"), core.lighting_factor);
            glUniform4f(glGetUniformLocation(prog, "fixed_color"), 0.0f, 0.0f, 0.0f, 0.0f);

            glUniform1i(glGetUniformLocation(prog, "is_directional_light"), 0);
            glUniform1i(glGetUniformLocation(prog, "is_shadow_mapping"), 0);
            glUniform1i(glGetUniformLocation(prog, "shadow_pass"), 0);

            if (core.is_set(data.show_faces))
            {
                glUniform1f(glGetUniformLocation(prog, "texture_factor"), core.is_set(data.show_texture) ? 1.0f : 0.0f);
                glUniform1f(glGetUniformLocation(prog, "matcap_factor"), core.is_set(data.use_matcap) ? 1.0f : 0.0f);
                glUniform1f(glGetUniformLocation(prog, "double_sided"), data.double_sided ? 1.0f : 0.0f);
                data.meshgl.draw_mesh(true);
                glUniform1f(glGetUniformLocation(prog, "matcap_factor"), 0.0f);
                glUniform1f(glGetUniformLocation(prog, "texture_factor"), 0.0f);
            }

            if (core.is_set(data.show_lines))
            {
                glLineWidth(data.line_width);
                glUniform4f(
                    glGetUniformLocation(prog, "fixed_color"),
                    data.line_color[0],
                    data.line_color[1],
                    data.line_color[2],
                    data.line_color[3]);
                data.meshgl.draw_mesh(false);
                glUniform4f(glGetUniformLocation(prog, "fixed_color"), 0.0f, 0.0f, 0.0f, 0.0f);
            }
        };

        const auto draw_rig_on_top = [&]()
        {
            auto& rig_data = m_viewer.data(m_sketch2rig_rig_mesh_index);
            if (!state.show_sketch2rig_rig || rig_data.V.rows() <= 0 || !(rig_data.is_visible & core.id))
                return;

            const bool old_core_depth = core.depth_test;
            const GLboolean old_depth_enabled = glIsEnabled(GL_DEPTH_TEST);
            GLboolean old_depth_mask = GL_TRUE;
            glGetBooleanv(GL_DEPTH_WRITEMASK, &old_depth_mask);

            core.depth_test = false;
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            core.draw(rig_data, false);

            core.depth_test = old_core_depth;
            glDepthMask(old_depth_mask);
            if (old_depth_enabled)
                glEnable(GL_DEPTH_TEST);
            else
                glDisable(GL_DEPTH_TEST);
        };

        const auto draw_scene_contents = [&]()
        {
            draw_mesh_with_two_lights(m_object_mesh_index);
            draw_mesh_with_two_lights(m_garment_mesh_index);
            draw_mesh_with_two_lights(m_smplh_mesh_index);

            core.draw(m_viewer.data(m_grid_mesh_index), false);
            core.draw(m_viewer.data(m_wind_arrow_mesh_index), false);
            draw_rig_on_top();
        };

        draw_scene_contents();

        const auto snapshot_ready = [&]() -> bool
        {
            const std::uint64_t request_id = state.scene_view_snapshot_request_id;
            if (request_id == 0 || request_id == m_last_scene_view_snapshot_request_id)
                return false;
            if (!state.scene_view_snapshot_pose_name.empty()
                && m_smplh_loaded_pose_name != state.scene_view_snapshot_pose_name)
                return false;
            return m_smplh_vertices.rows() > 0;
        };

        if (snapshot_ready())
        {
            const ViewportRect snapshot_view = sketch2rig_projection_rect(view);
            if (snapshot_view.valid && snapshot_view.width > 0 && snapshot_view.height > 0)
            {
                const Eigen::Vector4f old_viewport = core.viewport;
                core.viewport = Eigen::Vector4f(
                    (float)snapshot_view.x,
                    (float)snapshot_view.y,
                    (float)snapshot_view.width,
                    (float)snapshot_view.height);
                update_core_matrices(core);

                glEnable(GL_SCISSOR_TEST);
                glScissor(snapshot_view.x, snapshot_view.y, snapshot_view.width, snapshot_view.height);
                glViewport(snapshot_view.x, snapshot_view.y, snapshot_view.width, snapshot_view.height);
                glClearColor(0.45f, 0.45f, 0.47f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                draw_scene_contents();
                capture_scene_view_snapshot_if_requested(state, snapshot_view);

                core.viewport = old_viewport;
                update_core_matrices(core);
            }
        }

        capture_generated_cloth_render_if_requested(state);

        glDisable(GL_SCISSOR_TEST);
    }

    void Renderer::capture_scene_view_snapshot_if_requested(AppState& state, const ViewportRect& view)
    {
        const std::uint64_t request_id = state.scene_view_snapshot_request_id;
        if (request_id == 0 || request_id == m_last_scene_view_snapshot_request_id)
            return;
        if (!state.scene_view_snapshot_pose_name.empty()
            && m_smplh_loaded_pose_name != state.scene_view_snapshot_pose_name)
            return;
        if (m_smplh_vertices.rows() <= 0 || view.width <= 0 || view.height <= 0)
            return;

        std::vector<unsigned char> pixels((size_t)view.width * (size_t)view.height * 4u, 0u);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_BACK);
        glReadPixels(view.x, view.y, view.width, view.height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        std::vector<unsigned char> out = resize_gl_rgba_to_top_left_512(pixels, view.width, view.height);

        std::error_code ec;
        const std::filesystem::path renders_dir = std::filesystem::path("renders");
        std::filesystem::create_directories(renders_dir, ec);
        if (ec)
        {
            console_add_line(state, "Scene View snapshot failed: " + ec.message());
            return;
        }

        const std::filesystem::path out_path = renders_dir / "canva_scene_pose_check.png";
        const int ok = stbi_write_png(out_path.string().c_str(), 512, 512, 4, out.data(), 512 * 4);
        if (ok == 0)
        {
            console_add_line(state, "Scene View snapshot failed: could not write " + out_path.string());
            return;
        }

        state.scene_view_snapshot_path = out_path.string();
        state.scene_view_snapshot_completed_id = request_id;
        m_last_scene_view_snapshot_request_id = request_id;
        console_add_line(state, "Scene View snapshot saved: " + out_path.string());
    }

    void Renderer::capture_generated_cloth_render_if_requested(AppState& state)
    {
        const std::uint64_t request_id = state.cloth_render_request_id;
        if (request_id == 0 || request_id == m_last_cloth_render_request_id)
            return;
        if (m_garment_mesh_name.empty() || !state.show_garment_mesh)
        {
            m_last_cloth_render_request_id = request_id;
            state.cloth_render_completed_id = request_id;
            state.cloth_render_path.clear();
            console_add_line(state, "Generated cloth render failed: no garment is loaded/visible");
            return;
        }

        auto& garment = m_viewer.data(m_garment_mesh_index);
        if (garment.V.rows() <= 0 || garment.F.rows() <= 0)
        {
            m_last_cloth_render_request_id = request_id;
            state.cloth_render_completed_id = request_id;
            state.cloth_render_path.clear();
            console_add_line(state, "Generated cloth render failed: garment mesh is empty");
            return;
        }

        constexpr int kOut = 512;

        GLint old_framebuffer = 0;
        GLint old_viewport[4] = { 0, 0, 0, 0 };
        GLboolean old_scissor = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean old_blend = glIsEnabled(GL_BLEND);
        GLboolean old_depth = glIsEnabled(GL_DEPTH_TEST);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &old_framebuffer);
        glGetIntegerv(GL_VIEWPORT, old_viewport);

        GLuint fbo = 0;
        GLuint color = 0;
        GLuint depth = 0;
        glGenFramebuffers(1, &fbo);
        glGenTextures(1, &color);
        glGenRenderbuffers(1, &depth);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glBindTexture(GL_TEXTURE_2D, color);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kOut, kOut, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, kOut, kOut);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            console_add_line(state, "Generated cloth render failed: framebuffer incomplete");
            glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
            glDeleteRenderbuffers(1, &depth);
            glDeleteTextures(1, &color);
            glDeleteFramebuffers(1, &fbo);
            m_last_cloth_render_request_id = request_id;
            state.cloth_render_completed_id = request_id;
            state.cloth_render_path.clear();
            return;
        }

        auto& core = m_viewer.core();
        const Eigen::Vector4f old_core_viewport = core.viewport;
        if (state.scene_view_target_valid)
        {
            std::uint64_t force_camera = state.scene_view_snap_to_target_request_id - 1u;
            apply_target_camera(core, state, force_camera);
        }
        core.viewport = Eigen::Vector4f(0.0f, 0.0f, (float)kOut, (float)kOut);
        update_core_matrices(core);

        Eigen::Vector3d scene_center = garment.V.colwise().mean().transpose();
        const Eigen::RowVector3d mn = garment.V.colwise().minCoeff();
        const Eigen::RowVector3d mx = garment.V.colwise().maxCoeff();
        double scene_diag = (mx - mn).norm();
        if (!(scene_diag > 1e-9) || !std::isfinite(scene_diag))
            scene_diag = 1.0;

        const Eigen::Vector3d light1_world_d(
            (double)state.light_position[0],
            (double)state.light_position[1],
            (double)state.light_position[2]);
        Eigen::Vector3d d = light1_world_d - scene_center;
        if (!d.allFinite() || d.squaredNorm() < 1e-12)
            d = Eigen::Vector3d(0.35 * scene_diag, 0.45 * scene_diag, 0.55 * scene_diag);
        const Eigen::Vector3d light2_world_d = scene_center + Eigen::Vector3d(-d.x(), d.y(), -d.z());

        const Eigen::Vector4f light1_world(
            (float)light1_world_d.x(),
            (float)light1_world_d.y(),
            (float)light1_world_d.z(),
            1.0f);
        const Eigen::Vector4f light2_world(
            (float)light2_world_d.x(),
            (float)light2_world_d.y(),
            (float)light2_world_d.z(),
            1.0f);
        const Eigen::Vector3f light1_eye = (core.view * light1_world).head<3>();
        const Eigen::Vector3f light2_eye = (core.view * light2_world).head<3>();

        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, kOut, kOut);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (core.depth_test)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (garment.dirty)
        {
            garment.updateGL(garment, garment.invert_normals, garment.meshgl);
            garment.dirty = igl::opengl::MeshGL::DIRTY_NONE;
        }

        const size_t shader_index = (size_t)m_garment_mesh_index;
        if (shader_index < m_two_light_shader_installed.size()
            && !m_two_light_shader_installed[shader_index]
            && garment.meshgl.is_initialized)
        {
            const GLuint prog = build_two_light_program();
            if (prog != 0)
            {
                const GLuint old = garment.meshgl.shader_mesh;
                garment.meshgl.shader_mesh = prog;
                if (old != 0)
                    glDeleteProgram(old);
                m_two_light_shader_installed[shader_index] = true;
            }
        }

        garment.meshgl.bind_mesh();
        const GLuint prog = garment.meshgl.shader_mesh;

        glUniformMatrix4fv(glGetUniformLocation(prog, "view"), 1, GL_FALSE, core.view.data());
        glUniformMatrix4fv(glGetUniformLocation(prog, "proj"), 1, GL_FALSE, core.proj.data());
        glUniformMatrix4fv(glGetUniformLocation(prog, "normal_matrix"), 1, GL_FALSE, core.norm.data());
        glUniform1f(glGetUniformLocation(prog, "specular_exponent"), garment.shininess);
        glUniform3fv(glGetUniformLocation(prog, "light_position_eye"), 1, light1_eye.data());
        glUniform3fv(glGetUniformLocation(prog, "light_position_eye2"), 1, light2_eye.data());
        glUniform1f(glGetUniformLocation(prog, "light2_factor"), 0.55f);
        glUniform1f(glGetUniformLocation(prog, "lighting_factor"), core.lighting_factor);
        glUniform4f(glGetUniformLocation(prog, "fixed_color"), 0.0f, 0.0f, 0.0f, 0.0f);
        glUniform1i(glGetUniformLocation(prog, "is_directional_light"), 0);
        glUniform1i(glGetUniformLocation(prog, "is_shadow_mapping"), 0);
        glUniform1i(glGetUniformLocation(prog, "shadow_pass"), 0);
        glUniform1f(glGetUniformLocation(prog, "texture_factor"), core.is_set(garment.show_texture) ? 1.0f : 0.0f);
        glUniform1f(glGetUniformLocation(prog, "matcap_factor"), core.is_set(garment.use_matcap) ? 1.0f : 0.0f);
        glUniform1f(glGetUniformLocation(prog, "double_sided"), garment.double_sided ? 1.0f : 0.0f);
        garment.meshgl.draw_mesh(true);

        std::vector<unsigned char> pixels((size_t)kOut * (size_t)kOut * 4u, 0u);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, kOut, kOut, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        std::vector<unsigned char> out = resize_gl_rgba_to_top_left_512(pixels, kOut, kOut);

        std::error_code ec;
        const std::filesystem::path renders_dir = std::filesystem::path("renders");
        std::filesystem::create_directories(renders_dir, ec);
        const std::filesystem::path out_path = renders_dir / "canva_generated_cloth.png";
        bool wrote = false;
        if (!ec)
        {
            wrote = (stbi_write_png(out_path.string().c_str(), kOut, kOut, 4, out.data(), kOut * 4) != 0);
        }

        core.viewport = old_core_viewport;
        update_core_matrices(core);
        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
        glViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
        if (old_scissor)
            glEnable(GL_SCISSOR_TEST);
        else
            glDisable(GL_SCISSOR_TEST);
        if (old_blend)
            glEnable(GL_BLEND);
        else
            glDisable(GL_BLEND);
        if (old_depth)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);
        glDeleteRenderbuffers(1, &depth);
        glDeleteTextures(1, &color);
        glDeleteFramebuffers(1, &fbo);

        m_last_cloth_render_request_id = request_id;
        state.cloth_render_completed_id = request_id;
        if (wrote)
        {
            state.cloth_render_path = out_path.string();
            console_add_line(state, "Generated cloth render saved: " + out_path.string());
        }
        else
        {
            state.cloth_render_path.clear();
            console_add_line(state, "Generated cloth render failed: could not write " + out_path.string());
        }
    }

    Renderer::SceneCameraInfo Renderer::scene_camera_info() const
    {
        SceneCameraInfo info;
        if (!m_initialized)
            return info;

        const auto& core = m_viewer.core();
        info.view = core.view;
        info.proj = core.proj;
        info.camera_zoom = core.camera_zoom;
        info.camera_base_zoom = core.camera_base_zoom;
        info.orthographic = core.orthographic;
        info.valid = true;
        return info;
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
        m_sketch2rig_rig_source_path.clear();
        m_sketch2rig_rig_source_valid = false;

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
