#include "render/Renderer.h"
#include "core/Paths.h"
#include "simulation/ClothMeshProcessor.h"

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
#include <limits>
#include <vector>

#include <stb_image.h>
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

        static GLuint build_cloth_toon_program()
        {
            // Same attribute names as libigl's MeshGL, with extra uniforms for the ink outline pass.
            const std::filesystem::path vs_path = assets_dir() / "shaders" / "cloth_toon.vert.glsl";
            const std::filesystem::path fs_path = assets_dir() / "shaders" / "cloth_toon.frag.glsl";

            const std::string vs_src = read_text_file(vs_path);
            const std::string fs_src = read_text_file(fs_path);
            if (vs_src.empty() || fs_src.empty())
            {
                std::fprintf(
                    stderr,
                    "Cloth toon shader load failed (vs=%s, fs=%s)\n",
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

        static std::vector<unsigned char> resize_gl_rgba_to_top_left(
            const std::vector<unsigned char>& src,
            int src_w,
            int src_h,
            int out_size)
        {
            const int kOut = std::max(1, out_size);
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

        static float luminance(unsigned char r, unsigned char g, unsigned char b)
        {
            return (
                0.2126f * (float)r
                + 0.7152f * (float)g
                + 0.0722f * (float)b) / 255.0f;
        }

        static bool load_stroke_luminance_mean(
            const std::filesystem::path& path,
            float& out_mean)
        {
            out_mean = 0.0f;
            if (path.empty())
                return false;

            int width = 0;
            int height = 0;
            int channels = 0;
            stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
            if (!pixels || width <= 0 || height <= 0)
            {
                if (pixels)
                    stbi_image_free(pixels);
                return false;
            }

            const size_t count = (size_t)width * (size_t)height;
            float max_darkness = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                const unsigned char* px = pixels + i * 4u;
                const float alpha = (float)px[3] / 255.0f;
                const float lum = luminance(px[0], px[1], px[2]) * alpha + (1.0f - alpha);
                max_darkness = std::max(max_darkness, 1.0f - lum);
            }

            const float threshold = max_darkness / 3.0f;
            double sum = 0.0;
            size_t stroke_count = 0;
            for (size_t i = 0; i < count; ++i)
            {
                const unsigned char* px = pixels + i * 4u;
                const float alpha = (float)px[3] / 255.0f;
                const float lum = luminance(px[0], px[1], px[2]) * alpha + (1.0f - alpha);
                if (1.0f - lum <= threshold)
                    continue;
                sum += (double)lum;
                stroke_count++;
            }
            stbi_image_free(pixels);

            if (stroke_count == 0)
                return false;
            out_mean = std::clamp((float)(sum / (double)stroke_count), 0.0f, 1.0f);
            return true;
        }

        static bool match_render_stroke_luminance(
            std::vector<unsigned char>& rgba,
            float target_mean)
        {
            if (rgba.size() < 4u)
                return false;

            float max_darkness = 0.0f;
            for (size_t i = 0; i + 3u < rgba.size(); i += 4u)
            {
                if (rgba[i + 3u] < 128u)
                    continue;
                max_darkness = std::max(
                    max_darkness,
                    1.0f - luminance(rgba[i], rgba[i + 1u], rgba[i + 2u]));
            }
            if (max_darkness <= 1e-6f)
                return false;

            const float threshold = max_darkness / 3.0f;
            std::vector<float> stroke_darkness;
            stroke_darkness.reserve(rgba.size() / 16u);
            for (size_t i = 0; i + 3u < rgba.size(); i += 4u)
            {
                if (rgba[i + 3u] < 128u)
                    continue;
                const float darkness =
                    1.0f - luminance(rgba[i], rgba[i + 1u], rgba[i + 2u]);
                if (darkness > threshold)
                    stroke_darkness.push_back(darkness);
            }
            if (stroke_darkness.empty())
                return false;

            // Keep the sketch-matched value, but pull it halfway back toward
            // the raw ink value (black) so the rendered folds retain contrast.
            const float normalized_luminance = std::clamp(target_mean, 0.02f, 0.98f);
            const float target_luminance = 0.5f * normalized_luminance;
            const float target_darkness = 1.0f - target_luminance;
            const auto scaled_mean = [&](float scale)
            {
                double sum = 0.0;
                for (float darkness : stroke_darkness)
                    sum += (double)std::clamp(darkness * scale, 0.0f, 1.0f);
                return (float)(sum / (double)stroke_darkness.size());
            };

            float lo = 0.0f;
            float hi = 1.0f;
            while (scaled_mean(hi) < target_darkness && hi < 1024.0f)
                hi *= 2.0f;
            for (int iter = 0; iter < 32; ++iter)
            {
                const float mid = 0.5f * (lo + hi);
                if (scaled_mean(mid) < target_darkness)
                    lo = mid;
                else
                    hi = mid;
            }

            const float scale = hi;
            for (size_t i = 0; i + 3u < rgba.size(); i += 4u)
            {
                if (rgba[i + 3u] == 0u)
                    continue;
                for (size_t channel = 0; channel < 3u; ++channel)
                {
                    const float value = (float)rgba[i + channel] / 255.0f;
                    const float adjusted = 1.0f - std::clamp((1.0f - value) * scale, 0.0f, 1.0f);
                    rgba[i + channel] = (unsigned char)std::clamp<int>(
                        (int)std::lround(adjusted * 255.0f),
                        0,
                        255);
                }
            }
            return true;
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
        m_cloth_render_mesh_index = m_viewer.append_mesh(false);

        m_viewer.data(m_object_mesh_index).set_visible(false);
        m_viewer.data(m_garment_mesh_index).set_visible(false);
        m_viewer.data(m_smplh_mesh_index).set_visible(false);
        m_viewer.data(m_sketch2rig_rig_mesh_index).set_visible(false);
        m_viewer.data(m_cloth_render_mesh_index).set_visible(false);

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

    void Renderer::reset_computed_state(AppState& state)
    {
        if (m_cloth_sim_job_running && m_cloth_sim_cancel_flag)
            m_cloth_sim_cancel_flag->store(true);
        m_cloth_sim_restart_pending = false;
        ++m_pose_version;
        process_cloth_clear_simulation(state);
        state.controls.simulating = false;

        m_last_scene_view_snapshot_request_id = state.scene_view_snapshot_request_id;
        m_last_cloth_render_request_id = state.cloth_render_request_id;
        m_last_canva_displacement_request_id = state.canva_displacement_request_id;
        m_last_canva_iso_displacement_request_id = state.canva_iso_displacement_request_id;
        m_last_uv_image_request_id = state.uv_image_request_id;

        m_sketch2rig_rig_source_path.clear();
        m_sketch2rig_rig_last_error.clear();
        m_sketch2rig_rig_source_joints.resize(0, 0);
        m_sketch2rig_rig_scene_joints.resize(0, 0);
        m_sketch2rig_rig_parents.clear();
        m_sketch2rig_rig_source_valid = false;
        if (m_initialized)
        {
            auto& rig = m_viewer.data(m_sketch2rig_rig_mesh_index);
            rig.clear();
            rig.set_visible(false);

            auto& cloth_render = m_viewer.data(m_cloth_render_mesh_index);
            cloth_render.clear();
            cloth_render.set_visible(false);
        }

        if (m_sketch2uv_prediction_texture_id != 0)
        {
            glDeleteTextures(1, &m_sketch2uv_prediction_texture_id);
            m_sketch2uv_prediction_texture_id = 0;
        }
        m_sketch2uv_prediction_texture_w = 0;
        m_sketch2uv_prediction_texture_h = 0;
        m_sketch2uv_prediction_json_path.clear();
        m_sketch2uv_prediction_request_id = 0;
        m_sketch2uv_prediction_pixels.clear();
        m_sketch2uv_prediction_uv.clear();
        m_sketch2uv_prediction_mask.clear();

        if (m_sketch2iso_prediction_texture_id != 0)
        {
            glDeleteTextures(1, &m_sketch2iso_prediction_texture_id);
            m_sketch2iso_prediction_texture_id = 0;
        }
        m_sketch2iso_prediction_texture_w = 0;
        m_sketch2iso_prediction_texture_h = 0;
        m_sketch2iso_prediction_channels = 0;
        m_sketch2iso_prediction_npz_path.clear();
        m_sketch2iso_prediction_sketch_path.clear();
        m_sketch2iso_prediction_request_id = 0;
        m_sketch2iso_prediction_values.clear();
        m_sketch2iso_prediction_mask.clear();

        if (m_uv_debug_texture_id != 0)
        {
            glDeleteTextures(1, &m_uv_debug_texture_id);
            m_uv_debug_texture_id = 0;
        }
        m_uv_debug_texture_w = 0;
        m_uv_debug_texture_h = 0;
        m_smpl_uv_segments.clear();
        m_garment_uv_segments.clear();
        m_garment_uv_triangles.clear();
        m_garment_uv_overlay_valid = false;
        m_smpl_uv_vertex_colors.resize(0, 0);
        m_smpl_uv_per_vertex.resize(0, 0);
        m_smpl_uv_vertex_colors_valid = false;
        m_garment_uv_vertex_colors.resize(0, 0);
        m_garment_uv_vertex_colors_valid = false;

        release_smpl_body_preview();
        m_smplh_loaded_pose_name.clear();
        m_smplh_loaded_pose_axis_angle.resize(0);
        m_smplh_sim_preview_mesh_active = false;
        m_smplh_deform_dirty = true;
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
        const bool need_body_for_displacement =
            state.canva_displacement_request_id != m_last_canva_displacement_request_id
            || state.canva_iso_displacement_request_id != m_last_canva_iso_displacement_request_id;
        const bool need_body_for_smpl_layer =
            state.scene_view_snapshot_request_id != m_last_scene_view_snapshot_request_id;
        if (state.show_smpl || need_body_for_garment || need_body_for_displacement || need_body_for_smpl_layer)
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
        ensure_sketch2uv_prediction_texture(state);
        ensure_sketch2iso_prediction_texture(state);
        process_canva_displacement_map(state);
        process_canva_iso_displacement_map(state);

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

            const float wind_yaw_deg = state.xpbd_params.wind_yaw_deg;
            const float wind_pitch_deg = state.xpbd_params.wind_pitch_deg;
            const float wind_force = state.xpbd_params.wind_force;

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
        process_cloth_constraint_click(state);
        update_garment_constraint_visualization(state);

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

        const auto ensure_mesh_shader_installed = [&](int mesh_index, MeshShaderVariant variant)
        {
            const size_t i = (size_t)mesh_index;
            if (i >= m_mesh_shader_variant.size())
                return;
            if (m_mesh_shader_variant[i] == variant)
                return;

            auto& data = m_viewer.data(mesh_index);
            auto& meshgl = data.meshgl;
            if (!meshgl.is_initialized)
                return;

            const GLuint prog = (variant == MeshShaderVariant::ClothToon)
                ? build_cloth_toon_program()
                : build_two_light_program();
            if (prog == 0)
                return;

            const GLuint old = meshgl.shader_mesh;
            meshgl.shader_mesh = prog;
            if (old != 0)
                glDeleteProgram(old);

            m_mesh_shader_variant[i] = variant;
        };

        const auto draw_scene_mesh = [&](int mesh_index)
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

            const bool use_cloth_toon =
                state.cloth_toon_shader && mesh_index == m_garment_mesh_index;
            const MeshShaderVariant shader_variant = use_cloth_toon
                ? MeshShaderVariant::ClothToon
                : MeshShaderVariant::TwoLight;
            ensure_mesh_shader_installed(mesh_index, shader_variant);

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
            glUniform3fv(glGetUniformLocation(prog, "cloth_color"), 1, state.cloth_ink_color.data());
            glUniform1i(glGetUniformLocation(prog, "toon_outline_pass"), 0);
            glUniform1f(glGetUniformLocation(prog, "toon_outline_width"), 0.0f);

            glUniform1i(glGetUniformLocation(prog, "is_directional_light"), 0);
            glUniform1i(glGetUniformLocation(prog, "is_shadow_mapping"), 0);
            glUniform1i(glGetUniformLocation(prog, "shadow_pass"), 0);

            if (core.is_set(data.show_faces))
            {
                glUniform1f(glGetUniformLocation(prog, "texture_factor"), core.is_set(data.show_texture) ? 1.0f : 0.0f);
                glUniform1f(glGetUniformLocation(prog, "matcap_factor"), core.is_set(data.use_matcap) ? 1.0f : 0.0f);
                glUniform1f(glGetUniformLocation(prog, "double_sided"), data.double_sided ? 1.0f : 0.0f);

                if (use_cloth_toon)
                {
                    const GLboolean old_cull_enabled = glIsEnabled(GL_CULL_FACE);
                    GLint old_cull_face = GL_BACK;
                    GLboolean old_depth_mask = GL_TRUE;
                    glGetIntegerv(GL_CULL_FACE_MODE, &old_cull_face);
                    glGetBooleanv(GL_DEPTH_WRITEMASK, &old_depth_mask);

                    const float outline_width = std::clamp((float)(0.006 * scene_diag), 0.0025f, 0.0200f);
                    glEnable(GL_CULL_FACE);
                    glCullFace(GL_FRONT);
                    glDepthMask(GL_FALSE);
                    glUniform1i(glGetUniformLocation(prog, "toon_outline_pass"), 1);
                    glUniform1f(glGetUniformLocation(prog, "toon_outline_width"), outline_width);
                    data.meshgl.draw_mesh(true);

                    glUniform1i(glGetUniformLocation(prog, "toon_outline_pass"), 0);
                    glUniform1f(glGetUniformLocation(prog, "toon_outline_width"), 0.0f);
                    glDepthMask(old_depth_mask);
                    glCullFace((GLenum)old_cull_face);
                    if (!old_cull_enabled)
                        glDisable(GL_CULL_FACE);
                }

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
            draw_scene_mesh(m_object_mesh_index);
            draw_scene_mesh(m_garment_mesh_index);
            draw_scene_mesh(m_smplh_mesh_index);

            {
                auto& garment = m_viewer.data(m_garment_mesh_index);
                if (garment.points.rows() > 0)
                {
                    const auto old_faces = garment.show_faces;
                    const auto old_lines = garment.show_lines;
                    const auto old_overlay = garment.show_overlay;
                    garment.show_faces = 0;
                    garment.show_lines = 0;
                    garment.show_overlay = 1;
                    core.draw(garment, false);
                    garment.show_faces = old_faces;
                    garment.show_lines = old_lines;
                    garment.show_overlay = old_overlay;
                }
            }

            core.draw(m_viewer.data(m_grid_mesh_index), false);
            core.draw(m_viewer.data(m_wind_arrow_mesh_index), false);
            draw_rig_on_top();
        };

        draw_scene_contents();

        capture_scene_view_snapshot_if_requested(state, sketch2rig_projection_rect(view));

        capture_generated_cloth_render_if_requested(state);

        glDisable(GL_SCISSOR_TEST);
    }

    void Renderer::capture_scene_view_snapshot_if_requested(AppState& state, const ViewportRect& /*view*/)
    {
        const std::uint64_t request_id = state.scene_view_snapshot_request_id;
        if (request_id == 0 || request_id == m_last_scene_view_snapshot_request_id)
            return;
        if (!state.scene_view_snapshot_pose_name.empty()
            && m_smplh_loaded_pose_name != state.scene_view_snapshot_pose_name)
            return;
        if (m_smplh_vertices.rows() <= 0)
            return;

        auto& body = m_viewer.data(m_smplh_mesh_index);
        if (body.V.rows() <= 0 || body.F.rows() <= 0)
            return;

        constexpr int kOut = 512;

        GLint old_framebuffer = 0;
        GLint old_viewport[4] = { 0, 0, 0, 0 };
        GLint old_read_buffer = GL_BACK;
        GLboolean old_scissor = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean old_blend = glIsEnabled(GL_BLEND);
        GLboolean old_depth = glIsEnabled(GL_DEPTH_TEST);
        GLboolean old_cull = glIsEnabled(GL_CULL_FACE);
        GLint old_cull_face = GL_BACK;
        GLboolean old_depth_mask = GL_TRUE;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &old_framebuffer);
        glGetIntegerv(GL_VIEWPORT, old_viewport);
        glGetIntegerv(GL_READ_BUFFER, &old_read_buffer);
        glGetIntegerv(GL_CULL_FACE_MODE, &old_cull_face);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &old_depth_mask);

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
            console_add_line(state, "SMPL layer render failed: framebuffer incomplete");
            glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
            glReadBuffer((GLenum)old_read_buffer);
            glDeleteRenderbuffers(1, &depth);
            glDeleteTextures(1, &color);
            glDeleteFramebuffers(1, &fbo);
            m_last_scene_view_snapshot_request_id = request_id;
            state.scene_view_snapshot_completed_id = request_id;
            state.scene_view_snapshot_path.clear();
            return;
        }

        auto& core = m_viewer.core();
        const Eigen::Vector4f old_core_viewport = core.viewport;
        const Eigen::Matrix4f old_view = core.view;
        const Eigen::Matrix4f old_proj = core.proj;
        const Eigen::Matrix4f old_norm = core.norm;
        const Eigen::Vector4f old_background = core.background_color;
        const Eigen::Vector3f old_camera_eye = core.camera_eye;
        const Eigen::Vector3f old_camera_center = core.camera_center;
        const Eigen::Vector3f old_camera_up = core.camera_up;
        const float old_camera_zoom = core.camera_zoom;
        const float old_camera_base_zoom = core.camera_base_zoom;
        const Eigen::Vector3f old_camera_translation = core.camera_translation;
        const Eigen::Vector3f old_camera_base_translation = core.camera_base_translation;
        const Eigen::Quaternionf old_trackball_angle = core.trackball_angle;
        const float old_camera_view_angle = core.camera_view_angle;

        Eigen::MatrixXd snapshot_smpl_vertices = m_smplh_vertices;
        bool restore_body_vertices_after_snapshot = false;
        if (state.scene_view_snapshot_use_rigopt_camera)
        {
            MatrixX3dRM raw_vertices;
            std::string eval_err;
            if (smplh_evaluate_lbs(m_smplh_model, m_smplh_params, raw_vertices, &eval_err))
            {
                if (m_smplh_loaded_pose_mirror_x)
                    raw_vertices.col(0) *= -1.0;
                snapshot_smpl_vertices = raw_vertices;
                restore_body_vertices_after_snapshot = true;
            }
            else
            {
                console_add_line(state, "SMPL layer render warning: rig optimizer raw SMPL eval failed: " + eval_err);
            }
        }

        core.viewport = Eigen::Vector4f(0.0f, 0.0f, (float)kOut, (float)kOut);
        if (state.scene_view_snapshot_use_rigopt_camera)
        {
            const float source_size = (float)std::max(1, state.scene_view_snapshot_rigopt_camera_source_size);
            const float output_scale = (float)kOut / source_size;
            const float s = state.scene_view_snapshot_rigopt_camera_scale * output_scale;
            const float tx = state.scene_view_snapshot_rigopt_camera_translation[0] * output_scale;
            const float ty = state.scene_view_snapshot_rigopt_camera_translation[1] * output_scale;

            // Rig optimization renders with a fixed image-space orthographic camera:
            // px = x * scale + tx, py = -y * scale + ty. Do not reuse the
            // predicted scene camera here, or the debug silhouette and SMPL
            // layer can look like opposite sides of the body.
            core.view = Eigen::Matrix4f::Identity();

            float z_min = std::numeric_limits<float>::infinity();
            float z_max = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < snapshot_smpl_vertices.rows(); ++i)
            {
                const Eigen::Vector4f hp(
                    (float)snapshot_smpl_vertices(i, 0),
                    (float)snapshot_smpl_vertices(i, 1),
                    (float)snapshot_smpl_vertices(i, 2),
                    1.0f);
                const float z = (core.view * hp).z();
                if (std::isfinite(z))
                {
                    z_min = std::min(z_min, z);
                    z_max = std::max(z_max, z);
                }
            }
            if (!std::isfinite(z_min) || !std::isfinite(z_max) || std::abs(z_max - z_min) < 1e-6f)
            {
                z_min = -1.0f;
                z_max = 1.0f;
            }
            const float z_scale = 1.6f / std::max(1e-6f, z_max - z_min);
            const float z_bias = -0.8f - z_min * z_scale;

            core.proj = Eigen::Matrix4f::Zero();
            core.proj(0, 0) = 2.0f * s / (float)kOut;
            core.proj(0, 3) = 2.0f * tx / (float)kOut - 1.0f;
            core.proj(1, 1) = 2.0f * s / (float)kOut;
            core.proj(1, 3) = 1.0f - 2.0f * ty / (float)kOut;
            core.proj(2, 2) = z_scale;
            core.proj(2, 3) = z_bias;
            core.proj(3, 3) = 1.0f;
            core.norm = core.view.inverse().transpose();
        }
        else if (state.scene_view_target_valid)
        {
            std::uint64_t force_camera = state.scene_view_snap_to_target_request_id - 1u;
            apply_target_camera(core, state, force_camera);
            update_core_matrices(core);
        }
        else
        {
            update_core_matrices(core);
        }
        core.background_color = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);

        const Eigen::Vector3d scene_center = snapshot_smpl_vertices.colwise().mean().transpose();
        const Eigen::RowVector3d mn = snapshot_smpl_vertices.colwise().minCoeff();
        const Eigen::RowVector3d mx = snapshot_smpl_vertices.colwise().maxCoeff();
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
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (restore_body_vertices_after_snapshot)
        {
            body.set_vertices(snapshot_smpl_vertices);
            body.compute_normals();
        }

        if (body.dirty)
        {
            body.updateGL(body, body.invert_normals, body.meshgl);
            body.dirty = igl::opengl::MeshGL::DIRTY_NONE;
        }

        const size_t shader_index = (size_t)m_smplh_mesh_index;
        if (shader_index < m_mesh_shader_variant.size()
            && m_mesh_shader_variant[shader_index] != MeshShaderVariant::TwoLight
            && body.meshgl.is_initialized)
        {
            const GLuint prog = build_two_light_program();
            if (prog != 0)
            {
                const GLuint old = body.meshgl.shader_mesh;
                body.meshgl.shader_mesh = prog;
                if (old != 0)
                    glDeleteProgram(old);
                m_mesh_shader_variant[shader_index] = MeshShaderVariant::TwoLight;
            }
        }

        body.meshgl.bind_mesh();
        const GLuint prog = body.meshgl.shader_mesh;
        glUniformMatrix4fv(glGetUniformLocation(prog, "view"), 1, GL_FALSE, core.view.data());
        glUniformMatrix4fv(glGetUniformLocation(prog, "proj"), 1, GL_FALSE, core.proj.data());
        glUniformMatrix4fv(glGetUniformLocation(prog, "normal_matrix"), 1, GL_FALSE, core.norm.data());
        glUniform1f(glGetUniformLocation(prog, "specular_exponent"), body.shininess);
        glUniform3fv(glGetUniformLocation(prog, "light_position_eye"), 1, light1_eye.data());
        glUniform3fv(glGetUniformLocation(prog, "light_position_eye2"), 1, light2_eye.data());
        glUniform1f(glGetUniformLocation(prog, "light2_factor"), 0.55f);
        glUniform1f(glGetUniformLocation(prog, "lighting_factor"), core.lighting_factor);
        glUniform4f(glGetUniformLocation(prog, "fixed_color"), 0.0f, 0.0f, 0.0f, 0.0f);
        glUniform1i(glGetUniformLocation(prog, "is_directional_light"), 0);
        glUniform1i(glGetUniformLocation(prog, "is_shadow_mapping"), 0);
        glUniform1i(glGetUniformLocation(prog, "shadow_pass"), 0);
        glUniform1f(glGetUniformLocation(prog, "texture_factor"), core.is_set(body.show_texture) ? 1.0f : 0.0f);
        glUniform1f(glGetUniformLocation(prog, "matcap_factor"), core.is_set(body.use_matcap) ? 1.0f : 0.0f);
        glUniform1f(glGetUniformLocation(prog, "double_sided"), body.double_sided ? 1.0f : 0.0f);
        body.meshgl.draw_mesh(true);

        std::vector<unsigned char> pixels((size_t)kOut * (size_t)kOut * 4u, 0u);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, kOut, kOut, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        std::vector<unsigned char> out = resize_gl_rgba_to_top_left(pixels, kOut, kOut, kOut);

        if (restore_body_vertices_after_snapshot)
        {
            body.set_vertices(m_smplh_vertices);
            body.compute_normals();
        }

        core.viewport = old_core_viewport;
        core.view = old_view;
        core.proj = old_proj;
        core.norm = old_norm;
        core.background_color = old_background;
        core.camera_eye = old_camera_eye;
        core.camera_center = old_camera_center;
        core.camera_up = old_camera_up;
        core.camera_zoom = old_camera_zoom;
        core.camera_base_zoom = old_camera_base_zoom;
        core.camera_translation = old_camera_translation;
        core.camera_base_translation = old_camera_base_translation;
        core.trackball_angle = old_trackball_angle;
        core.camera_view_angle = old_camera_view_angle;

        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
        glReadBuffer((GLenum)old_read_buffer);
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
        if (old_cull)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);
        glCullFace((GLenum)old_cull_face);
        glDepthMask(old_depth_mask);
        glDeleteRenderbuffers(1, &depth);
        glDeleteTextures(1, &color);
        glDeleteFramebuffers(1, &fbo);

        std::error_code ec;
        const std::filesystem::path renders_dir = std::filesystem::path("renders");
        std::filesystem::create_directories(renders_dir, ec);
        if (ec)
        {
            console_add_line(state, "SMPL layer render failed: " + ec.message());
            m_last_scene_view_snapshot_request_id = request_id;
            state.scene_view_snapshot_completed_id = request_id;
            state.scene_view_snapshot_path.clear();
            return;
        }

        const std::filesystem::path out_path = renders_dir / "canva_smpl_layer.png";
        const int ok = stbi_write_png(out_path.string().c_str(), kOut, kOut, 4, out.data(), kOut * 4);
        if (ok == 0)
        {
            console_add_line(state, "SMPL layer render failed: could not write " + out_path.string());
            m_last_scene_view_snapshot_request_id = request_id;
            state.scene_view_snapshot_completed_id = request_id;
            state.scene_view_snapshot_path.clear();
            return;
        }

        state.scene_view_snapshot_path = out_path.string();
        state.scene_view_snapshot_completed_id = request_id;
        m_last_scene_view_snapshot_request_id = request_id;
        console_add_line(state, "SMPL layer render saved: " + out_path.string());
    }

    void Renderer::capture_generated_cloth_render_if_requested(AppState& state)
    {
        const std::uint64_t request_id = state.cloth_render_request_id;
        if (request_id == 0 || request_id == m_last_cloth_render_request_id)
            return;
        state.cloth_render_source_vertex_count = 0;
        state.cloth_render_vertex_count = 0;
        state.cloth_render_face_count = 0;
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

        state.cloth_render_source_vertex_count = garment.V.rows();
        igl::opengl::ViewerData* render_mesh = &garment;
        ClothRenderRefineResult refine_result;
        if (state.cloth_render_fine)
        {
            Eigen::MatrixXd refined_V;
            Eigen::MatrixXi refined_F;
            std::string refine_error;
            const double target = std::clamp((double)state.cloth_render_target_detail_size, 0.0001, 0.25);
            if (!refine_cloth_mesh_for_render(
                    garment.V,
                    garment.F,
                    target,
                    refined_V,
                    refined_F,
                    &refine_result,
                    &refine_error))
            {
                m_last_cloth_render_request_id = request_id;
                state.cloth_render_completed_id = request_id;
                state.cloth_render_path.clear();
                console_add_line(state, "Fine cloth render failed: " + refine_error);
                return;
            }

            auto& fine_mesh = m_viewer.data(m_cloth_render_mesh_index);
            fine_mesh.clear();
            fine_mesh.set_mesh(refined_V, refined_F);
            fine_mesh.show_faces = 1;
            fine_mesh.show_lines = 0;
            fine_mesh.show_overlay = 0;
            fine_mesh.double_sided = garment.double_sided;
            fine_mesh.shininess = garment.shininess;
            fine_mesh.set_colors(Eigen::RowVector3d(0.97, 0.74, 0.50));
            fine_mesh.set_visible(false);
            render_mesh = &fine_mesh;
        }

        state.cloth_render_vertex_count = render_mesh->V.rows();
        state.cloth_render_face_count = render_mesh->F.rows();

        constexpr int kOut = 1024;
        constexpr int kRender = 2048;

        GLint old_framebuffer = 0;
        GLint old_viewport[4] = { 0, 0, 0, 0 };
        GLint old_read_buffer = GL_BACK;
        GLboolean old_scissor = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean old_blend = glIsEnabled(GL_BLEND);
        GLboolean old_depth = glIsEnabled(GL_DEPTH_TEST);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &old_framebuffer);
        glGetIntegerv(GL_VIEWPORT, old_viewport);
        glGetIntegerv(GL_READ_BUFFER, &old_read_buffer);

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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kRender, kRender, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, kRender, kRender);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            console_add_line(state, "Generated cloth render failed: framebuffer incomplete");
            glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
            glReadBuffer((GLenum)old_read_buffer);
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
        if (!state.cloth_render_use_current_camera && state.scene_view_target_valid)
        {
            std::uint64_t force_camera = state.scene_view_snap_to_target_request_id - 1u;
            apply_target_camera(core, state, force_camera);
        }
        core.viewport = Eigen::Vector4f(0.0f, 0.0f, (float)kRender, (float)kRender);
        update_core_matrices(core);

        Eigen::Vector3d scene_center = render_mesh->V.colwise().mean().transpose();
        const Eigen::RowVector3d mn = render_mesh->V.colwise().minCoeff();
        const Eigen::RowVector3d mx = render_mesh->V.colwise().maxCoeff();
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
        glViewport(0, 0, kRender, kRender);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (core.depth_test)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (render_mesh->dirty)
        {
            render_mesh->updateGL(*render_mesh, render_mesh->invert_normals, render_mesh->meshgl);
            render_mesh->dirty = igl::opengl::MeshGL::DIRTY_NONE;
        }

        const int rendered_mesh_index = state.cloth_render_fine
            ? m_cloth_render_mesh_index
            : m_garment_mesh_index;
        const size_t shader_index = (size_t)rendered_mesh_index;
        if (shader_index < m_mesh_shader_variant.size()
            && m_mesh_shader_variant[shader_index] != MeshShaderVariant::ClothToon
            && render_mesh->meshgl.is_initialized)
        {
            const GLuint prog = build_cloth_toon_program();
            if (prog != 0)
            {
                const GLuint old = render_mesh->meshgl.shader_mesh;
                render_mesh->meshgl.shader_mesh = prog;
                if (old != 0)
                    glDeleteProgram(old);
                m_mesh_shader_variant[shader_index] = MeshShaderVariant::ClothToon;
            }
        }

        render_mesh->meshgl.bind_mesh();
        const GLuint prog = render_mesh->meshgl.shader_mesh;

        glUniformMatrix4fv(glGetUniformLocation(prog, "view"), 1, GL_FALSE, core.view.data());
        glUniformMatrix4fv(glGetUniformLocation(prog, "proj"), 1, GL_FALSE, core.proj.data());
        glUniformMatrix4fv(glGetUniformLocation(prog, "normal_matrix"), 1, GL_FALSE, core.norm.data());
        glUniform1f(glGetUniformLocation(prog, "specular_exponent"), render_mesh->shininess);
        glUniform3fv(glGetUniformLocation(prog, "light_position_eye"), 1, light1_eye.data());
        glUniform3fv(glGetUniformLocation(prog, "light_position_eye2"), 1, light2_eye.data());
        glUniform1f(glGetUniformLocation(prog, "light2_factor"), 0.55f);
        glUniform1f(glGetUniformLocation(prog, "lighting_factor"), core.lighting_factor);
        glUniform4f(glGetUniformLocation(prog, "fixed_color"), 0.0f, 0.0f, 0.0f, 0.0f);
        glUniform3fv(glGetUniformLocation(prog, "cloth_color"), 1, state.cloth_ink_color.data());
        glUniform1i(glGetUniformLocation(prog, "toon_outline_pass"), 0);
        glUniform1f(glGetUniformLocation(prog, "toon_outline_width"), 0.0f);
        glUniform1i(glGetUniformLocation(prog, "is_directional_light"), 0);
        glUniform1i(glGetUniformLocation(prog, "is_shadow_mapping"), 0);
        glUniform1i(glGetUniformLocation(prog, "shadow_pass"), 0);
        glUniform1f(glGetUniformLocation(prog, "texture_factor"), core.is_set(render_mesh->show_texture) ? 1.0f : 0.0f);
        glUniform1f(glGetUniformLocation(prog, "matcap_factor"), core.is_set(render_mesh->use_matcap) ? 1.0f : 0.0f);
        glUniform1f(glGetUniformLocation(prog, "double_sided"), render_mesh->double_sided ? 1.0f : 0.0f);

        const GLboolean old_cull_enabled = glIsEnabled(GL_CULL_FACE);
        GLint old_cull_face = GL_BACK;
        GLboolean old_depth_mask = GL_TRUE;
        glGetIntegerv(GL_CULL_FACE_MODE, &old_cull_face);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &old_depth_mask);

        const float outline_width = std::clamp((float)(0.006 * scene_diag), 0.0025f, 0.0200f);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        glDepthMask(GL_FALSE);
        glUniform1i(glGetUniformLocation(prog, "toon_outline_pass"), 1);
        glUniform1f(glGetUniformLocation(prog, "toon_outline_width"), outline_width);
        render_mesh->meshgl.draw_mesh(true);

        glUniform1i(glGetUniformLocation(prog, "toon_outline_pass"), 0);
        glUniform1f(glGetUniformLocation(prog, "toon_outline_width"), 0.0f);
        glDepthMask(old_depth_mask);
        glCullFace((GLenum)old_cull_face);
        if (!old_cull_enabled)
            glDisable(GL_CULL_FACE);

        render_mesh->meshgl.draw_mesh(true);

        std::vector<unsigned char> pixels((size_t)kRender * (size_t)kRender * 4u, 0u);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, kRender, kRender, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        std::vector<unsigned char> out = resize_gl_rgba_to_top_left(pixels, kRender, kRender, kOut);

        float sketch_stroke_mean = 0.0f;
        const bool matched_stroke_mean =
            load_stroke_luminance_mean(
                std::filesystem::path(state.cloth_render_stroke_reference_path),
                sketch_stroke_mean)
            && match_render_stroke_luminance(out, sketch_stroke_mean);

        std::error_code ec;
        const std::filesystem::path renders_dir = std::filesystem::path("renders");
        std::filesystem::create_directories(renders_dir, ec);
        const std::filesystem::path out_path = renders_dir /
            (state.cloth_render_fine ? "canva_fine_cloth.png" : "canva_generated_cloth.png");
        bool wrote = false;
        if (!ec)
        {
            wrote = (stbi_write_png(out_path.string().c_str(), kOut, kOut, 4, out.data(), kOut * 4) != 0);
        }

        core.viewport = old_core_viewport;
        update_core_matrices(core);
        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
        glReadBuffer((GLenum)old_read_buffer);
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
            if (state.cloth_render_fine)
            {
                console_add_line(state,
                    "Fine cloth render saved: " + out_path.string()
                    + " (#V " + std::to_string(state.cloth_render_source_vertex_count)
                    + " -> " + std::to_string(state.cloth_render_vertex_count)
                    + ", #F=" + std::to_string(state.cloth_render_face_count)
                    + ", passes=" + std::to_string(refine_result.passes) + ")");
            }
            else
            {
                console_add_line(state, "Generated cloth render saved: " + out_path.string());
            }
            if (matched_stroke_mean)
            {
                console_add_line(state,
                    "Cloth render ink blended halfway from sketch toward black: "
                    + std::to_string(sketch_stroke_mean));
            }
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

    Renderer::UvDebugTextureInfo Renderer::sketch2uv_prediction_texture_info() const
    {
        UvDebugTextureInfo info;
        info.texture_id = m_sketch2uv_prediction_texture_id;
        info.width = m_sketch2uv_prediction_texture_w;
        info.height = m_sketch2uv_prediction_texture_h;
        info.valid = (m_sketch2uv_prediction_texture_id != 0
            && m_sketch2uv_prediction_texture_w > 0
            && m_sketch2uv_prediction_texture_h > 0);
        return info;
    }

    bool Renderer::sketch2uv_prediction_uv_at(int x, int y, float& out_u, float& out_v) const
    {
        const int w = m_sketch2uv_prediction_texture_w;
        const int h = m_sketch2uv_prediction_texture_h;
        if (x < 0 || y < 0 || x >= w || y >= h || w <= 0 || h <= 0)
            return false;

        const size_t pix = (size_t)y * (size_t)w + (size_t)x;
        const size_t count = (size_t)w * (size_t)h;
        if (m_sketch2uv_prediction_uv.size() >= count * 2u
            && m_sketch2uv_prediction_mask.size() >= count)
        {
            const float mask = m_sketch2uv_prediction_mask[pix];
            const float u = m_sketch2uv_prediction_uv[pix];
            const float v = m_sketch2uv_prediction_uv[count + pix];
            if (mask > 0.5f && std::isfinite(u) && std::isfinite(v))
            {
                out_u = std::clamp(u, 0.0f, 1.0f);
                out_v = std::clamp(v, 0.0f, 1.0f);
                return true;
            }
            return false;
        }

        if (m_sketch2uv_prediction_pixels.size() < count * 4u)
            return false;

        const unsigned char* px = m_sketch2uv_prediction_pixels.data() + pix * 4u;
        if (px[0] == 0u && px[1] == 0u && px[2] == 0u)
            return false;

        out_u = (float)px[0] / 255.0f;
        out_v = (float)px[1] / 255.0f;
        return true;
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

        if (m_sketch2uv_prediction_texture_id != 0)
        {
            glDeleteTextures(1, &m_sketch2uv_prediction_texture_id);
            m_sketch2uv_prediction_texture_id = 0;
        }
        m_sketch2uv_prediction_texture_w = 0;
        m_sketch2uv_prediction_texture_h = 0;
        m_sketch2uv_prediction_json_path.clear();
        m_sketch2uv_prediction_request_id = 0;
        m_sketch2uv_prediction_pixels.clear();
        m_sketch2uv_prediction_uv.clear();
        m_sketch2uv_prediction_mask.clear();

        if (m_sketch2iso_prediction_texture_id != 0)
        {
            glDeleteTextures(1, &m_sketch2iso_prediction_texture_id);
            m_sketch2iso_prediction_texture_id = 0;
        }
        m_sketch2iso_prediction_texture_w = 0;
        m_sketch2iso_prediction_texture_h = 0;
        m_sketch2iso_prediction_channels = 0;
        m_sketch2iso_prediction_npz_path.clear();
        m_sketch2iso_prediction_sketch_path.clear();
        m_sketch2iso_prediction_request_id = 0;
        m_sketch2iso_prediction_values.clear();
        m_sketch2iso_prediction_mask.clear();
        m_iso_vertex_descriptors.clear();
        m_iso_vertex_descriptor_count = 0;
        m_iso_vertex_descriptor_dimensions = 0;
        m_iso_vertex_descriptors_load_attempted = false;
        m_iso_kd_nodes.clear();
        m_iso_kd_root = -1;

        release_smpl_body_preview();

        for (auto& data : m_viewer.data_list)
            data.meshgl.free();
        m_mesh_shader_variant.fill(MeshShaderVariant::Default);
        for (auto& core : m_viewer.core_list)
            core.shut();
        m_viewer.shutdown_plugins();
    }
}
