#include "render/Renderer.h"

#include <glad/glad.h>
#include <igl/frustum.h>
#include <igl/look_at.h>

#include <algorithm>
#include <cmath>

namespace poseweaver
{
    namespace
    {
        constexpr float kSketch2RigVerticalFovRad = 0.7348438501358032f;
        constexpr float kSketch2RigVerticalFovDeg =
            kSketch2RigVerticalFovRad * 180.0f / 3.14159265358979323846f;

        static void preview_camera_matrices(
            const AppState& state,
            float width,
            float height,
            Eigen::Matrix4f& out_view,
            Eigen::Matrix4f& out_proj)
        {
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

            if (!eye.allFinite())
                eye = Eigen::Vector3f(0.0f, 0.0f, 4.0f);
            if (!dir.allFinite() || dir.squaredNorm() < 1e-10f)
                dir = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
            dir.normalize();

            up -= dir * up.dot(dir);
            if (!up.allFinite() || up.squaredNorm() < 1e-10f)
                up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
            up.normalize();

            igl::look_at(eye, eye + dir, up, out_view);

            constexpr float kPi = 3.14159265358979323846f;
            const float dnear = 0.01f;
            const float dfar = 100.0f;
            const float aspect = (height > 1.0f) ? (width / height) : 1.0f;
            const float fov_deg =
                (state.scene_view_target_fov_degrees > 1.0f
                    && std::isfinite(state.scene_view_target_fov_degrees))
                ? state.scene_view_target_fov_degrees
                : kSketch2RigVerticalFovDeg;
            const float f_h = std::tan(fov_deg / 360.0f * kPi) * dnear;
            const float f_w = f_h * aspect;
            igl::frustum(-f_w, f_w, -f_h, f_h, dnear, dfar, out_proj);
        }

        static bool barycentric_uv(
            const Eigen::Vector2d& p,
            const Eigen::Vector2d& a,
            const Eigen::Vector2d& b,
            const Eigen::Vector2d& c,
            Eigen::Vector3d& out_bary)
        {
            const Eigen::Vector2d v0 = b - a;
            const Eigen::Vector2d v1 = c - a;
            const Eigen::Vector2d v2 = p - a;
            const double d00 = v0.dot(v0);
            const double d01 = v0.dot(v1);
            const double d11 = v1.dot(v1);
            const double d20 = v2.dot(v0);
            const double d21 = v2.dot(v1);
            const double denom = d00 * d11 - d01 * d01;
            if (!(std::abs(denom) > 1e-14) || !std::isfinite(denom))
                return false;

            const double b1 = (d11 * d20 - d01 * d21) / denom;
            const double b2 = (d00 * d21 - d01 * d20) / denom;
            const double b0 = 1.0 - b1 - b2;
            if (!std::isfinite(b0) || !std::isfinite(b1) || !std::isfinite(b2))
                return false;

            out_bary = Eigen::Vector3d(b0, b1, b2);
            return true;
        }
    }

    Renderer::SmplBodyPreviewInfo Renderer::render_smpl_body_preview(AppState& state, int width, int height)
    {
        SmplBodyPreviewInfo info;
        if (!m_initialized || width <= 1 || height <= 1 || !state.scene_view_target_valid)
            return info;

        ensure_smplh_mesh_loaded(state);
        auto& body = m_viewer.data(m_smplh_mesh_index);
        if (body.V.rows() <= 0 || body.F.rows() <= 0 || m_smplh_vertices.rows() <= 0)
            return info;

        width = std::clamp(width, 32, 2048);
        height = std::clamp(height, 32, 2048);

        if (m_smpl_body_preview_fbo == 0)
            glGenFramebuffers(1, &m_smpl_body_preview_fbo);
        if (m_smpl_body_preview_texture_id == 0)
            glGenTextures(1, &m_smpl_body_preview_texture_id);
        if (m_smpl_body_preview_depth == 0)
            glGenRenderbuffers(1, &m_smpl_body_preview_depth);

        if (width != m_smpl_body_preview_w || height != m_smpl_body_preview_h)
        {
            glBindTexture(GL_TEXTURE_2D, m_smpl_body_preview_texture_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

            glBindRenderbuffer(GL_RENDERBUFFER, m_smpl_body_preview_depth);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);

            m_smpl_body_preview_w = width;
            m_smpl_body_preview_h = height;
        }

        GLint old_framebuffer = 0;
        GLint old_viewport[4] = { 0, 0, 0, 0 };
        GLboolean old_scissor = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean old_blend = glIsEnabled(GL_BLEND);
        GLboolean old_depth = glIsEnabled(GL_DEPTH_TEST);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &old_framebuffer);
        glGetIntegerv(GL_VIEWPORT, old_viewport);

        glBindFramebuffer(GL_FRAMEBUFFER, m_smpl_body_preview_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_smpl_body_preview_texture_id, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_smpl_body_preview_depth);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)old_framebuffer);
            return info;
        }

        auto& core = m_viewer.core();
        const Eigen::Vector4f old_core_viewport = core.viewport;
        const Eigen::Matrix4f old_view = core.view;
        const Eigen::Matrix4f old_proj = core.proj;
        const Eigen::Matrix4f old_norm = core.norm;
        const Eigen::Vector4f old_background = core.background_color;

        preview_camera_matrices(state, (float)width, (float)height, m_smpl_body_preview_view, m_smpl_body_preview_proj);
        m_smpl_body_preview_camera_valid = true;

        core.viewport = Eigen::Vector4f(0.0f, 0.0f, (float)width, (float)height);
        core.view = m_smpl_body_preview_view;
        core.proj = m_smpl_body_preview_proj;
        core.norm = core.view.inverse().transpose();
        core.background_color = Eigen::Vector4f(0.45f, 0.45f, 0.47f, 1.0f);
        core.is_directional_light = false;
        core.is_shadow_mapping = false;

        const Eigen::Vector3d scene_center = m_smplh_vertices.colwise().mean().transpose();
        const Eigen::RowVector3d mn = m_smplh_vertices.colwise().minCoeff();
        const Eigen::RowVector3d mx = m_smplh_vertices.colwise().maxCoeff();
        double scene_diag = (mx - mn).norm();
        if (!(scene_diag > 1e-9) || !std::isfinite(scene_diag))
            scene_diag = 1.0;

        const Eigen::Vector3d light_world_d = scene_center + Eigen::Vector3d(0.35 * scene_diag, 0.45 * scene_diag, 0.55 * scene_diag);
        const Eigen::Vector4f light_world(
            (float)light_world_d.x(),
            (float)light_world_d.y(),
            (float)light_world_d.z(),
            1.0f);
        core.light_position = (core.view * light_world).head<3>();

        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, width, height);
        glClearColor(0.45f, 0.45f, 0.47f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (body.dirty)
        {
            body.updateGL(body, body.invert_normals, body.meshgl);
            body.dirty = igl::opengl::MeshGL::DIRTY_NONE;
        }
        const auto old_body_visible = body.is_visible;
        body.set_visible(true);
        core.draw(body, false);
        body.is_visible = old_body_visible;

        core.viewport = old_core_viewport;
        core.view = old_view;
        core.proj = old_proj;
        core.norm = old_norm;
        core.background_color = old_background;

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

        info.texture_id = m_smpl_body_preview_texture_id;
        info.width = m_smpl_body_preview_w;
        info.height = m_smpl_body_preview_h;
        info.valid = m_smpl_body_preview_texture_id != 0 && m_smpl_body_preview_w > 0 && m_smpl_body_preview_h > 0;
        return info;
    }

    bool Renderer::smpl_body_point_for_uv(float u, float v, Eigen::Vector3f& out_point) const
    {
        if (m_smplh_vertices.rows() <= 0 || m_smplh_model.F.rows() <= 0)
            return false;

        const Eigen::Vector2d p(
            (double)std::clamp(u, 0.0f, 1.0f),
            (double)std::clamp(v, 0.0f, 1.0f));

        const Eigen::MatrixXi& F = m_smplh_model.F;
        const bool have_layout =
            m_smpl_uv_layout_loaded
            && m_smpl_F_uv.rows() == F.rows()
            && m_smpl_F_uv.cols() >= 3
            && m_smpl_uv_coords.rows() > 0;
        const bool have_fallback =
            !have_layout
            && m_smpl_uv_per_vertex.rows() == m_smplh_vertices.rows()
            && m_smpl_uv_per_vertex.cols() >= 2;
        if (!have_layout && !have_fallback)
            return false;

        int best_face = -1;
        Eigen::Vector3d best_bary = Eigen::Vector3d::Zero();
        double best_margin = -1e30;
        constexpr double kInsideEps = 1e-4;

        for (int i = 0; i < F.rows(); ++i)
        {
            Eigen::Vector2d a, b, c;
            if (have_layout)
            {
                const int t0 = m_smpl_F_uv(i, 0);
                const int t1 = m_smpl_F_uv(i, 1);
                const int t2 = m_smpl_F_uv(i, 2);
                if (t0 < 0 || t1 < 0 || t2 < 0
                    || t0 >= m_smpl_uv_coords.rows()
                    || t1 >= m_smpl_uv_coords.rows()
                    || t2 >= m_smpl_uv_coords.rows())
                    continue;
                a = m_smpl_uv_coords.row(t0).head<2>().transpose();
                b = m_smpl_uv_coords.row(t1).head<2>().transpose();
                c = m_smpl_uv_coords.row(t2).head<2>().transpose();
            }
            else
            {
                const int v0 = F(i, 0);
                const int v1 = F(i, 1);
                const int v2 = F(i, 2);
                if (v0 < 0 || v1 < 0 || v2 < 0
                    || v0 >= m_smpl_uv_per_vertex.rows()
                    || v1 >= m_smpl_uv_per_vertex.rows()
                    || v2 >= m_smpl_uv_per_vertex.rows())
                    continue;
                a = m_smpl_uv_per_vertex.row(v0).head<2>().transpose();
                b = m_smpl_uv_per_vertex.row(v1).head<2>().transpose();
                c = m_smpl_uv_per_vertex.row(v2).head<2>().transpose();
            }

            Eigen::Vector3d bary;
            if (!barycentric_uv(p, a, b, c, bary))
                continue;

            const double margin = std::min(bary.x(), std::min(bary.y(), bary.z()));
            if (margin >= -kInsideEps && margin > best_margin)
            {
                best_margin = margin;
                best_face = i;
                best_bary = bary;
            }
        }

        if (best_face < 0)
            return false;

        const int v0 = F(best_face, 0);
        const int v1 = F(best_face, 1);
        const int v2 = F(best_face, 2);
        if (v0 < 0 || v1 < 0 || v2 < 0
            || v0 >= m_smplh_vertices.rows()
            || v1 >= m_smplh_vertices.rows()
            || v2 >= m_smplh_vertices.rows())
            return false;

        const Eigen::Vector3d p3 =
            best_bary.x() * m_smplh_vertices.row(v0).transpose()
            + best_bary.y() * m_smplh_vertices.row(v1).transpose()
            + best_bary.z() * m_smplh_vertices.row(v2).transpose();
        if (!p3.allFinite())
            return false;

        out_point = p3.cast<float>();
        return true;
    }

    bool Renderer::smpl_body_point_for_vertex(int vertex, Eigen::Vector3f& out_point) const
    {
        if (vertex < 0 || vertex >= m_smplh_vertices.rows())
            return false;
        const Eigen::Vector3d point = m_smplh_vertices.row(vertex).transpose();
        if (!point.allFinite())
            return false;
        out_point = point.cast<float>();
        return true;
    }

    bool Renderer::smpl_body_preview_project(const Eigen::Vector3f& point, float& out_x01, float& out_y01) const
    {
        if (!m_smpl_body_preview_camera_valid || !point.allFinite())
            return false;

        const Eigen::Vector4f hp(point.x(), point.y(), point.z(), 1.0f);
        const Eigen::Vector4f clip = (m_smpl_body_preview_proj * m_smpl_body_preview_view) * hp;
        if (!(std::abs(clip.w()) > 1e-8f) || !std::isfinite(clip.w()))
            return false;

        const Eigen::Vector3f ndc = clip.head<3>() / clip.w();
        if (!ndc.allFinite() || ndc.z() < -1.0f || ndc.z() > 1.0f)
            return false;

        out_x01 = ndc.x() * 0.5f + 0.5f;
        out_y01 = -ndc.y() * 0.5f + 0.5f;
        return std::isfinite(out_x01) && std::isfinite(out_y01);
    }

    void Renderer::release_smpl_body_preview()
    {
        if (m_smpl_body_preview_depth != 0)
        {
            glDeleteRenderbuffers(1, &m_smpl_body_preview_depth);
            m_smpl_body_preview_depth = 0;
        }
        if (m_smpl_body_preview_texture_id != 0)
        {
            glDeleteTextures(1, &m_smpl_body_preview_texture_id);
            m_smpl_body_preview_texture_id = 0;
        }
        if (m_smpl_body_preview_fbo != 0)
        {
            glDeleteFramebuffers(1, &m_smpl_body_preview_fbo);
            m_smpl_body_preview_fbo = 0;
        }
        m_smpl_body_preview_w = 0;
        m_smpl_body_preview_h = 0;
        m_smpl_body_preview_camera_valid = false;
    }
}
