#include "ui/Ui.h"

#include "render/Renderer.h"
#include "ui/canva/CanvaPanel.h"
#include "ui/UiInternal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <imgui.h>
#include <ImGuizmo.h>

namespace poseweaver
{
    namespace
    {
        static bool project_to_screen(
            const Eigen::Matrix4f& view,
            const Eigen::Matrix4f& proj,
            const Eigen::Vector3f& p,
            const ImVec2& rect_top_left,
            const ImVec2& rect_size,
            ImVec2& out)
        {
            const Eigen::Vector4f hp(p.x(), p.y(), p.z(), 1.0f);
            const Eigen::Vector4f clip = (proj * view) * hp;
            const float w = clip.w();
            if (!(w > 1e-8f) || !std::isfinite(w))
                return false;

            const Eigen::Vector3f ndc = clip.head<3>() / w;
            if (!std::isfinite(ndc.x()) || !std::isfinite(ndc.y()) || !std::isfinite(ndc.z()))
                return false;

            // NDC [-1,1] -> screen, with Y flipped.
            out.x = rect_top_left.x + (ndc.x() * 0.5f + 0.5f) * rect_size.x;
            out.y = rect_top_left.y + (-ndc.y() * 0.5f + 0.5f) * rect_size.y;
            return true;
        }

        static Eigen::Matrix4f make_translation(const std::array<float, 3>& p)
        {
            Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
            T(0, 3) = p[0];
            T(1, 3) = p[1];
            T(2, 3) = p[2];
            return T;
        }

        static Eigen::Matrix4f make_wind_transform(const std::array<float, 3>& pivot, float yaw_deg, float pitch_deg)
        {
            constexpr float kPi = 3.14159265358979323846f;
            const float yaw = yaw_deg * (kPi / 180.0f);
            const float pitch = pitch_deg * (kPi / 180.0f);

            const float cy = std::cos(yaw);
            const float sy = std::sin(yaw);
            const float cp = std::cos(pitch);
            const float sp = std::sin(pitch);

            // R = R_y(yaw) * R_x(pitch)
            Eigen::Matrix3f Ry;
            Ry << cy, 0.0f, sy,
                0.0f, 1.0f, 0.0f,
                -sy, 0.0f, cy;

            Eigen::Matrix3f Rx;
            Rx << 1.0f, 0.0f, 0.0f,
                0.0f, cp, -sp,
                0.0f, sp, cp;

            const Eigen::Matrix3f R = (Ry * Rx).eval();

            Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
            T.block<3, 3>(0, 0) = R;
            T(0, 3) = pivot[0];
            T(1, 3) = pivot[1];
            T(2, 3) = pivot[2];
            return T;
        }

        static void yaw_pitch_from_transform(const Eigen::Matrix4f& T, float prev_yaw_deg, float& out_yaw_deg, float& out_pitch_deg)
        {
            // Extract direction = R * +Z (matches Renderer wind_dir_from_angles_deg)
            const Eigen::Vector3f dir = T.block<3, 3>(0, 0) * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
            const float dx = dir.x();
            const float dy = dir.y();
            const float dz = dir.z();

            constexpr float kPi = 3.14159265358979323846f;

            // pitch = asin(-dir.y) because dir.y = -sin(pitch)
            const float dy_clamped = std::clamp(dy, -1.0f, 1.0f);
            const float pitch = std::asin(-dy_clamped);
            const float cp = std::cos(pitch);

            float yaw = 0.0f;
            if (std::abs(cp) < 1e-4f)
            {
                // Yaw is unstable near +/-90deg pitch; keep previous.
                yaw = prev_yaw_deg * (kPi / 180.0f);
            }
            else
            {
                yaw = std::atan2(dx, dz);
            }

            out_yaw_deg = yaw * (180.0f / kPi);
            out_pitch_deg = pitch * (180.0f / kPi);
        }

        static float max_abs_diff(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b)
        {
            return (a - b).cwiseAbs().maxCoeff();
        }

        static bool scene_toolbar_toggle(const char* label, bool active)
        {
            if (active)
            {
                ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
            }

            const bool pressed = ImGui::Button(label);

            if (active)
                ImGui::PopStyleColor(2);

            return pressed;
        }

        struct UvPanePlacement
        {
            ImVec2 origin = ImVec2(0.0f, 0.0f);
            ImVec2 image_size = ImVec2(0.0f, 0.0f);
            ImVec2 clip_min = ImVec2(0.0f, 0.0f);
            ImVec2 clip_max = ImVec2(0.0f, 0.0f);
            bool valid = false;
        };

        struct UvHoverPoint
        {
            float u = 0.0f;
            float v = 0.0f;
            bool valid = false;
        };

        struct IsoHoverPoint
        {
            int vertex = -1;
            float distance = 0.0f;
            bool sampled = false;
            bool mapped = false;
        };

        static UvPanePlacement draw_uv_projection_pane(
            AppState& state,
            Renderer& renderer,
            const Renderer::UvDebugTextureInfo& info,
            const ImVec2& pane_size)
        {
            UvPanePlacement placement;
            if (pane_size.x <= 1.0f || pane_size.y <= 1.0f)
                return placement;

            const ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("uv_canvas", pane_size,
                ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);

            const bool hovered = ImGui::IsItemHovered();
            const bool active = ImGui::IsItemActive();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (!dl)
                return placement;

            const float tex_w = (float)info.width;
            const float tex_h = (float)info.height;
            const float fit_scale = (tex_w > 0.0f && tex_h > 0.0f)
                ? std::min(pane_size.x / tex_w, pane_size.y / tex_h)
                : 1.0f;

            if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                const ImVec2 d = ImGui::GetIO().MouseDelta;
                state.uv_view_pan_x += d.x;
                state.uv_view_pan_y += d.y;
            }

            if (hovered)
            {
                const float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0f)
                {
                    const float zoom_before = state.uv_view_zoom;
                    const float zoom_mul = (wheel > 0.0f) ? 1.12f : (1.0f / 1.12f);
                    state.uv_view_zoom = std::clamp(state.uv_view_zoom * zoom_mul, 0.05f, 50.0f);
                    const float zoom_after = state.uv_view_zoom;

                    const ImVec2 mouse = ImGui::GetIO().MousePos;
                    const ImVec2 canvas_center =
                        ImVec2(canvas_pos.x + pane_size.x * 0.5f, canvas_pos.y + pane_size.y * 0.5f);
                    const ImVec2 centered_origin = ImVec2(
                        canvas_center.x - 0.5f * tex_w * fit_scale * zoom_before,
                        canvas_center.y - 0.5f * tex_h * fit_scale * zoom_before);
                    const ImVec2 origin_before =
                        ImVec2(centered_origin.x + state.uv_view_pan_x, centered_origin.y + state.uv_view_pan_y);
                    const ImVec2 p_img = ImVec2(mouse.x - origin_before.x, mouse.y - origin_before.y);

                    const float s = (zoom_before > 0.0f) ? (zoom_after / zoom_before) : 1.0f;
                    const ImVec2 origin_after_centered = ImVec2(
                        canvas_center.x - 0.5f * tex_w * fit_scale * zoom_after,
                        canvas_center.y - 0.5f * tex_h * fit_scale * zoom_after);
                    const ImVec2 origin_after = ImVec2(mouse.x - p_img.x * s, mouse.y - p_img.y * s);

                    state.uv_view_pan_x = origin_after.x - origin_after_centered.x;
                    state.uv_view_pan_y = origin_after.y - origin_after_centered.y;
                }
            }

            const float zoom = state.uv_view_zoom;
            const ImVec2 canvas_center =
                ImVec2(canvas_pos.x + pane_size.x * 0.5f, canvas_pos.y + pane_size.y * 0.5f);
            const ImVec2 img_size = ImVec2(tex_w * fit_scale * zoom, tex_h * fit_scale * zoom);
            const ImVec2 origin = ImVec2(
                canvas_center.x - img_size.x * 0.5f + state.uv_view_pan_x,
                canvas_center.y - img_size.y * 0.5f + state.uv_view_pan_y);

            dl->PushClipRect(canvas_pos, ImVec2(canvas_pos.x + pane_size.x, canvas_pos.y + pane_size.y), true);

            ImTextureID tid = (ImTextureID)(intptr_t)info.texture_id;
            dl->AddImage(tid, origin, ImVec2(origin.x + img_size.x, origin.y + img_size.y));

            const auto& cloth_tris = renderer.garment_uv_triangles();
            if (!cloth_tris.empty())
            {
                constexpr size_t kMaxFillTris = 20000;
                if (cloth_tris.size() <= kMaxFillTris)
                {
                    const ImU32 cloth_fill_col = IM_COL32(0, 128, 255, 45);
                    for (const auto& t : cloth_tris)
                    {
                        const float u0 = std::clamp(t.u0, 0.0f, 1.0f);
                        const float v0 = std::clamp(t.v0, 0.0f, 1.0f);
                        const float u1 = std::clamp(t.u1, 0.0f, 1.0f);
                        const float v1 = std::clamp(t.v1, 0.0f, 1.0f);
                        const float u2 = std::clamp(t.u2, 0.0f, 1.0f);
                        const float v2 = std::clamp(t.v2, 0.0f, 1.0f);

                        const ImVec2 a =
                            ImVec2(origin.x + u0 * img_size.x, origin.y + (1.0f - v0) * img_size.y);
                        const ImVec2 b =
                            ImVec2(origin.x + u1 * img_size.x, origin.y + (1.0f - v1) * img_size.y);
                        const ImVec2 c =
                            ImVec2(origin.x + u2 * img_size.x, origin.y + (1.0f - v2) * img_size.y);
                        dl->AddTriangleFilled(a, b, c, cloth_fill_col);
                    }
                }
            }

            const auto& segs = renderer.smpl_uv_wireframe_segments();
            const ImU32 col = IM_COL32(0, 0, 0, 255);
            constexpr float kThickness = 0.75f;
            for (const auto& s : segs)
            {
                const float u0 = std::clamp(s.u0, 0.0f, 1.0f);
                const float v0 = std::clamp(s.v0, 0.0f, 1.0f);
                const float u1 = std::clamp(s.u1, 0.0f, 1.0f);
                const float v1 = std::clamp(s.v1, 0.0f, 1.0f);

                const ImVec2 a = ImVec2(origin.x + u0 * img_size.x, origin.y + (1.0f - v0) * img_size.y);
                const ImVec2 b = ImVec2(origin.x + u1 * img_size.x, origin.y + (1.0f - v1) * img_size.y);
                dl->AddLine(a, b, col, kThickness);
            }

            const auto& cloth_segs = renderer.garment_uv_wireframe_segments();
            if (!cloth_segs.empty())
            {
                const ImU32 cloth_line_col = IM_COL32(0, 90, 220, 230);
                constexpr float kClothThickness = 1.15f;
                for (const auto& s : cloth_segs)
                {
                    const float u0 = std::clamp(s.u0, 0.0f, 1.0f);
                    const float v0 = std::clamp(s.v0, 0.0f, 1.0f);
                    const float u1 = std::clamp(s.u1, 0.0f, 1.0f);
                    const float v1 = std::clamp(s.v1, 0.0f, 1.0f);

                    const ImVec2 a =
                        ImVec2(origin.x + u0 * img_size.x, origin.y + (1.0f - v0) * img_size.y);
                    const ImVec2 b =
                        ImVec2(origin.x + u1 * img_size.x, origin.y + (1.0f - v1) * img_size.y);
                    dl->AddLine(a, b, cloth_line_col, kClothThickness);
                }
            }

            dl->AddRect(origin, ImVec2(origin.x + img_size.x, origin.y + img_size.y), IM_COL32(0, 0, 0, 100));
            dl->PopClipRect();

            placement.origin = origin;
            placement.image_size = img_size;
            placement.clip_min = canvas_pos;
            placement.clip_max = ImVec2(canvas_pos.x + pane_size.x, canvas_pos.y + pane_size.y);
            placement.valid = img_size.x > 0.0f && img_size.y > 0.0f;
            return placement;
        }

        static UvPanePlacement draw_sketch2uv_prediction_pane(
            Renderer& renderer,
            const Renderer::UvDebugTextureInfo& info,
            const ImVec2& pane_size,
            UvHoverPoint& hover_point)
        {
            UvPanePlacement placement;
            if (pane_size.x <= 1.0f || pane_size.y <= 1.0f || !info.valid)
                return placement;

            const ImVec2 pane_pos = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("uv_prediction_canvas", pane_size,
                ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
            const bool hovered = ImGui::IsItemHovered();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (!dl)
                return placement;

            const float tex_w = (float)info.width;
            const float tex_h = (float)info.height;
            const float fit_scale = (tex_w > 0.0f && tex_h > 0.0f)
                ? std::min(pane_size.x / tex_w, pane_size.y / tex_h)
                : 1.0f;
            const ImVec2 img_size = ImVec2(tex_w * fit_scale, tex_h * fit_scale);
            const ImVec2 origin = ImVec2(
                pane_pos.x + (pane_size.x - img_size.x) * 0.5f,
                pane_pos.y + (pane_size.y - img_size.y) * 0.5f);

            dl->PushClipRect(pane_pos, ImVec2(pane_pos.x + pane_size.x, pane_pos.y + pane_size.y), true);
            ImTextureID tid = (ImTextureID)(intptr_t)info.texture_id;
            dl->AddImage(tid, origin, ImVec2(origin.x + img_size.x, origin.y + img_size.y));
            dl->AddRect(origin, ImVec2(origin.x + img_size.x, origin.y + img_size.y), IM_COL32(0, 0, 0, 100));

            if (hovered && img_size.x > 0.0f && img_size.y > 0.0f)
            {
                const ImVec2 mouse = ImGui::GetIO().MousePos;
                if (mouse.x >= origin.x && mouse.x < origin.x + img_size.x
                    && mouse.y >= origin.y && mouse.y < origin.y + img_size.y)
                {
                    const int px = std::clamp(
                        (int)std::floor(((mouse.x - origin.x) / img_size.x) * tex_w),
                        0,
                        std::max(0, info.width - 1));
                    const int py = std::clamp(
                        (int)std::floor(((mouse.y - origin.y) / img_size.y) * tex_h),
                        0,
                        std::max(0, info.height - 1));
                    float u = 0.0f;
                    float v = 0.0f;
                    if (renderer.sketch2uv_prediction_uv_at(px, py, u, v))
                    {
                        hover_point.u = u;
                        hover_point.v = v;
                        hover_point.valid = true;
                        const ImVec2 p = ImVec2(
                            origin.x + (((float)px + 0.5f) / tex_w) * img_size.x,
                            origin.y + (((float)py + 0.5f) / tex_h) * img_size.y);
                        dl->AddCircle(p, 5.0f, IM_COL32(255, 235, 40, 240), 16, 1.5f);
                        dl->AddCircle(p, 2.0f, IM_COL32(0, 0, 0, 220), 12, 1.0f);
                    }
                }
            }

            dl->PopClipRect();

            placement.origin = origin;
            placement.image_size = img_size;
            placement.clip_min = pane_pos;
            placement.clip_max = ImVec2(pane_pos.x + pane_size.x, pane_pos.y + pane_size.y);
            placement.valid = img_size.x > 0.0f && img_size.y > 0.0f;
            return placement;
        }

        static bool predicted_pose_camera_available(const AppState& state)
        {
            if (!state.scene_view_target_valid || state.scene_view_snapshot_pose_name.empty())
                return false;
            if (state.selection.pose_index < 0 || state.selection.pose_index >= (int)state.selection.poses.size())
                return false;
            return state.selection.poses[(size_t)state.selection.pose_index] == state.scene_view_snapshot_pose_name;
        }

        static UvPanePlacement draw_smpl_body_preview_pane(
            AppState& state,
            Renderer& renderer,
            const ImVec2& pane_size,
            bool square_camera = false)
        {
            UvPanePlacement placement;
            if (pane_size.x <= 1.0f || pane_size.y <= 1.0f)
                return placement;

            const ImVec2 pane_pos = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("smpl_body_preview_canvas", pane_size,
                ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (!dl)
                return placement;

            const float render_extent = std::min(pane_size.x, pane_size.y);
            const int render_w = std::max(
                32,
                (int)std::floor(square_camera ? render_extent : pane_size.x));
            const int render_h = std::max(
                32,
                (int)std::floor(square_camera ? render_extent : pane_size.y));
            const Renderer::SmplBodyPreviewInfo info = renderer.render_smpl_body_preview(state, render_w, render_h);
            if (!info.valid)
                return placement;

            const float tex_w = (float)info.width;
            const float tex_h = (float)info.height;
            const float fit_scale = (tex_w > 0.0f && tex_h > 0.0f)
                ? std::min(pane_size.x / tex_w, pane_size.y / tex_h)
                : 1.0f;
            const ImVec2 img_size = ImVec2(tex_w * fit_scale, tex_h * fit_scale);
            const ImVec2 origin = ImVec2(
                pane_pos.x + (pane_size.x - img_size.x) * 0.5f,
                pane_pos.y + (pane_size.y - img_size.y) * 0.5f);

            dl->PushClipRect(pane_pos, ImVec2(pane_pos.x + pane_size.x, pane_pos.y + pane_size.y), true);
            ImTextureID tid = (ImTextureID)(intptr_t)info.texture_id;
            dl->AddImage(
                tid,
                origin,
                ImVec2(origin.x + img_size.x, origin.y + img_size.y),
                ImVec2(0.0f, 1.0f),
                ImVec2(1.0f, 0.0f));
            dl->AddRect(origin, ImVec2(origin.x + img_size.x, origin.y + img_size.y), IM_COL32(0, 0, 0, 100));
            dl->PopClipRect();

            placement.origin = origin;
            placement.image_size = img_size;
            placement.clip_min = pane_pos;
            placement.clip_max = ImVec2(pane_pos.x + pane_size.x, pane_pos.y + pane_size.y);
            placement.valid = img_size.x > 0.0f && img_size.y > 0.0f;
            return placement;
        }

        static void draw_uv_hover_indicator(const UvPanePlacement& projection, const UvHoverPoint& hover_point)
        {
            if (!projection.valid || !hover_point.valid)
                return;

            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (!dl)
                return;

            const float u = std::clamp(hover_point.u, 0.0f, 1.0f);
            const float v = std::clamp(hover_point.v, 0.0f, 1.0f);
            const ImVec2 p = ImVec2(
                projection.origin.x + u * projection.image_size.x,
                projection.origin.y + (1.0f - v) * projection.image_size.y);

            dl->PushClipRect(projection.clip_min, projection.clip_max, true);
            dl->AddCircleFilled(p, 4.5f, IM_COL32(255, 235, 40, 235), 18);
            dl->AddCircle(p, 8.0f, IM_COL32(0, 0, 0, 235), 20, 1.5f);
            dl->AddLine(ImVec2(p.x - 12.0f, p.y), ImVec2(p.x + 12.0f, p.y), IM_COL32(0, 0, 0, 210), 1.0f);
            dl->AddLine(ImVec2(p.x, p.y - 12.0f), ImVec2(p.x, p.y + 12.0f), IM_COL32(0, 0, 0, 210), 1.0f);
            dl->PopClipRect();
        }

        static void draw_smpl_body_hover_indicator(
            Renderer& renderer,
            const UvPanePlacement& preview,
            const UvHoverPoint& hover_point)
        {
            if (!preview.valid || !hover_point.valid)
                return;

            Eigen::Vector3f body_point;
            if (!renderer.smpl_body_point_for_uv(hover_point.u, hover_point.v, body_point))
                return;

            float x01 = 0.0f;
            float y01 = 0.0f;
            if (!renderer.smpl_body_preview_project(body_point, x01, y01))
                return;

            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (!dl)
                return;

            const ImVec2 p = ImVec2(
                preview.origin.x + x01 * preview.image_size.x,
                preview.origin.y + y01 * preview.image_size.y);

            dl->PushClipRect(preview.clip_min, preview.clip_max, true);
            dl->AddCircleFilled(p, 5.0f, IM_COL32(255, 235, 40, 235), 18);
            dl->AddCircle(p, 9.0f, IM_COL32(0, 0, 0, 235), 20, 1.6f);
            dl->AddCircle(p, 14.0f, IM_COL32(255, 235, 40, 120), 24, 1.2f);
            dl->PopClipRect();
        }

        static UvPanePlacement draw_sketch2iso_prediction_pane(
            Renderer& renderer,
            const Renderer::UvDebugTextureInfo& info,
            float distance_threshold,
            const ImVec2& pane_size,
            IsoHoverPoint& hover_point)
        {
            UvPanePlacement placement;
            if (pane_size.x <= 1.0f || pane_size.y <= 1.0f || !info.valid)
                return placement;

            const ImVec2 pane_pos = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("iso_prediction_canvas", pane_size,
                ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
            const bool hovered = ImGui::IsItemHovered();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (!dl)
                return placement;

            const float tex_w = (float)info.width;
            const float tex_h = (float)info.height;
            const float fit_scale = (tex_w > 0.0f && tex_h > 0.0f)
                ? std::min(pane_size.x / tex_w, pane_size.y / tex_h)
                : 1.0f;
            const ImVec2 image_size(tex_w * fit_scale, tex_h * fit_scale);
            const ImVec2 origin(
                pane_pos.x + (pane_size.x - image_size.x) * 0.5f,
                pane_pos.y + (pane_size.y - image_size.y) * 0.5f);

            dl->PushClipRect(
                pane_pos,
                ImVec2(pane_pos.x + pane_size.x, pane_pos.y + pane_size.y),
                true);
            dl->AddImage(
                (ImTextureID)(intptr_t)info.texture_id,
                origin,
                ImVec2(origin.x + image_size.x, origin.y + image_size.y));
            dl->AddRect(
                origin,
                ImVec2(origin.x + image_size.x, origin.y + image_size.y),
                IM_COL32(0, 0, 0, 100));

            if (hovered && image_size.x > 0.0f && image_size.y > 0.0f)
            {
                const ImVec2 mouse = ImGui::GetIO().MousePos;
                if (mouse.x >= origin.x && mouse.x < origin.x + image_size.x
                    && mouse.y >= origin.y && mouse.y < origin.y + image_size.y)
                {
                    const int px = std::clamp(
                        (int)std::floor(((mouse.x - origin.x) / image_size.x) * tex_w),
                        0,
                        std::max(0, info.width - 1));
                    const int py = std::clamp(
                        (int)std::floor(((mouse.y - origin.y) / image_size.y) * tex_h),
                        0,
                        std::max(0, info.height - 1));
                    int vertex = -1;
                    float distance = 0.0f;
                    const bool mapped = renderer.sketch2iso_prediction_match_at(
                        px,
                        py,
                        distance_threshold,
                        vertex,
                        distance);
                    if (vertex >= 0)
                    {
                        hover_point.vertex = vertex;
                        hover_point.distance = distance;
                        hover_point.sampled = true;
                        hover_point.mapped = mapped;
                        const ImVec2 point(
                            origin.x + (((float)px + 0.5f) / tex_w) * image_size.x,
                            origin.y + (((float)py + 0.5f) / tex_h) * image_size.y);
                        const ImU32 color = mapped
                            ? IM_COL32(255, 235, 40, 240)
                            : IM_COL32(30, 110, 255, 245);
                        dl->AddCircle(point, 6.0f, color, 18, 2.0f);
                        dl->AddCircle(point, 2.0f, IM_COL32(0, 0, 0, 220), 12, 1.0f);
                    }
                }
            }
            dl->PopClipRect();

            placement.origin = origin;
            placement.image_size = image_size;
            placement.clip_min = pane_pos;
            placement.clip_max = ImVec2(pane_pos.x + pane_size.x, pane_pos.y + pane_size.y);
            placement.valid = image_size.x > 0.0f && image_size.y > 0.0f;
            return placement;
        }

        static void draw_smpl_body_iso_hover_indicator(
            Renderer& renderer,
            const UvPanePlacement& preview,
            const IsoHoverPoint& hover_point)
        {
            if (!preview.valid || !hover_point.mapped || hover_point.vertex < 0)
                return;

            Eigen::Vector3f body_point;
            if (!renderer.smpl_body_point_for_vertex(hover_point.vertex, body_point))
                return;

            float x01 = 0.0f;
            float y01 = 0.0f;
            if (!renderer.smpl_body_preview_project(body_point, x01, y01))
                return;

            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (!dl)
                return;

            const ImVec2 point(
                preview.origin.x + x01 * preview.image_size.x,
                preview.origin.y + y01 * preview.image_size.y);
            dl->PushClipRect(preview.clip_min, preview.clip_max, true);
            dl->AddCircleFilled(point, 5.0f, IM_COL32(255, 235, 40, 235), 18);
            dl->AddCircle(point, 9.0f, IM_COL32(0, 0, 0, 235), 20, 1.6f);
            dl->AddCircle(point, 14.0f, IM_COL32(255, 235, 40, 120), 24, 1.2f);
            dl->PopClipRect();
        }

    }

    void Ui::draw_cloth_view(AppState& state, Renderer& renderer, const UiFrameInfo& frame)
    {
        ImGui::Begin("Scene View", &state.show_cloth_view,
            ui_internal::fixed_panel_flags() | ImGuiWindowFlags_NoBackground);

        state.canva_tab_active = false;

        const ImVec2 avail_outer = ImGui::GetContentRegionAvail();
        if (avail_outer.x <= 1.0f || avail_outer.y <= 1.0f)
        {
            state.cloth_view.valid = false;
            ImGui::End();
            return;
        }

        if (!ImGui::BeginTabBar("##center_tabs"))
        {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabItem("Scene View"))
        {
            ImGui::PushID("scene_toolbar");
            if (scene_toolbar_toggle("SMPL (X)", state.show_smpl_character))
                state.show_smpl_character = !state.show_smpl_character;
            ImGui::SameLine();
            if (scene_toolbar_toggle("Rig (W)", state.show_sketch2rig_rig))
                state.show_sketch2rig_rig = !state.show_sketch2rig_rig;
            ImGui::SameLine();
            if (scene_toolbar_toggle("Ink Cloth", state.cloth_toon_shader))
                state.cloth_toon_shader = !state.cloth_toon_shader;
            ImGui::SameLine();
            if (ImGui::Button("Snap to Target (A)"))
                state.scene_view_snap_to_target_request_id++;
            ImGui::PopID();
            ImGui::Separator();

            const ImVec2 top_left = ImGui::GetCursorScreenPos();
            const ImVec2 avail = ImGui::GetContentRegionAvail();

            // ImGuizmo v1.61 refuses to activate if any ImGui item is hovered/active — do not place one spanning the canvas.
            const bool was_using_gizmo = state.scene_view_gizmo_using;

            state.scene_view_gizmo_over = false;
            state.scene_view_gizmo_using = false;

            const ImGuiIO& io = ImGui::GetIO();
            const ImVec2 scale = io.DisplayFramebufferScale;

            const ImGuiViewport* vp = ImGui::GetMainViewport();
            const ImVec2 vp_pos = vp ? vp->Pos : ImVec2(0.0f, 0.0f);
            const ImVec2 local = ImVec2(top_left.x - vp_pos.x, top_left.y - vp_pos.y);

            const int x = static_cast<int>(local.x * scale.x);
            const int w = static_cast<int>(avail.x * scale.x);

            const int y_top = static_cast<int>(local.y * scale.y);
            const int h = static_cast<int>(avail.y * scale.y);
            const int y = frame.fb_height - (y_top + h);

            state.cloth_view = ViewportRect{ x, y, w, h, (w > 0 && h > 0) };

            if (avail.x > 1.0f && avail.y > 1.0f)
            {
                const auto cam = renderer.scene_camera_info();
                if (cam.valid)
                {
                    ImGuizmo::BeginFrame();
                    ImGuizmo::SetDrawlist();
                    ImGuizmo::SetOrthographic(cam.orthographic);
                    ImGuizmo::SetRect(top_left.x, top_left.y, avail.x, avail.y);

                    // Match libigl's ImGuizmoWidget scaling workaround.
                    Eigen::Matrix4f view_m = (cam.view / std::max(1e-8f, cam.camera_zoom));
                    Eigen::Matrix4f proj_m = cam.proj;
                    if (cam.orthographic)
                        view_m(2, 3) -= 1000.0f; // hack depth, as in libigl

                    const float z = cam.camera_base_zoom;
                    const Eigen::Matrix4f S = (Eigen::Matrix4f() <<
                        z, 0, 0, 0,
                        0, z, 0, 0,
                        0, 0, z, 0,
                        0, 0, 0, 1).finished();
                    view_m = (view_m * S.inverse()).eval();

                    // ImGuizmo v1.61 has a single global context — draw/manipulate only one gizmo per frame.
                    const ImVec2 mouse = ImGui::GetIO().MousePos;
                    ImVec2 light_screen(0.0f, 0.0f);
                    ImVec2 wind_screen(0.0f, 0.0f);
                    const bool light_ok = project_to_screen(
                        cam.view, cam.proj,
                        Eigen::Vector3f(state.light_position[0], state.light_position[1], state.light_position[2]),
                        top_left, avail, light_screen);
                    const bool wind_ok = project_to_screen(
                        cam.view, cam.proj,
                        Eigen::Vector3f(state.wind_arrow_pivot[0], state.wind_arrow_pivot[1], state.wind_arrow_pivot[2]),
                        top_left, avail, wind_screen);

                    // Cache for GLFW callbacks (mouse routing).
                    state.scene_view_light_marker_valid = light_ok;
                    state.scene_view_wind_marker_valid = wind_ok;
                    if (light_ok)
                        state.scene_view_light_marker_screen = { light_screen.x, light_screen.y };
                    if (wind_ok)
                        state.scene_view_wind_marker_screen = { wind_screen.x, wind_screen.y };

                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    if (dl)
                    {
                        if (wind_ok)
                            dl->AddCircleFilled(wind_screen, 4.0f, IM_COL32(220, 220, 220, 200));
                        if (light_ok)
                            dl->AddCircleFilled(light_screen, 4.0f, IM_COL32(255, 255, 180, 220));
                        if (state.cloth_constraints_edit_enabled && state.cloth_constraint_drag_active)
                        {
                            ImVec2 a(
                                vp_pos.x + state.cloth_constraint_drag_start_screen[0],
                                vp_pos.y + state.cloth_constraint_drag_start_screen[1]);
                            ImVec2 b(
                                vp_pos.x + state.cloth_constraint_drag_end_screen[0],
                                vp_pos.y + state.cloth_constraint_drag_end_screen[1]);
                            a.x = std::clamp(a.x, top_left.x, top_left.x + avail.x);
                            b.x = std::clamp(b.x, top_left.x, top_left.x + avail.x);
                            a.y = std::clamp(a.y, top_left.y, top_left.y + avail.y);
                            b.y = std::clamp(b.y, top_left.y, top_left.y + avail.y);

                            const ImVec2 mn(std::min(a.x, b.x), std::min(a.y, b.y));
                            const ImVec2 mx(std::max(a.x, b.x), std::max(a.y, b.y));
                            const ImU32 fill = (state.cloth_constraint_edit_mode == 2)
                                ? IM_COL32(255, 105, 80, 38)
                                : IM_COL32(70, 230, 255, 42);
                            const ImU32 outline = (state.cloth_constraint_edit_mode == 2)
                                ? IM_COL32(255, 120, 90, 230)
                                : IM_COL32(80, 235, 255, 230);
                            dl->AddRectFilled(mn, mx, fill);
                            dl->AddRect(mn, mx, outline, 0.0f, 0, 1.5f);
                        }
                    }

                    const float dxl = mouse.x - light_screen.x;
                    const float dyl = mouse.y - light_screen.y;
                    const float dxw = mouse.x - wind_screen.x;
                    const float dyw = mouse.y - wind_screen.y;
                    const float dist2_light = (light_ok) ? (dxl * dxl + dyl * dyl) : 1e30f;
                    const float dist2_wind = (wind_ok) ? (dxw * dxw + dyw * dyw) : 1e30f;

                    constexpr float kSwitchRadiusPx = 80.0f;
                    const float switch_r2 = kSwitchRadiusPx * kSwitchRadiusPx;

                    const bool near_light = (dist2_light < switch_r2);
                    const bool near_wind = (dist2_wind < switch_r2);

                    const bool should_show_any = was_using_gizmo || near_light || near_wind;

                    if (!was_using_gizmo)
                    {
                        if (!should_show_any)
                        {
                            state.scene_view_active_gizmo = -1;
                        }
                        else
                        {
                            if (near_light && (!near_wind || dist2_light <= dist2_wind))
                                state.scene_view_active_gizmo = 1;
                            else
                                state.scene_view_active_gizmo = 0;
                        }
                    }

                    if (state.scene_view_active_gizmo == 1)
                    {
                        Eigen::Matrix4f T = make_translation(state.light_position);
                        const Eigen::Matrix4f T0 = T;
                        Eigen::Matrix4f Tg = (S * T).eval();
                        ImGuizmo::Manipulate(
                            view_m.data(), proj_m.data(),
                            ImGuizmo::TRANSLATE, ImGuizmo::WORLD,
                            Tg.data(), nullptr, nullptr);
                        T = (S.inverse() * Tg).eval();

                        if (max_abs_diff(T, T0) > 1e-7f)
                        {
                            state.light_position[0] = T(0, 3);
                            state.light_position[1] = T(1, 3);
                            state.light_position[2] = T(2, 3);
                            state.light_position_initialized = true;
                        }
                    }
                    else if (state.scene_view_active_gizmo == 0)
                    {
                        float* wind_yaw_deg = &state.xpbd_params.wind_yaw_deg;
                        float* wind_pitch_deg = &state.xpbd_params.wind_pitch_deg;

                        Eigen::Matrix4f T = make_wind_transform(
                            state.wind_arrow_pivot,
                            *wind_yaw_deg,
                            *wind_pitch_deg);
                        const Eigen::Matrix4f T0 = T;
                        Eigen::Matrix4f Tg = (S * T).eval();
                        ImGuizmo::Manipulate(
                            view_m.data(), proj_m.data(),
                            ImGuizmo::ROTATE, ImGuizmo::LOCAL,
                            Tg.data(), nullptr, nullptr);
                        T = (S.inverse() * Tg).eval();

                        if (max_abs_diff(T, T0) > 1e-7f)
                        {
                            float yaw = *wind_yaw_deg;
                            float pitch = *wind_pitch_deg;
                            yaw_pitch_from_transform(T, *wind_yaw_deg, yaw, pitch);
                            if (std::isfinite(yaw) && std::isfinite(pitch))
                            {
                                *wind_yaw_deg = yaw;
                                *wind_pitch_deg = std::clamp(pitch, -89.9f, 89.9f);
                            }
                        }
                    }

                    state.scene_view_gizmo_over = ImGuizmo::IsOver();
                    state.scene_view_gizmo_using = ImGuizmo::IsUsing();
                }
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Uv projection"))
        {
            state.cloth_view = ViewportRect{ 0, 0, 0, 0, true };

            const auto info = renderer.uv_debug_texture_info();
            if (!info.valid)
            {
                ImGui::TextUnformatted("UV image not generated yet.");
                ImGui::TextUnformatted("Generate via: Controls -> SMPL -> Generate UV Image.");
            }
            else
            {
                if (renderer.smpl_uv_layout_loaded())
                    ImGui::Text("UV layout: %s", renderer.smpl_uv_layout_source().c_str());
                else
                    ImGui::TextUnformatted("UV layout: (missing) using fallback planar projection");

                const ImVec2 canvas_size = ImGui::GetContentRegionAvail();

                if (canvas_size.x > 1.0f && canvas_size.y > 1.0f)
                {
                    const auto prediction_info = renderer.sketch2uv_prediction_texture_info();
                    const bool show_prediction = prediction_info.valid && canvas_size.x > 320.0f;
                    const bool show_body_preview =
                        show_prediction
                        && predicted_pose_camera_available(state)
                        && canvas_size.x > 560.0f;
                    const float gap = show_prediction ? 10.0f : 0.0f;
                    const float total_gap = show_body_preview ? 2.0f * gap : gap;
                    const float projection_w = show_prediction
                        ? (show_body_preview
                            ? std::max(1.0f, std::floor((canvas_size.x - total_gap) * 0.42f))
                            : std::max(1.0f, std::floor((canvas_size.x - gap) * 0.58f)))
                        : canvas_size.x;
                    const float prediction_w = show_prediction
                        ? (show_body_preview
                            ? std::max(1.0f, std::floor((canvas_size.x - total_gap) * 0.29f))
                            : std::max(1.0f, canvas_size.x - gap - projection_w))
                        : 0.0f;
                    const float body_w = show_body_preview
                        ? std::max(1.0f, canvas_size.x - total_gap - projection_w - prediction_w)
                        : 0.0f;

                    UvHoverPoint hover_point;
                    UvPanePlacement body_preview;
                    ImGui::PushID("uv_projection_pane");
                    const UvPanePlacement projection = draw_uv_projection_pane(
                        state,
                        renderer,
                        info,
                        ImVec2(projection_w, canvas_size.y));
                    ImGui::PopID();

                    if (show_prediction)
                    {
                        ImGui::SameLine(0.0f, gap);
                        ImGui::PushID("sketch2uv_prediction_pane");
                        draw_sketch2uv_prediction_pane(
                            renderer,
                            prediction_info,
                            ImVec2(prediction_w, canvas_size.y),
                            hover_point);
                        ImGui::PopID();
                        draw_uv_hover_indicator(projection, hover_point);

                        if (show_body_preview)
                        {
                            ImGui::SameLine(0.0f, gap);
                            ImGui::PushID("smpl_body_preview_pane");
                            body_preview = draw_smpl_body_preview_pane(
                                state,
                                renderer,
                                ImVec2(body_w, canvas_size.y));
                            ImGui::PopID();
                            draw_smpl_body_hover_indicator(renderer, body_preview, hover_point);
                        }
                    }
                }
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Iso projection"))
        {
            state.cloth_view = ViewportRect{ 0, 0, 0, 0, true };

            ImGui::SetNextItemWidth(140.0f);
            ImGui::InputFloat(
                "ISO distance threshold",
                &state.iso_distance_threshold,
                0.01f,
                0.05f,
                "%.4f");
            state.iso_distance_threshold =
                std::clamp(state.iso_distance_threshold, 0.0f, 10.0f);
            ImGui::SameLine();
            ImGui::TextDisabled("Blue hover = unmapped");

            const auto prediction_info = renderer.sketch2iso_prediction_texture_info();
            if (!prediction_info.valid)
            {
                ImGui::TextUnformatted("ISO sketch is not available yet.");
                ImGui::TextUnformatted("Generate or load an ISO map from the canva tab.");
            }
            else
            {
                const ImVec2 canvas_size = ImGui::GetContentRegionAvail();
                if (canvas_size.x > 1.0f && canvas_size.y > 1.0f)
                {
                    const bool show_body_preview =
                        predicted_pose_camera_available(state)
                        && canvas_size.x > 420.0f;
                    const float gap = show_body_preview ? 10.0f : 0.0f;
                    const float sketch_width = show_body_preview
                        ? std::max(1.0f, std::floor((canvas_size.x - gap) * 0.5f))
                        : canvas_size.x;
                    const float body_width = show_body_preview
                        ? std::max(1.0f, canvas_size.x - gap - sketch_width)
                        : 0.0f;

                    IsoHoverPoint hover_point;
                    ImGui::PushID("sketch2iso_prediction_pane");
                    draw_sketch2iso_prediction_pane(
                        renderer,
                        prediction_info,
                        state.iso_distance_threshold,
                        ImVec2(sketch_width, canvas_size.y),
                        hover_point);
                    ImGui::PopID();

                    if (show_body_preview)
                    {
                        ImGui::SameLine(0.0f, gap);
                        ImGui::PushID("iso_smpl_body_preview_pane");
                        const UvPanePlacement body_preview = draw_smpl_body_preview_pane(
                            state,
                            renderer,
                            ImVec2(body_width, canvas_size.y),
                            true);
                        ImGui::PopID();
                        draw_smpl_body_iso_hover_indicator(renderer, body_preview, hover_point);
                    }

                    if (hover_point.sampled)
                    {
                        ImGui::SetTooltip(
                            hover_point.mapped
                                ? "Vertex %d, ISO distance %.5f"
                                : "Unmapped, nearest vertex %d, ISO distance %.5f",
                            hover_point.vertex,
                            hover_point.distance);
                    }
                }
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("canva"))
        {
            state.canva_tab_active = true;
            state.cloth_view = ViewportRect{ 0, 0, 0, 0, true };

            ui_canva::draw_canva_tab(state, renderer);

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
        ImGui::End();
    }
}
