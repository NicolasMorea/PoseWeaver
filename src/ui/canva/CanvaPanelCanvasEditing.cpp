#include "ui/canva/CanvaPanelInternal.h"

#include "ui/canva/CanvaPreprocess.h"
#include "core/Paths.h"
#include "render/Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cerrno>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tinyfiledialogs.h>
#include <cnpy.h>

namespace poseweaver::ui_canva
{
    static void canva_ensure_layer_texture(CanvaLayer& layer, int w, int h)
    {
        if (layer.texture_id == 0)
            glGenTextures(1, &layer.texture_id);

        glBindTexture(GL_TEXTURE_2D, layer.texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
            layer.pixels.empty() ? nullptr : layer.pixels.data());

        glBindTexture(GL_TEXTURE_2D, 0);
        layer.dirty = false;
    }

     void canva_upload_layer(CanvaLayer& layer, int w, int h)
    {
        if (layer.texture_id == 0)
        {
            canva_ensure_layer_texture(layer, w, h);
            return;
        }

        glBindTexture(GL_TEXTURE_2D, layer.texture_id);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, layer.pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        layer.dirty = false;
    }

     CanvaLayer canva_make_layer(int w, int h, int id, const std::string& name)
    {
        CanvaLayer layer;
        layer.id = id;
        layer.name = name;
        layer.enabled = true;
        layer.offset = ImVec2(0.0f, 0.0f);
        layer.scale = 1.0f;
        layer.pixels.assign((size_t)w * (size_t)h * 4u, 0u); // transparent
        layer.dirty = true;
        layer.bounds_dirty = true;
        layer.bounds_min_x = 0;
        layer.bounds_min_y = 0;
        layer.bounds_max_x_excl = w;
        layer.bounds_max_y_excl = h;
        canva_ensure_layer_texture(layer, w, h);
        return layer;
    }

     void canva_delete_layer_texture(CanvaLayer& layer)
    {
        if (layer.texture_id != 0)
        {
            glDeleteTextures(1, &layer.texture_id);
            layer.texture_id = 0;
        }
    }

    void canva_delete_sequence_preview_textures(CanvaState& canva)
    {
        for (CanvaLayer& layer : canva.sequence_preview_layers)
            canva_delete_layer_texture(layer);
        canva.sequence_preview_layers.clear();
        canva.sequence_preview_active = false;
        canva.sequence_preview_frame_id = 0;
    }

     int canva_find_layer_index_by_id(const std::vector<CanvaLayer>& layers, int layer_id)
    {
        for (int i = 0; i < (int)layers.size(); ++i)
        {
            if (layers[(size_t)i].id == layer_id)
                return i;
        }
        return -1;
    }

     void canva_prune_undo_for_layer(std::vector<CanvaUndo>& undo, int layer_id)
    {
        undo.erase(
            std::remove_if(undo.begin(), undo.end(), [&](const CanvaUndo& u) { return u.layer_id == layer_id; }),
            undo.end());
    }

     void canva_push_undo(CanvaState& canva, int layer_id, const std::vector<unsigned char>& pixels, const ImVec2& offset, float scale)
    {
        constexpr int kMaxUndoSteps = 30;
        CanvaUndo u;
        u.layer_id = layer_id;
        u.pixels = pixels;
        u.offset = offset;
        u.scale = scale;
        canva.undo.emplace_back(std::move(u));
        if ((int)canva.undo.size() > kMaxUndoSteps)
            canva.undo.erase(canva.undo.begin(), canva.undo.begin() + ((int)canva.undo.size() - kMaxUndoSteps));
    }

     bool canva_apply_undo(CanvaState& canva)
    {
        if (canva.undo.empty())
            return false;

        CanvaUndo u = std::move(canva.undo.back());
        canva.undo.pop_back();

        const int idx = canva_find_layer_index_by_id(canva.layers, u.layer_id);
        if (idx < 0)
            return false;

        CanvaLayer& layer = canva.layers[(size_t)idx];
        layer.pixels = std::move(u.pixels);
        layer.offset = u.offset;
        layer.scale = u.scale;
        layer.dirty = true;
        layer.bounds_dirty = true;

        canva.active_layer = idx;
        canva.drawing = false;
        canva.sizing = false;
        canva.transforming = false;
        canva.transform_layer = -1;

        return true;
    }

    static void canva_blend_over(unsigned char* dst_rgba, const ImVec4& src)
    {
        const float sa = std::clamp(src.w, 0.0f, 1.0f);
        const float sr = std::clamp(src.x, 0.0f, 1.0f);
        const float sg = std::clamp(src.y, 0.0f, 1.0f);
        const float sb = std::clamp(src.z, 0.0f, 1.0f);

        const float dr = (float)dst_rgba[0] / 255.0f;
        const float dg = (float)dst_rgba[1] / 255.0f;
        const float db = (float)dst_rgba[2] / 255.0f;
        const float da = (float)dst_rgba[3] / 255.0f;

        const float out_a = sa + da * (1.0f - sa);
        if (out_a <= 1e-6f)
        {
            dst_rgba[0] = dst_rgba[1] = dst_rgba[2] = dst_rgba[3] = 0;
            return;
        }

        const float out_r = (sr * sa + dr * da * (1.0f - sa)) / out_a;
        const float out_g = (sg * sa + dg * da * (1.0f - sa)) / out_a;
        const float out_b = (sb * sa + db * da * (1.0f - sa)) / out_a;

        dst_rgba[0] = (unsigned char)std::clamp<int>((int)(out_r * 255.0f + 0.5f), 0, 255);
        dst_rgba[1] = (unsigned char)std::clamp<int>((int)(out_g * 255.0f + 0.5f), 0, 255);
        dst_rgba[2] = (unsigned char)std::clamp<int>((int)(out_b * 255.0f + 0.5f), 0, 255);
        dst_rgba[3] = (unsigned char)std::clamp<int>((int)(out_a * 255.0f + 0.5f), 0, 255);
    }

     void canva_stamp_circle(CanvaLayer& layer, int w, int h, float cx, float cy, float radius, const ImVec4& col)
    {
        if (layer.pixels.empty() || w <= 0 || h <= 0)
            return;

        const int r = (int)std::ceil(std::max(0.5f, radius));
        const int x0 = std::max(0, (int)std::floor(cx) - r);
        const int x1 = std::min(w - 1, (int)std::floor(cx) + r);
        const int y0 = std::max(0, (int)std::floor(cy) - r);
        const int y1 = std::min(h - 1, (int)std::floor(cy) + r);

        const float rr = radius * radius;
        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x)
            {
                const float dx = ((float)x + 0.5f) - cx;
                const float dy = ((float)y + 0.5f) - cy;
                if (dx * dx + dy * dy <= rr)
                {
                    unsigned char* px = &layer.pixels[((size_t)y * (size_t)w + (size_t)x) * 4u];
                    canva_blend_over(px, col);
                }
            }
        }
        layer.dirty = true;
        layer.bounds_dirty = true;
    }

     void canva_draw_segment(CanvaLayer& layer, int w, int h, ImVec2 a, ImVec2 b, float radius, const ImVec4& col)
    {
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float dist = std::sqrt(dx * dx + dy * dy);

        const float step = std::max(0.5f, radius * 0.35f);
        const int n = (dist > 1e-4f) ? (int)std::ceil(dist / step) : 1;

        for (int i = 0; i <= n; ++i)
        {
            const float t = (n > 0) ? ((float)i / (float)n) : 0.0f;
            const float x = a.x + dx * t;
            const float y = a.y + dy * t;
            canva_stamp_circle(layer, w, h, x, y, radius, col);
        }
    }

    static bool canva_layer_has_opaque_pixel_at(const CanvaLayer& layer, int w, int h, const ImVec2& canvas_pos)
    {
        const float s = std::max(1e-6f, layer.scale);
        const ImVec2 p = ImVec2((canvas_pos.x - layer.offset.x) / s, (canvas_pos.y - layer.offset.y) / s);
        const int x = (int)std::floor(p.x);
        const int y = (int)std::floor(p.y);
        if (x < 0 || y < 0 || x >= w || y >= h)
            return false;
        if (layer.pixels.empty())
            return false;
        const size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 4u;
        return layer.pixels[idx + 3u] != 0u;
    }

     int canva_pick_layer_at(const CanvaState& canva, const ImVec2& canvas_pos)
    {
        // layers are stored top->bottom, so iterate forward for topmost.
        for (int i = 0; i < (int)canva.layers.size(); ++i)
        {
            const CanvaLayer& layer = canva.layers[(size_t)i];
            if (!layer.enabled)
                continue;
            if (canva_layer_has_opaque_pixel_at(layer, canva.width, canva.height, canvas_pos))
                return i;
        }
        return -1;
    }

     void canva_cancel_interactions(CanvaState& canva)
    {
        canva.drawing = false;
        canva.sizing = false;
        canva.transforming = false;
        canva.scaling = false;
        canva.transform_layer = -1;
    }

     void canva_reset_all(AppState& state, CanvaState& canva, Renderer& renderer)
    {
        for (CanvaLayer& layer : canva.layers)
            canva_delete_layer_texture(layer);
        canva_delete_sequence_preview_textures(canva);
        for (CanvaSequenceTrackFrame& frame : canva.sequence_cloth_frames)
            canva_delete_layer_texture(frame.layer);
        for (CanvaSequenceTrackFrame& frame : canva.sequence_sketch_frames)
            canva_delete_layer_texture(frame.layer);
        canva_delete_render_frame_dirs();

        std::future<RigOptimizationJobResult> optimizer_future =
            std::move(canva.rig_optimization_future);
        const bool optimizer_pending = optimizer_future.valid();

        const std::uint64_t rig_active = state.sketch2rig.active_request_id;
        const std::uint64_t betas_active = state.sketch2betas.active_request_id;
        const std::uint64_t uv_active = state.sketch2uv.active_request_id;
        const std::uint64_t iso_active = state.sketch2iso.active_request_id;
        const std::uint64_t depth_active = state.sketch2depth.active_request_id;
        const std::uint64_t vitpose_active = state.vitpose.active_request_id;

        CanvaState fresh;
        fresh.brush_radius = state.canva_brush_radius;
        fresh.brush_color = ImVec4(
            state.canva_brush_color[0],
            state.canva_brush_color[1],
            state.canva_brush_color[2],
            state.canva_brush_color[3]);
        fresh.last_undo_request_id = state.canva_undo_request_id;
        fresh.last_sketch2rig_import_request_id =
            std::max(state.sketch2rig.completed_request_id, rig_active);
        fresh.last_sketch2betas_import_request_id =
            std::max(state.sketch2betas.completed_request_id, betas_active);
        fresh.last_sketch2uv_import_request_id =
            std::max(state.sketch2uv.completed_request_id, uv_active);
        fresh.last_sketch2iso_import_request_id =
            std::max(state.sketch2iso.completed_request_id, iso_active);
        fresh.last_sketch2depth_import_request_id =
            std::max(state.sketch2depth.completed_request_id, depth_active);
        fresh.last_vitpose_import_request_id =
            std::max(state.vitpose.completed_request_id, vitpose_active);
        fresh.ignore_sketch2rig_request_id = rig_active;
        fresh.ignore_sketch2betas_request_id = betas_active;
        fresh.ignore_sketch2uv_request_id = uv_active;
        fresh.ignore_sketch2iso_request_id = iso_active;
        fresh.ignore_sketch2depth_request_id = depth_active;
        fresh.ignore_vitpose_request_id = vitpose_active;
        fresh.last_scene_view_snapshot_completed_id = state.scene_view_snapshot_request_id;
        fresh.last_displacement_completed_id = state.canva_displacement_request_id;
        fresh.last_iso_displacement_completed_id = state.canva_iso_displacement_request_id;
        fresh.last_cloth_render_completed_id = state.cloth_render_request_id;
        fresh.sequence_frame_rate = canva.sequence_frame_rate;
        fresh.sequence_sub_frames = canva.sequence_sub_frames;
        fresh.sequence_generate_amount = canva.sequence_generate_amount;
        fresh.rig_optimization_future = std::move(optimizer_future);
        fresh.rig_optimization_job_running = optimizer_pending;
        fresh.rig_optimization_discard_result = optimizer_pending;
        fresh.next_layer_id = 1;
        fresh.layers.emplace_back(
            canva_make_layer(fresh.width, fresh.height, fresh.next_layer_id++, "Layer 1"));
        fresh.active_layer = 0;
        canva = std::move(fresh);

        state.sketch2rig.last_sketch_path.clear();
        state.sketch2rig.last_output_path.clear();
        state.sketch2betas.last_sketch_path.clear();
        state.sketch2betas.last_output_path.clear();
        state.sketch2uv.last_sketch_path.clear();
        state.sketch2uv.last_output_path.clear();
        state.sketch2iso.last_sketch_path.clear();
        state.sketch2iso.last_output_path.clear();
        state.sketch2depth.last_sketch_path.clear();
        state.sketch2depth.last_output_path.clear();
        state.vitpose.last_sketch_path.clear();
        state.vitpose.last_output_path.clear();

        state.selection.pose_index = -1;
        state.pending_pose_selection.clear();
        state.pose_asset_reload_request_id++;
        state.smpl_shape.betas.fill(0.0f);

        state.scene_view_target_valid = false;
        state.scene_view_sequence_rig_overlay_active = false;
        state.scene_view_sequence_rig_overlay_visible = false;
        state.scene_view_sequence_rig_overlay_path.clear();
        state.scene_view_raw_rig_root_valid = false;
        state.scene_view_raw_rig_height_valid = false;
        state.scene_view_smpl_fit_valid = false;
        state.scene_view_smpl_fit_scale = { 1.0f, 1.0f, 1.0f };
        state.scene_view_smpl_fit_translation = { 0.0f, 0.0f, 0.0f };
        state.scene_view_smpl_fit_rms = 0.0f;
        state.scene_view_snapshot_pose_name.clear();
        state.scene_view_snapshot_path.clear();
        state.scene_view_snapshot_use_rigopt_camera = false;
        state.scene_view_snapshot_rigopt_camera_scale = 1.0f;
        state.scene_view_snapshot_rigopt_camera_translation = { 0.0f, 0.0f };

        state.cloth_render_path.clear();
        state.cloth_render_stroke_reference_path.clear();
        state.cloth_render_output_dir.clear();
        state.cloth_render_camera_rig_path.clear();
        state.cloth_render_fine = false;
        state.cloth_render_use_current_camera = false;
        state.cloth_render_source_vertex_count = 0;
        state.cloth_render_vertex_count = 0;
        state.cloth_render_face_count = 0;

        state.canva_displacement_uv_json_path.clear();
        state.canva_displacement_model_input_path.clear();
        state.canva_displacement_path.clear();
        state.canva_manual_uv_png_path.clear();
        state.canva_manual_iso_npz_path.clear();
        state.canva_iso_displacement_npz_path.clear();
        state.canva_iso_displacement_model_input_path.clear();
        state.canva_iso_displacement_path.clear();
        state.cloth_fitting_output_dir.clear();

        state.uv_view_zoom = 1.0f;
        state.uv_view_pan_x = 0.0f;
        state.uv_view_pan_y = 0.0f;
        state.smpl_color_by_uv = false;
        state.garment_color_by_smpl_uv = false;

        renderer.reset_computed_state(state);
        console_add_line(state, "Canva: Reset all computed data");
    }}
