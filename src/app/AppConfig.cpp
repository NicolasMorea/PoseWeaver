#include "app/AppConfig.h"

#include "app/AppState.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <imgui.h>
#include <imgui_internal.h>

namespace
{
    static bool g_read_seen_collision_geometry = false;
    static bool g_read_seen_simulation_steps = false;

    static void* pw_imgui_settings_read_open(ImGuiContext*, ImGuiSettingsHandler* handler, const char* name)
    {
        if (!handler || !name)
            return nullptr;
        if (std::strcmp(name, "State") != 0)
            return nullptr;
        g_read_seen_collision_geometry = false;
        g_read_seen_simulation_steps = false;
        return handler->UserData;
    }

    static void infer_simulation_steps_from_legacy_dt(poseweaver::AppState* state)
    {
        if (!state || g_read_seen_simulation_steps)
            return;
        const float duration = std::max(0.0f, state->xpbd_params.duration_seconds);
        const float dt = state->xpbd_params.dt;
        if (duration > 0.0f && dt > 0.0f && std::isfinite(duration) && std::isfinite(dt))
            state->xpbd_params.simulation_steps = std::clamp(
                (int)std::llround((double)duration / (double)dt),
                1,
                1000000);
    }

    static void pw_imgui_settings_read_line(ImGuiContext*, ImGuiSettingsHandler* handler, void*, const char* line)
    {
        if (!handler || !handler->UserData || !line)
            return;

        auto* state = static_cast<poseweaver::AppState*>(handler->UserData);

        float v = 0.0f;
        int iv = 0;
        if (std::sscanf(line, "GarmentMinClothSkinDistance=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->garment_min_cloth_skin_distance = std::clamp(v, 0.0f, 0.02f);
        }

        if (std::sscanf(line, "GarmentVertexMergeEnabled=%d", &iv) == 1)
        {
            state->garment_vertex_merge_enabled = (iv != 0);
        }
        if (std::sscanf(line, "GarmentVertexMergeDistance=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->garment_vertex_merge_distance = std::clamp(v, 0.0f, 0.05f);
        }
        if (std::sscanf(line, "ClothToonShader=%d", &iv) == 1)
        {
            state->cloth_ink_style = iv != 0 ? poseweaver::ClothInkStyle::Detailed : poseweaver::ClothInkStyle::Off;
        }
        if (std::sscanf(line, "ClothInkStyle=%d", &iv) == 1)
        {
            state->cloth_ink_style = static_cast<poseweaver::ClothInkStyle>(std::clamp(iv, 0, 3));
        }
        {
            float r = 1.0f, g = 1.0f, b = 1.0f;
            if (std::sscanf(line, "ClothInkColor=%f,%f,%f", &r, &g, &b) == 3)
            {
                if (std::isfinite(r) && std::isfinite(g) && std::isfinite(b))
                {
                    state->cloth_ink_color[0] = std::clamp(r, 0.0f, 1.0f);
                    state->cloth_ink_color[1] = std::clamp(g, 0.0f, 1.0f);
                    state->cloth_ink_color[2] = std::clamp(b, 0.0f, 1.0f);
                }
            }
        }

        if (std::sscanf(line, "SimulationPreview=%d", &iv) == 1)
        {
            state->controls.preview = (iv != 0);
        }

        if (std::sscanf(line, "SmoothSequence=%d", &iv) == 1)
        {
            state->controls.smooth_sequence = (iv != 0);
        }
        if (std::sscanf(line, "SmoothSequenceLength=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->controls.smooth_sequence_length = std::clamp(v, 0.0f, 1.0f);
        }
        if (std::sscanf(line, "SaveSmplSequence=%d", &iv) == 1)
        {
            state->controls.save_smpl_sequence = (iv != 0);
        }
        if (std::sscanf(line, "ShowCollisionProxySequence=%d", &iv) == 1)
        {
            state->controls.show_collision_proxy_sequence = (iv != 0);
        }
        if (std::sscanf(line, "CanvaSequencePoseTransitionFrames=%d", &iv) == 1)
        {
            state->canva_sequence_pose_transition_frames = std::clamp(iv, 0, 240);
        }
        if (std::sscanf(line, "CanvaPreferredBonePrediction=%d", &iv) == 1)
        {
            state->canva_preferred_bone_prediction = std::clamp(iv, 0, 1);
        }

        if (std::sscanf(line, "SimulationMaxFrames=%d", &iv) == 1)
        {
            state->controls.max_frames = std::clamp(iv, 1, 1000000);
        }
        if (std::sscanf(line, "SimulationFps=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->controls.fps = std::clamp(v, 0.1f, 10000.0f);
        }

        if (std::sscanf(line, "WindYawDeg=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_yaw_deg = std::clamp(v, -3600.0f, 3600.0f);
        }
        if (std::sscanf(line, "WindPitchDeg=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_pitch_deg = std::clamp(v, -89.9f, 89.9f);
        }

        if (std::sscanf(line, "LightPosX=%f", &v) == 1)
        {
            if (std::isfinite(v))
            {
                state->light_position[0] = std::clamp(v, -10000.0f, 10000.0f);
                state->light_position_initialized = true;
            }
        }
        if (std::sscanf(line, "LightPosY=%f", &v) == 1)
        {
            if (std::isfinite(v))
            {
                state->light_position[1] = std::clamp(v, -10000.0f, 10000.0f);
                state->light_position_initialized = true;
            }
        }
        if (std::sscanf(line, "LightPosZ=%f", &v) == 1)
        {
            if (std::isfinite(v))
            {
                state->light_position[2] = std::clamp(v, -10000.0f, 10000.0f);
                state->light_position_initialized = true;
            }
        }

        if (std::sscanf(line, "WindRandomMagnitude=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_random_magnitude = std::clamp(v, 0.0f, 1e9f);
        }
        if (std::sscanf(line, "WindRandomTimeScale=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_random_time_scale = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "WindRandomSpatialScale=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_random_spatial_scale = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "WindRandomSpace=%f", &v) == 1)
        {
            if (std::isfinite(v))
            {
                state->xpbd_params.wind_random_spatial_scale = std::clamp(v, 0.0f, 1000.0f);
                state->xpbd_params.wind_random_magnitude = std::max(
                    state->xpbd_params.wind_random_magnitude, std::clamp(v, 0.0f, 1e9f));
            }
        }
        if (std::sscanf(line, "WindRandomTime=%f", &v) == 1)
        {
            if (std::isfinite(v))
            {
                state->xpbd_params.wind_random_time_scale = std::clamp(v, 0.0f, 1000.0f);
                state->xpbd_params.wind_random_magnitude = std::max(
                    state->xpbd_params.wind_random_magnitude, std::clamp(v, 0.0f, 1e9f));
            }
        }

        if (std::sscanf(line, "CanvaBrushRadius=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->canva_brush_radius = std::clamp(v, 0.5f, 250.0f);
        }
        if (std::sscanf(line, "ClothFittingDepthMargin=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->cloth_fitting_depth_margin = std::clamp(v, 0.0f, 0.05f);
        }
        if (std::sscanf(line, "ClothFittingDepthMarginPred=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->cloth_fitting_depth_margin_pred = std::clamp(v, 0.0f, 1.0f);
        }
        if (std::sscanf(line, "ClothFittingBodyDistanceThreshold=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->cloth_fitting_body_distance_threshold = std::clamp(v, 0.0f, 1.0f);
        }
        if (std::sscanf(line, "ClothFittingWarpRadius=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->cloth_fitting_warp_radius = std::clamp(v, 0.0f, 512.0f);
        }
        if (std::sscanf(line, "ClothFittingWarpForce=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->cloth_fitting_warp_force = std::clamp(v, 0.0f, 6.0f);
        }
        if (std::sscanf(line, "ClothFittingWarpSteps=%d", &iv) == 1)
        {
            state->cloth_fitting_warp_steps = std::clamp(iv, 1, 32);
        }
        if (std::sscanf(line, "IsoDistanceThreshold=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->iso_distance_threshold = std::clamp(v, 0.0f, 10.0f);
        }
        if (std::sscanf(line, "ClothRenderTargetDetailSize=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->cloth_render_target_detail_size = std::clamp(v, 0.0001f, 0.25f);
        }
        {
            float r = 0.0f, g = 0.0f, b = 0.0f, a = 1.0f;
            if (std::sscanf(line, "CanvaBrushColor=%f,%f,%f,%f", &r, &g, &b, &a) == 4)
            {
                if (std::isfinite(r) && std::isfinite(g) && std::isfinite(b) && std::isfinite(a))
                {
                    state->canva_brush_color[0] = std::clamp(r, 0.0f, 1.0f);
                    state->canva_brush_color[1] = std::clamp(g, 0.0f, 1.0f);
                    state->canva_brush_color[2] = std::clamp(b, 0.0f, 1.0f);
                    state->canva_brush_color[3] = std::clamp(a, 0.0f, 1.0f);
                }
            }
        }

        if (std::sscanf(line, "XpbdDt=%f", &v) == 1)
        {
            if (std::isfinite(v))
            {
                state->xpbd_params.dt = std::clamp(v, 1e-6f, 1.0f);
                infer_simulation_steps_from_legacy_dt(state);
            }
        }
        if (std::sscanf(line, "SimulationDevice=%d", &iv) == 1)
        {
            state->xpbd_params.device = iv == 1
                ? poseweaver::SimulationDevice::Gpu
                : poseweaver::SimulationDevice::Cpu;
        }
        if (std::sscanf(line, "XpbdDurationSeconds=%f", &v) == 1)
        {
            if (std::isfinite(v))
            {
                state->xpbd_params.duration_seconds = std::clamp(v, 0.0f, 3600.0f);
                infer_simulation_steps_from_legacy_dt(state);
            }
        }
        if (std::sscanf(line, "XpbdSimulationSteps=%d", &iv) == 1)
        {
            g_read_seen_simulation_steps = true;
            state->xpbd_params.simulation_steps = std::clamp(iv, 1, 1000000);
            state->xpbd_params.dt = poseweaver::solver_dt_seconds(state->xpbd_params);
        }
        if (std::sscanf(line, "XpbdSolverIterations=%d", &iv) == 1)
        {
            state->xpbd_params.solver_iterations = std::clamp(iv, 1, 200);
        }
        if (std::sscanf(line, "XpbdSubsteps=%d", &iv) == 1)
        {
            state->xpbd_params.substeps = std::clamp(iv, 1, 16);
        }
        if (std::sscanf(line, "XpbdGravity=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.gravity = std::clamp(v, -1000.0f, 1000.0f);
        }

        if (std::sscanf(line, "XpbdStretchEnabled=%d", &iv) == 1)
        {
            state->xpbd_params.stretch_enabled = (iv != 0);
        }
        if (std::sscanf(line, "XpbdStretchCompliance=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.stretch_compliance = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "XpbdBaseStretchMult=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.base_stretch_mult = std::clamp(v, 0.0f, 10.0f);
        }

        if (std::sscanf(line, "XpbdBendEnabled=%d", &iv) == 1)
        {
            state->xpbd_params.bend_enabled = (iv != 0);
        }
        if (std::sscanf(line, "XpbdBendCompliance=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.bend_compliance = std::clamp(v, 0.0f, 1000.0f);
        }

        if (std::sscanf(line, "XpbdCollisionEnabled=%d", &iv) == 1)
        {
            state->xpbd_params.collision_enabled = (iv != 0);
        }
        if (std::sscanf(line, "XpbdCollisionCompliance=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.collision_compliance = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "XpbdCollisionThickness=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.collision_thickness = std::clamp(v, 0.0f, 10.0f);
        }
        if (std::sscanf(line, "XpbdSelfCollisionEnabled=%d", &iv) == 1)
        {
            state->xpbd_params.self_collision_enabled = (iv != 0);
        }
        if (std::sscanf(line, "XpbdSelfCollisionCompliance=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.self_collision_compliance = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "XpbdSelfCollisionThickness=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.self_collision_thickness = std::clamp(v, 0.0f, 10.0f);
        }

        if (std::sscanf(line, "XpbdFrictionEnabled=%d", &iv) == 1)
        {
            state->xpbd_params.enable_friction = (iv != 0);
        }
        if (std::sscanf(line, "XpbdFrictionMu=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.friction_mu = std::clamp(v, 0.0f, 10.0f);
        }
        if (std::sscanf(line, "XpbdDamping=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.damping = std::clamp(v, 0.0f, 0.99f);
        }

        if (std::sscanf(line, "XpbdWindForce=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_force = std::clamp(v, 0.0f, 1e9f);
        }
        if (std::sscanf(line, "XpbdWindYawDeg=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_yaw_deg = std::clamp(v, -3600.0f, 3600.0f);
        }
        if (std::sscanf(line, "XpbdWindPitchDeg=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_pitch_deg = std::clamp(v, -89.9f, 89.9f);
        }
        if (std::sscanf(line, "XpbdWindRandomMagnitude=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_random_magnitude = std::clamp(v, 0.0f, 1e9f);
        }
        if (std::sscanf(line, "XpbdWindRandomTimeScale=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_random_time_scale = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "XpbdWindRandomSpatialScale=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_random_spatial_scale = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "XpbdWindRandomSpace=%f", &v) == 1)
        {
            if (std::isfinite(v))
            {
                state->xpbd_params.wind_random_spatial_scale = std::clamp(v, 0.0f, 1000.0f);
                state->xpbd_params.wind_random_magnitude = std::max(
                    state->xpbd_params.wind_random_magnitude, std::clamp(v, 0.0f, 1e9f));
            }
        }
        if (std::sscanf(line, "XpbdWindRandomTime=%f", &v) == 1)
        {
            if (std::isfinite(v))
            {
                state->xpbd_params.wind_random_time_scale = std::clamp(v, 0.0f, 1000.0f);
                state->xpbd_params.wind_random_magnitude = std::max(
                    state->xpbd_params.wind_random_magnitude, std::clamp(v, 0.0f, 1e9f));
            }
        }
        if (std::sscanf(line, "XpbdCollisionGeometry=%d", &iv) == 1)
        {
            g_read_seen_collision_geometry = true;
            iv = std::clamp(iv, 0, 2);
            state->xpbd_params.collision_geometry = (iv == 1)
                ? poseweaver::CollisionGeometryType::SimplifiedSmpl
                : (iv == 2) ? poseweaver::CollisionGeometryType::SimpleProxy : poseweaver::CollisionGeometryType::RawSmpl;
            state->xpbd_params.simplify_smpl =
                state->xpbd_params.collision_geometry == poseweaver::CollisionGeometryType::SimplifiedSmpl;
        }
        if (std::sscanf(line, "XpbdSimplifySmpl=%d", &iv) == 1)
        {
            if (!g_read_seen_collision_geometry)
            {
                state->xpbd_params.simplify_smpl = (iv != 0);
                state->xpbd_params.collision_geometry = state->xpbd_params.simplify_smpl
                    ? poseweaver::CollisionGeometryType::SimplifiedSmpl
                    : poseweaver::CollisionGeometryType::RawSmpl;
            }
        }
        if (std::sscanf(line, "XpbdSmplGeometryPercent=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.smpl_geometry_percent = std::clamp(v, 1.0f, 100.0f);
        }
    }

    static void pw_imgui_settings_write_all(ImGuiContext*, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
    {
        if (!handler || !handler->UserData || !buf)
            return;

        auto* state = static_cast<poseweaver::AppState*>(handler->UserData);

        buf->appendf("[PoseWeaver][State]\n");
        buf->appendf("GarmentMinClothSkinDistance=%.6f\n", state->garment_min_cloth_skin_distance);
        buf->appendf("GarmentVertexMergeEnabled=%d\n", state->garment_vertex_merge_enabled ? 1 : 0);
        buf->appendf("GarmentVertexMergeDistance=%.9f\n", state->garment_vertex_merge_distance);
        buf->appendf("ClothInkStyle=%d\n", static_cast<int>(state->cloth_ink_style));
        buf->appendf("ClothInkColor=%.6f,%.6f,%.6f\n",
            state->cloth_ink_color[0], state->cloth_ink_color[1], state->cloth_ink_color[2]);
        buf->appendf("SimulationPreview=%d\n", state->controls.preview ? 1 : 0);
        buf->appendf("SmoothSequence=%d\n", state->controls.smooth_sequence ? 1 : 0);
        buf->appendf("SmoothSequenceLength=%.6f\n", state->controls.smooth_sequence_length);
        buf->appendf("SaveSmplSequence=%d\n", state->controls.save_smpl_sequence ? 1 : 0);
        buf->appendf("ShowCollisionProxySequence=%d\n", state->controls.show_collision_proxy_sequence ? 1 : 0);
        buf->appendf("CanvaSequencePoseTransitionFrames=%d\n", state->canva_sequence_pose_transition_frames);
        buf->appendf("CanvaPreferredBonePrediction=%d\n", std::clamp(state->canva_preferred_bone_prediction, 0, 1));

        buf->appendf("SimulationFps=%.9f\n", state->controls.fps);
        buf->appendf("LightPosX=%.6f\n", state->light_position[0]);
        buf->appendf("LightPosY=%.6f\n", state->light_position[1]);
        buf->appendf("LightPosZ=%.6f\n", state->light_position[2]);

        buf->appendf("CanvaBrushRadius=%.6f\n", state->canva_brush_radius);
        buf->appendf("CanvaBrushColor=%.6f,%.6f,%.6f,%.6f\n",
            state->canva_brush_color[0], state->canva_brush_color[1], state->canva_brush_color[2], state->canva_brush_color[3]);
        buf->appendf("ClothFittingDepthMargin=%.9f\n", state->cloth_fitting_depth_margin);
        buf->appendf("ClothFittingDepthMarginPred=%.9f\n", state->cloth_fitting_depth_margin_pred);
        buf->appendf("ClothFittingBodyDistanceThreshold=%.9f\n", state->cloth_fitting_body_distance_threshold);
        buf->appendf("ClothFittingWarpRadius=%.6f\n", state->cloth_fitting_warp_radius);
        buf->appendf("ClothFittingWarpForce=%.6f\n", state->cloth_fitting_warp_force);
        buf->appendf("ClothFittingWarpSteps=%d\n", state->cloth_fitting_warp_steps);
        buf->appendf("IsoDistanceThreshold=%.9f\n", state->iso_distance_threshold);
        buf->appendf("ClothRenderTargetDetailSize=%.9f\n", state->cloth_render_target_detail_size);

        buf->appendf("SimulationDevice=%d\n",
            state->xpbd_params.device == poseweaver::SimulationDevice::Gpu ? 1 : 0);
        buf->appendf("XpbdDt=%.9f\n", poseweaver::solver_dt_seconds(state->xpbd_params));
        buf->appendf("XpbdDurationSeconds=%.9f\n", state->xpbd_params.duration_seconds);
        buf->appendf("XpbdSimulationSteps=%d\n", state->xpbd_params.simulation_steps);
        buf->appendf("XpbdSolverIterations=%d\n", state->xpbd_params.solver_iterations);
        buf->appendf("XpbdSubsteps=%d\n", state->xpbd_params.substeps);
        buf->appendf("XpbdGravity=%.9f\n", state->xpbd_params.gravity);

        buf->appendf("XpbdStretchEnabled=%d\n", state->xpbd_params.stretch_enabled ? 1 : 0);
        buf->appendf("XpbdStretchCompliance=%.9f\n", state->xpbd_params.stretch_compliance);
        buf->appendf("XpbdBaseStretchMult=%.9f\n", state->xpbd_params.base_stretch_mult);

        buf->appendf("XpbdBendEnabled=%d\n", state->xpbd_params.bend_enabled ? 1 : 0);
        buf->appendf("XpbdBendCompliance=%.9f\n", state->xpbd_params.bend_compliance);

        buf->appendf("XpbdCollisionEnabled=%d\n", state->xpbd_params.collision_enabled ? 1 : 0);
        buf->appendf("XpbdCollisionCompliance=%.9f\n", state->xpbd_params.collision_compliance);
        buf->appendf("XpbdCollisionThickness=%.9f\n", state->xpbd_params.collision_thickness);
        buf->appendf("XpbdSelfCollisionEnabled=%d\n", state->xpbd_params.self_collision_enabled ? 1 : 0);
        buf->appendf("XpbdSelfCollisionCompliance=%.9f\n", state->xpbd_params.self_collision_compliance);
        buf->appendf("XpbdSelfCollisionThickness=%.9f\n", state->xpbd_params.self_collision_thickness);

        buf->appendf("XpbdFrictionEnabled=%d\n", state->xpbd_params.enable_friction ? 1 : 0);
        buf->appendf("XpbdFrictionMu=%.9f\n", state->xpbd_params.friction_mu);
        buf->appendf("XpbdDamping=%.9f\n", state->xpbd_params.damping);

        buf->appendf("XpbdWindForce=%.9f\n", state->xpbd_params.wind_force);
        buf->appendf("XpbdWindYawDeg=%.6f\n", state->xpbd_params.wind_yaw_deg);
        buf->appendf("XpbdWindPitchDeg=%.6f\n", state->xpbd_params.wind_pitch_deg);
        buf->appendf("XpbdWindRandomMagnitude=%.9f\n", state->xpbd_params.wind_random_magnitude);
        buf->appendf("XpbdWindRandomTimeScale=%.6f\n", state->xpbd_params.wind_random_time_scale);
        buf->appendf("XpbdWindRandomSpatialScale=%.6f\n", state->xpbd_params.wind_random_spatial_scale);
        const int collision_geometry = (state->xpbd_params.collision_geometry == poseweaver::CollisionGeometryType::SimplifiedSmpl)
            ? 1
            : (state->xpbd_params.collision_geometry == poseweaver::CollisionGeometryType::SimpleProxy) ? 2 : 0;
        buf->appendf("XpbdCollisionGeometry=%d\n", collision_geometry);
        buf->appendf("XpbdSmplGeometryPercent=%.6f\n", state->xpbd_params.smpl_geometry_percent);

        buf->appendf("\n");
    }
}

namespace poseweaver
{
    void install_imgui_state_settings_handler(AppState& state)
    {
        ImGuiSettingsHandler handler;
        handler.TypeName = "PoseWeaver";
        handler.TypeHash = ImHashStr("PoseWeaver");
        handler.UserData = &state;
        handler.ReadOpenFn = pw_imgui_settings_read_open;
        handler.ReadLineFn = pw_imgui_settings_read_line;
        handler.WriteAllFn = pw_imgui_settings_write_all;

        if (ImGuiContext* ctx = ImGui::GetCurrentContext())
            ctx->SettingsHandlers.push_back(handler);

        if (const char* ini = ImGui::GetIO().IniFilename)
            ImGui::LoadIniSettingsFromDisk(ini);
    }

    void save_imgui_settings_to_disk()
    {
        if (const char* ini = ImGui::GetIO().IniFilename)
            ImGui::SaveIniSettingsToDisk(ini);
    }
}
