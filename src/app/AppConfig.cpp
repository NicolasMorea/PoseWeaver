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
    static void* pw_imgui_settings_read_open(ImGuiContext*, ImGuiSettingsHandler* handler, const char* name)
    {
        if (!handler || !name)
            return nullptr;
        if (std::strcmp(name, "State") != 0)
            return nullptr;
        return handler->UserData;
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
            state->cloth_toon_shader = (iv != 0);
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

        if (std::sscanf(line, "SimulationDt=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.dt = std::clamp(v, 1e-6f, 1.0f);
        }
        if (std::sscanf(line, "SimulationDurationSeconds=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.duration_seconds = std::clamp(v, 0.0f, 3600.0f);
        }
        if (std::sscanf(line, "SimulationSolverIterations=%d", &iv) == 1)
        {
            state->params.solver_iterations = std::clamp(iv, 1, 200);
        }
        if (std::sscanf(line, "SimulationGravity=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.gravity = std::clamp(v, -1000.0f, 1000.0f);
        }

        if (std::sscanf(line, "SimulationStretchEnabled=%d", &iv) == 1)
        {
            state->params.stretch_enabled = (iv != 0);
        }
        if (std::sscanf(line, "SimulationStretchStiffness=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.stretch_stiffness = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "SimulationBaseStretchMult=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.base_stretch_mult = std::clamp(v, 0.0f, 10.0f);
        }

        if (std::sscanf(line, "SimulationBendEnabled=%d", &iv) == 1)
        {
            state->params.bend_enabled = (iv != 0);
        }
        if (std::sscanf(line, "SimulationBendStiffness=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.bend_stiffness = std::clamp(v, 0.0f, 1000.0f);
        }

        if (std::sscanf(line, "SimulationIsoBendEnabled=%d", &iv) == 1)
        {
            state->params.iso_bend_enabled = (iv != 0);
        }
        if (std::sscanf(line, "SimulationIsoBendStiffness=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.iso_bend_stiffness = std::clamp(v, 0.0f, 1000.0f);
        }

        if (std::sscanf(line, "SimulationCollisionEnabled=%d", &iv) == 1)
        {
            state->params.collision_enabled = (iv != 0);
        }
        if (std::sscanf(line, "SimulationCollisionStiffness=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.collision_stiffness = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "SimulationCollisionThickness=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.collision_thickness = std::clamp(v, 0.0f, 10.0f);
        }

        if (std::sscanf(line, "SimulationFrictionEnabled=%d", &iv) == 1)
        {
            state->params.enable_friction = (iv != 0);
        }
        if (std::sscanf(line, "SimulationFrictionMu=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.friction_mu = std::clamp(v, 0.0f, 10.0f);
        }
        if (std::sscanf(line, "SimulationDamping=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.damping = std::clamp(v, 0.0f, 0.99f);
        }

        if (std::sscanf(line, "SimulationWindForce=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.wind_force = std::clamp(v, 0.0f, 1e9f);
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
                state->params.wind_yaw_deg = std::clamp(v, -3600.0f, 3600.0f);
        }
        if (std::sscanf(line, "WindPitchDeg=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.wind_pitch_deg = std::clamp(v, -89.9f, 89.9f);
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

        if (std::sscanf(line, "WindRandomSpace=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.wind_random_space = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "WindRandomTime=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->params.wind_random_time = std::clamp(v, 0.0f, 1000.0f);
        }

        if (std::sscanf(line, "CanvaBrushRadius=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->canva_brush_radius = std::clamp(v, 0.5f, 250.0f);
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
                state->xpbd_params.dt = std::clamp(v, 1e-6f, 1.0f);
        }
        if (std::sscanf(line, "XpbdDurationSeconds=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.duration_seconds = std::clamp(v, 0.0f, 3600.0f);
        }
        if (std::sscanf(line, "XpbdSolverIterations=%d", &iv) == 1)
        {
            state->xpbd_params.solver_iterations = std::clamp(iv, 1, 200);
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
        if (std::sscanf(line, "XpbdWindRandomSpace=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_random_space = std::clamp(v, 0.0f, 1000.0f);
        }
        if (std::sscanf(line, "XpbdWindRandomTime=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->xpbd_params.wind_random_time = std::clamp(v, 0.0f, 1000.0f);
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
        buf->appendf("ClothToonShader=%d\n", state->cloth_toon_shader ? 1 : 0);
        buf->appendf("SimulationPreview=%d\n", state->controls.preview ? 1 : 0);
        buf->appendf("SmoothSequence=%d\n", state->controls.smooth_sequence ? 1 : 0);
        buf->appendf("SmoothSequenceLength=%.6f\n", state->controls.smooth_sequence_length);

        buf->appendf("SimulationDt=%.9f\n", state->params.dt);
        buf->appendf("SimulationDurationSeconds=%.9f\n", state->params.duration_seconds);
        buf->appendf("SimulationSolverIterations=%d\n", state->params.solver_iterations);
        buf->appendf("SimulationGravity=%.9f\n", state->params.gravity);

        buf->appendf("SimulationStretchEnabled=%d\n", state->params.stretch_enabled ? 1 : 0);
        buf->appendf("SimulationStretchStiffness=%.9f\n", state->params.stretch_stiffness);
        buf->appendf("SimulationBaseStretchMult=%.9f\n", state->params.base_stretch_mult);

        buf->appendf("SimulationBendEnabled=%d\n", state->params.bend_enabled ? 1 : 0);
        buf->appendf("SimulationBendStiffness=%.9f\n", state->params.bend_stiffness);

        buf->appendf("SimulationIsoBendEnabled=%d\n", state->params.iso_bend_enabled ? 1 : 0);
        buf->appendf("SimulationIsoBendStiffness=%.9f\n", state->params.iso_bend_stiffness);

        buf->appendf("SimulationCollisionEnabled=%d\n", state->params.collision_enabled ? 1 : 0);
        buf->appendf("SimulationCollisionStiffness=%.9f\n", state->params.collision_stiffness);
        buf->appendf("SimulationCollisionThickness=%.9f\n", state->params.collision_thickness);

        buf->appendf("SimulationFrictionEnabled=%d\n", state->params.enable_friction ? 1 : 0);
        buf->appendf("SimulationFrictionMu=%.9f\n", state->params.friction_mu);
        buf->appendf("SimulationDamping=%.9f\n", state->params.damping);

        buf->appendf("SimulationWindForce=%.9f\n", state->params.wind_force);

        buf->appendf("SimulationMaxFrames=%d\n", state->controls.max_frames);
        buf->appendf("SimulationFps=%.9f\n", state->controls.fps);

        buf->appendf("WindYawDeg=%.6f\n", state->params.wind_yaw_deg);
        buf->appendf("WindPitchDeg=%.6f\n", state->params.wind_pitch_deg);
        buf->appendf("LightPosX=%.6f\n", state->light_position[0]);
        buf->appendf("LightPosY=%.6f\n", state->light_position[1]);
        buf->appendf("LightPosZ=%.6f\n", state->light_position[2]);
        buf->appendf("WindRandomSpace=%.6f\n", state->params.wind_random_space);
        buf->appendf("WindRandomTime=%.6f\n", state->params.wind_random_time);

        buf->appendf("CanvaBrushRadius=%.6f\n", state->canva_brush_radius);
        buf->appendf("CanvaBrushColor=%.6f,%.6f,%.6f,%.6f\n",
            state->canva_brush_color[0], state->canva_brush_color[1], state->canva_brush_color[2], state->canva_brush_color[3]);

        buf->appendf("XpbdDt=%.9f\n", state->xpbd_params.dt);
        buf->appendf("XpbdDurationSeconds=%.9f\n", state->xpbd_params.duration_seconds);
        buf->appendf("XpbdSolverIterations=%d\n", state->xpbd_params.solver_iterations);
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
        buf->appendf("XpbdWindRandomSpace=%.6f\n", state->xpbd_params.wind_random_space);
        buf->appendf("XpbdWindRandomTime=%.6f\n", state->xpbd_params.wind_random_time);
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
