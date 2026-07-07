#include "app/AppConfig.h"
#include "app/AppState.h"
#include "core/Assets.h"
#include "core/Paths.h"
#include "render/Renderer.h"
#include "services/ModelInferenceService.h"
#include "ui/Ui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <glad/glad.h>
#include <backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

namespace
{
    static void force_native_cursor_visible(GLFWwindow* window)
    {
        if (window)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    static void show_cursor_on_launch(GLFWwindow* window)
    {
        if (!window)
            return;

        glfwSetCursor(window, nullptr);
        force_native_cursor_visible(window);
    }

    struct NativeCursorGuard
    {
        GLFWwindow* window = nullptr;
        GLFWcursor* cursors[ImGuiMouseCursor_COUNT] = {};

        void init(GLFWwindow* w)
        {
            window = w;
            apply(ImGuiMouseCursor_Arrow);
        }

        void apply(ImGuiMouseCursor requested)
        {
            if (!window)
                return;

            if (requested == ImGuiMouseCursor_None || requested < 0 || requested >= ImGuiMouseCursor_COUNT)
                requested = ImGuiMouseCursor_Arrow;

            GLFWcursor* cursor = cursors[requested] ? cursors[requested] : cursors[ImGuiMouseCursor_Arrow];
            if (cursor)
                glfwSetCursor(window, cursor);
            else
                glfwSetCursor(window, nullptr);
            force_native_cursor_visible(window);
        }

        void apply_imgui_cursor()
        {
            ImGui::GetIO().MouseDrawCursor = false;
            apply(ImGui::GetMouseCursor());
        }

        void shutdown()
        {
            if (window)
            {
                glfwSetCursor(window, nullptr);
                force_native_cursor_visible(window);
            }
            for (GLFWcursor*& cursor : cursors)
            {
                if (cursor)
                {
                    glfwDestroyCursor(cursor);
                    cursor = nullptr;
                }
            }
            window = nullptr;
        }
    };

    struct GlfwAppContext
    {
        poseweaver::Renderer* renderer = nullptr;
        poseweaver::AppState* state = nullptr;
        NativeCursorGuard* cursor_guard = nullptr;
    };

    static void pw_glfw_error_callback(int error, const char* description)
    {
        std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
    }

    static GlfwAppContext* app_from(GLFWwindow* window)
    {
        return static_cast<GlfwAppContext*>(glfwGetWindowUserPointer(window));
    }

    static bool cursor_in_cloth_view(GLFWwindow* window, const poseweaver::ViewportRect& r, double x, double y)
    {
        if (!r.valid)
            return false;

        int win_w = 0, win_h = 0;
        int fb_w = 0, fb_h = 0;
        glfwGetWindowSize(window, &win_w, &win_h);
        glfwGetFramebufferSize(window, &fb_w, &fb_h);

        const double sx = (win_w > 0) ? (double)fb_w / (double)win_w : 1.0;
        const double sy = (win_h > 0) ? (double)fb_h / (double)win_h : 1.0;

        const int x_fb = (int)(x * sx);
        const int y_fb_top = (int)(y * sy);
        const int y_fb = std::clamp(fb_h - 1 - y_fb_top, 0, std::max(0, fb_h - 1));

        return (x_fb >= r.x && x_fb < (r.x + r.width) && y_fb >= r.y && y_fb < (r.y + r.height));
    }

    static bool cursor_framebuffer_pos(GLFWwindow* window, double x, double y, int& out_x, int& out_y)
    {
        int win_w = 0, win_h = 0;
        int fb_w = 0, fb_h = 0;
        glfwGetWindowSize(window, &win_w, &win_h);
        glfwGetFramebufferSize(window, &fb_w, &fb_h);

        if (fb_w <= 0 || fb_h <= 0)
            return false;

        const double sx = (win_w > 0) ? (double)fb_w / (double)win_w : 1.0;
        const double sy = (win_h > 0) ? (double)fb_h / (double)win_h : 1.0;

        out_x = std::clamp((int)(x * sx), 0, std::max(0, fb_w - 1));
        const int y_fb_top = (int)(y * sy);
        out_y = std::clamp(fb_h - 1 - y_fb_top, 0, std::max(0, fb_h - 1));
        return true;
    }

    static bool g_forward_viewer_mouse = false;
    static bool g_constraint_dragging = false;
    static int g_constraint_drag_start_x = 0;
    static int g_constraint_drag_start_y = 0;

    static void pw_glfw_window_focus_callback(GLFWwindow* window, int focused)
    {
        ImGui_ImplGlfw_WindowFocusCallback(window, focused);

        if (!focused)
            return;

        force_native_cursor_visible(window);
        if (auto* app = app_from(window); app && app->cursor_guard)
            app->cursor_guard->apply(ImGuiMouseCursor_Arrow);
    }

    static void pw_glfw_cursor_enter_callback(GLFWwindow* window, int entered)
    {
        ImGui_ImplGlfw_CursorEnterCallback(window, entered);

        if (!entered)
            return;

        force_native_cursor_visible(window);
        if (auto* app = app_from(window); app && app->cursor_guard)
            app->cursor_guard->apply_imgui_cursor();
    }

    static void pw_glfw_cursor_pos_callback(GLFWwindow* window, double x, double y)
    {
        auto* app = app_from(window);
        if (!app || !app->renderer || !app->state)
            return;

        if (g_constraint_dragging && app->state->cloth_constraints_edit_enabled)
        {
            app->state->cloth_constraint_drag_end_screen = { (float)x, (float)y };
        }

        if (app->state->scene_view_gizmo_using)
            return;

        if (g_forward_viewer_mouse || cursor_in_cloth_view(window, app->state->cloth_view, x, y))
            app->renderer->on_cursor_pos(x, y);
    }

    static void pw_glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
    {
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

        auto* app = app_from(window);
        if (!app || !app->renderer || !app->state)
            return;

        double x = 0.0, y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        const bool in_view = cursor_in_cloth_view(window, app->state->cloth_view, x, y);

        const auto dist2_to = [&](const std::array<float, 2>& p)
        {
            const double dx = x - (double)p[0];
            const double dy = y - (double)p[1];
            return dx * dx + dy * dy;
        };

        constexpr double kPickRadiusPx = 90.0;
        const double pick_r2 = kPickRadiusPx * kPickRadiusPx;
        const bool near_light = app->state->scene_view_light_marker_valid
            && (dist2_to(app->state->scene_view_light_marker_screen) < pick_r2);
        const bool near_wind = app->state->scene_view_wind_marker_valid
            && (dist2_to(app->state->scene_view_wind_marker_screen) < pick_r2);

        const bool block_viewer = (app->state->scene_view_gizmo_over || app->state->scene_view_gizmo_using
            || near_light || near_wind || app->state->cloth_constraints_edit_enabled);

        if (button == GLFW_MOUSE_BUTTON_LEFT && (in_view || g_constraint_dragging)
            && app->state->cloth_constraints_edit_enabled
            && !(app->state->scene_view_gizmo_over || app->state->scene_view_gizmo_using || near_light || near_wind))
        {
            int x_fb = 0;
            int y_fb = 0;
            if (action == GLFW_PRESS && cursor_framebuffer_pos(window, x, y, x_fb, y_fb))
            {
                g_constraint_dragging = true;
                g_constraint_drag_start_x = x_fb;
                g_constraint_drag_start_y = y_fb;
                app->state->cloth_constraint_drag_active = true;
                app->state->cloth_constraint_drag_start_screen = { (float)x, (float)y };
                app->state->cloth_constraint_drag_end_screen = { (float)x, (float)y };
                app->state->cloth_constraint_click_mods = mods;
            }
            else if (action == GLFW_RELEASE && g_constraint_dragging && cursor_framebuffer_pos(window, x, y, x_fb, y_fb))
            {
                app->state->cloth_constraint_selection_x0 = g_constraint_drag_start_x;
                app->state->cloth_constraint_selection_y0 = g_constraint_drag_start_y;
                app->state->cloth_constraint_selection_x1 = x_fb;
                app->state->cloth_constraint_selection_y1 = y_fb;
                app->state->cloth_constraint_click_mods = mods;
                app->state->cloth_constraint_click_request_id++;
            }
        }

        if (action == GLFW_PRESS && in_view && !block_viewer)
            g_forward_viewer_mouse = true;

        if (g_forward_viewer_mouse)
            app->renderer->on_mouse_button(button, action, mods);

        if (action == GLFW_RELEASE)
        {
            g_forward_viewer_mouse = false;
            if (button == GLFW_MOUSE_BUTTON_LEFT)
            {
                g_constraint_dragging = false;
                app->state->cloth_constraint_drag_active = false;
            }
        }
    }

    static void pw_glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);

        auto* app = app_from(window);
        if (!app || !app->renderer || !app->state)
            return;

        double x = 0.0, y = 0.0;
        glfwGetCursorPos(window, &x, &y);

        const auto dist2_to = [&](const std::array<float, 2>& p)
        {
            const double dx = x - (double)p[0];
            const double dy = y - (double)p[1];
            return dx * dx + dy * dy;
        };
        constexpr double kPickRadiusPx = 90.0;
        const double pick_r2 = kPickRadiusPx * kPickRadiusPx;
        const bool near_light = app->state->scene_view_light_marker_valid
            && (dist2_to(app->state->scene_view_light_marker_screen) < pick_r2);
        const bool near_wind = app->state->scene_view_wind_marker_valid
            && (dist2_to(app->state->scene_view_wind_marker_screen) < pick_r2);

        if (cursor_in_cloth_view(window, app->state->cloth_view, x, y)
            && !(app->state->scene_view_gizmo_over || app->state->scene_view_gizmo_using || near_light || near_wind))
            app->renderer->on_scroll(xoffset, yoffset);
    }

    static void pw_glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

        // Simulation shortcuts work even when ImGui wants the keyboard, except when typing.
        if (auto* app = app_from(window); app && app->state)
        {
            auto& state = *app->state;
            if (action == GLFW_PRESS && !ImGui::GetIO().WantTextInput)
            {
                // Canva shortcuts: handle in GLFW callback for reliability.
                if (state.canva_tab_active && (mods & GLFW_MOD_CONTROL) && key == GLFW_KEY_Z)
                {
                    state.canva_undo_request_id++;
                    poseweaver::console_add_line(state, "Canva: Ctrl+Z detected (GLFW)");
                    return;
                }

                const bool plain_key = (mods & (GLFW_MOD_CONTROL | GLFW_MOD_ALT | GLFW_MOD_SUPER)) == 0;
                if (plain_key && key == GLFW_KEY_A)
                {
                    state.scene_view_snap_to_target_request_id++;
                    return;
                }

                if (plain_key && key == GLFW_KEY_X)
                {
                    state.show_smpl_character = !state.show_smpl_character;
                    return;
                }

                if (plain_key && key == GLFW_KEY_W)
                {
                    state.show_sketch2rig_rig = !state.show_sketch2rig_rig;
                    return;
                }

                if (key == GLFW_KEY_SPACE && state.controls.simulating)
                {
                    state.cloth_cancel_simulation_request_id++;
                    state.controls.playing = false;
                    return;
                }

                if ((key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) && !ImGui::IsAnyItemActive())
                {
                    state.cloth_simulate_request_id++;
                    state.controls.simulating = true;
                    state.controls.playing = false;
                    state.controls.current_frame = 0;
                    return;
                }
            }
        }

        if (ImGui::GetIO().WantCaptureKeyboard)
            return;

        if (auto* app = app_from(window); app && app->state)
        {
            auto& state = *app->state;

            if (key == GLFW_KEY_C && action == GLFW_PRESS)
            {
                state.soft_cursor_enabled = !state.soft_cursor_enabled;
                return;
            }

            const bool have_frames = (state.controls.frame_count > 0);
            const bool can_control_playback = have_frames && !state.controls.simulating;
            const int frame_max = have_frames ? (state.controls.frame_count - 1) : 0;

            if (can_control_playback && (action == GLFW_PRESS || action == GLFW_REPEAT))
            {
                if (key == GLFW_KEY_SPACE)
                {
                    state.controls.playing = !state.controls.playing;
                    return;
                }
                if (key == GLFW_KEY_LEFT)
                {
                    state.controls.playing = false;
                    if (mods & GLFW_MOD_CONTROL)
                        state.controls.current_frame = 0;
                    else
                        state.controls.current_frame = std::max(0, state.controls.current_frame - 1);
                    return;
                }
                if (key == GLFW_KEY_RIGHT)
                {
                    state.controls.playing = false;
                    if (mods & GLFW_MOD_CONTROL)
                        state.controls.current_frame = std::max(0, frame_max);
                    else
                        state.controls.current_frame = std::min(std::max(0, frame_max), state.controls.current_frame + 1);
                    return;
                }
            }

        }

        if (auto* app = app_from(window); app && app->renderer)
            app->renderer->on_key(key, scancode, action, mods);
    }

    static void pw_glfw_char_mods_callback(GLFWwindow* window, unsigned int codepoint, int mods)
    {
        ImGui_ImplGlfw_CharCallback(window, codepoint);

        // libigl's default shortcuts are fed via the GLFW character callback.
        if (ImGui::GetIO().WantCaptureKeyboard)
            return;

        // Reserve scene toolbar shortcuts handled in the key callback.
        if (codepoint == (unsigned int)'a' || codepoint == (unsigned int)'A'
            || codepoint == (unsigned int)'w' || codepoint == (unsigned int)'W'
            || codepoint == (unsigned int)'x' || codepoint == (unsigned int)'X')
            return;
        if (auto* app = app_from(window); app && app->renderer)
            app->renderer->on_char(codepoint, mods);
    }
}

int main(int, char**)
{
    glfwSetErrorCallback(pw_glfw_error_callback);
    if (!glfwInit())
        return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1600, 900, "PoseWeaver", nullptr, nullptr);
    if (window == nullptr)
        return 1;

    show_cursor_on_launch(window);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        return 1;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    ImGui::GetIO().MouseDrawCursor = false;

    ImGui_ImplGlfw_InitForOpenGL(window, false);

    const char* glsl_version = "#version 330";
    ImGui_ImplOpenGL3_Init(glsl_version);

    poseweaver::AppState state;
    poseweaver::AssetIndex assets(poseweaver::assets_dir());
    poseweaver::Ui ui;
    poseweaver::Renderer renderer;
    poseweaver::ModelInferenceService sketch2rig("Sketch2Rig");
    poseweaver::ModelInferenceService sketch2betas("Sketch2Betas");
    poseweaver::ModelInferenceService sketch2uv("Sketch2UV");
    poseweaver::ModelInferenceService sketch2iso("Sketch2ISO");
    poseweaver::ModelInferenceService sketch2depth("Sketch2Depth");
    poseweaver::ModelInferenceService vitpose("VitPose");
    NativeCursorGuard cursor_guard;
    cursor_guard.init(window);
    state.sketch2rig.service = &sketch2rig;
    state.sketch2betas.service = &sketch2betas;
    state.sketch2uv.service = &sketch2uv;
    state.sketch2iso.service = &sketch2iso;
    state.sketch2depth.service = &sketch2depth;
    state.vitpose.service = &vitpose;

    poseweaver::install_imgui_state_settings_handler(state);

    GlfwAppContext app;
    app.renderer = &renderer;
    app.state = &state;
    app.cursor_guard = &cursor_guard;

    renderer.init(window);
    cursor_guard.apply(ImGuiMouseCursor_Arrow);
    glfwSetWindowUserPointer(window, &app);

    glfwSetWindowFocusCallback(window, pw_glfw_window_focus_callback);
    glfwSetCursorEnterCallback(window, pw_glfw_cursor_enter_callback);
    glfwSetCursorPosCallback(window, pw_glfw_cursor_pos_callback);
    glfwSetMouseButtonCallback(window, pw_glfw_mouse_button_callback);
    glfwSetScrollCallback(window, pw_glfw_scroll_callback);
    glfwSetKeyCallback(window, pw_glfw_key_callback);
    glfwSetCharModsCallback(window, pw_glfw_char_mods_callback);

    if (const char* version = (const char*)glGetString(GL_VERSION))
        poseweaver::console_add_line(state, std::string("GL_VERSION: ") + version);
    if (const char* renderer_str = (const char*)glGetString(GL_RENDERER))
        poseweaver::console_add_line(state, std::string("GL_RENDERER: ") + renderer_str);
    if (const char* vendor = (const char*)glGetString(GL_VENDOR))
        poseweaver::console_add_line(state, std::string("GL_VENDOR: ") + vendor);
    poseweaver::console_add_line(state, "PoseWeaver started");
    sketch2rig.start(
        poseweaver::assets_dir() / "Sketch2Rig",
        std::filesystem::path(POSEWEAVER_SKETCH2RIG_WORKER));
    sketch2betas.start(
        poseweaver::assets_dir() / "Sketch2Betas",
        std::filesystem::path(POSEWEAVER_SKETCH2BETAS_WORKER));
    sketch2uv.start(
        poseweaver::assets_dir() / "Sketch2Uv",
        std::filesystem::path(POSEWEAVER_SKETCH2UV_WORKER));
    sketch2iso.start(
        poseweaver::assets_dir() / "Sketch2Iso",
        std::filesystem::path(POSEWEAVER_SKETCH2ISO_WORKER));
    sketch2depth.start(
        poseweaver::assets_dir() / "Sketch2Depth",
        std::filesystem::path(POSEWEAVER_SKETCH2DEPTH_WORKER));
    vitpose.start(
        poseweaver::assets_dir() / "VitPose",
        std::filesystem::path(POSEWEAVER_VITPOSE_WORKER),
        "vitpose_base_coco_aic_mpii.pth");

    double last_time = glfwGetTime();
    double play_accum = 0.0;
    std::uint64_t last_asset_refresh_request_id = state.asset_refresh_request_id;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        force_native_cursor_visible(window);

        int fb_w = 0, fb_h = 0;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);

        renderer.on_framebuffer_resize(fb_w, fb_h);

        const double now = glfwGetTime();
        const double dt = now - last_time;
        last_time = now;

        if (state.controls.playing)
        {
            play_accum += dt;
            const double frame_dt = 1.0 / std::max(1.0f, state.controls.fps);
            const int frame_count = (state.controls.frame_count > 0) ? state.controls.frame_count : state.controls.max_frames;
            while (play_accum >= frame_dt)
            {
                play_accum -= frame_dt;
                state.controls.current_frame = (state.controls.current_frame + 1) % std::max(1, frame_count);
            }
        }
        else
        {
            play_accum = 0.0;
        }

        sketch2rig.drain_events(state, state.sketch2rig);
        sketch2betas.drain_events(state, state.sketch2betas);
        sketch2uv.drain_events(state, state.sketch2uv);
        sketch2iso.drain_events(state, state.sketch2iso);
        sketch2depth.drain_events(state, state.sketch2depth);
        vitpose.drain_events(state, state.vitpose);

        if (state.asset_refresh_request_id != last_asset_refresh_request_id)
        {
            assets.refresh();
            state.selection.poses = assets.poses();
            if (!state.pending_pose_selection.empty())
            {
                const auto it = std::find(
                    state.selection.poses.begin(),
                    state.selection.poses.end(),
                    state.pending_pose_selection);
                if (it != state.selection.poses.end())
                    state.selection.pose_index = (int)std::distance(state.selection.poses.begin(), it);
                state.pending_pose_selection.clear();
            }
            last_asset_refresh_request_id = state.asset_refresh_request_id;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        force_native_cursor_visible(window);

        ui.draw(state, assets, renderer, poseweaver::UiFrameInfo{ fb_w, fb_h });
        cursor_guard.apply_imgui_cursor();

        if (state.cursor_restore_request)
        {
            state.cursor_restore_request = false;
            // Cycle HIDDEN→NORMAL to force GLFW to re-issue XDefineCursor/SetCursor
            // even when it believes the mode is already NORMAL (no-op otherwise).
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            cursor_guard.apply(ImGuiMouseCursor_Arrow);
        }

        ImGui::Render();

        renderer.render(state, fb_w, fb_h);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    renderer.shutdown();
    cursor_guard.apply(ImGuiMouseCursor_Arrow);
    sketch2rig.stop();
    sketch2betas.stop();
    sketch2uv.stop();
    sketch2iso.stop();
    sketch2depth.stop();
    vitpose.stop();

    poseweaver::save_imgui_settings_to_disk();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    cursor_guard.shutdown();
    ImGui::DestroyContext();

    force_native_cursor_visible(window);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
