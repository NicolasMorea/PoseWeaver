#include "app/AppConfig.h"
#include "app/AppState.h"
#include "core/Assets.h"
#include "core/Paths.h"
#include "render/Renderer.h"
#include "ui/Ui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <glad/glad.h>
#include <backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

namespace
{
    struct GlfwAppContext
    {
        poseweaver::Renderer* renderer = nullptr;
        poseweaver::AppState* state = nullptr;
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

    static bool g_forward_viewer_mouse = false;

    static void pw_glfw_cursor_pos_callback(GLFWwindow* window, double x, double y)
    {
        auto* app = app_from(window);
        if (!app || !app->renderer || !app->state)
            return;

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
            || near_light || near_wind);

        if (action == GLFW_PRESS && in_view && !block_viewer)
            g_forward_viewer_mouse = true;

        if (g_forward_viewer_mouse)
            app->renderer->on_mouse_button(button, action, mods);

        if (action == GLFW_RELEASE)
            g_forward_viewer_mouse = false;
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

            if (key == GLFW_KEY_B && action == GLFW_PRESS)
            {
                state.show_smpl_character = !state.show_smpl_character;
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

        // Reserve 'B' for toggling SMPL visibility in the key callback.
        if (codepoint == (unsigned int)'b' || codepoint == (unsigned int)'B')
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

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        return 1;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, false);

    const char* glsl_version = "#version 330";
    ImGui_ImplOpenGL3_Init(glsl_version);

    poseweaver::AppState state;
    poseweaver::AssetIndex assets(poseweaver::assets_dir());
    poseweaver::Ui ui;
    poseweaver::Renderer renderer;

    poseweaver::install_imgui_state_settings_handler(state);

    GlfwAppContext app;
    app.renderer = &renderer;
    app.state = &state;

    renderer.init(window);
    glfwSetWindowUserPointer(window, &app);

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

    double last_time = glfwGetTime();
    double play_accum = 0.0;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

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

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ui.draw(state, assets, renderer, poseweaver::UiFrameInfo{ fb_w, fb_h });

        ImGui::Render();

        renderer.render(state, fb_w, fb_h);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    renderer.shutdown();

    poseweaver::save_imgui_settings_to_disk();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
