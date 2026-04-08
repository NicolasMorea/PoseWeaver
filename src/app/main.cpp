#include "app/AppState.h"
#include "core/Assets.h"
#include "core/Paths.h"
#include "render/Renderer.h"
#include "ui/Ui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <imgui.h>
#include <imgui_internal.h>
#include <backends/imgui_impl_glfw.h>
#include <glad/glad.h>
#include <backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

namespace
{
    static void* pw_imgui_settings_read_open(ImGuiContext*, ImGuiSettingsHandler* handler, const char* name)
    {
        if (!handler || !name)
            return nullptr;
        // We only store a single block: [PoseWeaver][State]
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
        if (std::sscanf(line, "GarmentMinClothSkinDistance=%f", &v) == 1)
        {
            if (std::isfinite(v))
                state->garment_min_cloth_skin_distance = std::clamp(v, 0.0f, 0.02f);
        }
    }

    static void pw_imgui_settings_write_all(ImGuiContext*, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
    {
        if (!handler || !handler->UserData || !buf)
            return;

        auto* state = static_cast<poseweaver::AppState*>(handler->UserData);

        buf->appendf("[PoseWeaver][State]\n");
        buf->appendf("GarmentMinClothSkinDistance=%.6f\n", state->garment_min_cloth_skin_distance);
        buf->appendf("\n");
    }

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

        if (action == GLFW_PRESS && in_view)
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
        if (cursor_in_cloth_view(window, app->state->cloth_view, x, y))
            app->renderer->on_scroll(xoffset, yoffset);
    }

    static void pw_glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

        if (ImGui::GetIO().WantCaptureKeyboard)
            return;

        if (auto* app = app_from(window); app && app->renderer)
            app->renderer->on_key(key, scancode, action, mods);
    }

    static void pw_glfw_char_callback(GLFWwindow* window, unsigned int codepoint)
    {
        ImGui_ImplGlfw_CharCallback(window, codepoint);
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

    // Init ImGui
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

    // Serialize some states into imgui.ini.
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

    GlfwAppContext app;
    app.renderer = &renderer;
    app.state = &state;

    renderer.init(window);
    glfwSetWindowUserPointer(window, &app);

    glfwSetCursorPosCallback(window, pw_glfw_cursor_pos_callback);
    glfwSetMouseButtonCallback(window, pw_glfw_mouse_button_callback);
    glfwSetScrollCallback(window, pw_glfw_scroll_callback);
    glfwSetKeyCallback(window, pw_glfw_key_callback);
    glfwSetCharCallback(window, pw_glfw_char_callback);

    // debug info
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

        // time stepping for the simulation (if playing)
        const double now = glfwGetTime();
        const double dt = now - last_time;
        last_time = now;

        if (state.controls.playing)
        {
            play_accum += dt;
            const double frame_dt = 1.0 / std::max(1.0f, state.controls.fps);
            while (play_accum >= frame_dt)
            {
                play_accum -= frame_dt;
                state.controls.current_frame = (state.controls.current_frame + 1) % std::max(1, state.controls.max_frames);
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

        // Render ImGui on top
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    renderer.shutdown();

    // Ensure custom settings are written out on exit.
    if (const char* ini = ImGui::GetIO().IniFilename)
        ImGui::SaveIniSettingsToDisk(ini);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
