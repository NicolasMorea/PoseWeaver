#pragma once

#include <memory>
#include <string>

namespace poseweaver
{
    class AssetIndex;
    class Renderer;
    struct AppState;

    class AutomationServer
    {
    public:
        AutomationServer();
        ~AutomationServer();

        AutomationServer(const AutomationServer&) = delete;
        AutomationServer& operator=(const AutomationServer&) = delete;

        bool start(std::string* out_error = nullptr);
        void pump(AppState& state, AssetIndex& assets, Renderer& renderer);
        void stop();

        const std::string& socket_path() const;
        bool running() const;
        bool shutdown_requested() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
