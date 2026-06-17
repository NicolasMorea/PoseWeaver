#pragma once

#include "app/AppState.h"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace poseweaver
{
    class ModelInferenceService
    {
    public:
        explicit ModelInferenceService(std::string label = "Model");
        ~ModelInferenceService();

        ModelInferenceService(const ModelInferenceService&) = delete;
        ModelInferenceService& operator=(const ModelInferenceService&) = delete;

        void start(
            const std::filesystem::path& model_dir,
            const std::filesystem::path& worker_script = {});
        bool request_inference(
            std::uint64_t request_id,
            const std::filesystem::path& sketch_path,
            const std::filesystem::path& output_path,
            std::string* out_error);
        void drain_events(AppState& state, ModelInferenceUiState& ui_state);
        void stop();

    private:
        enum class EventKind
        {
            Log,
            Loading,
            Ready,
            Busy,
            Result,
            Error,
            Stopped,
        };

        struct Event
        {
            EventKind kind = EventKind::Log;
            std::uint64_t request_id = 0;
            bool ok = false;
            std::string message;
            std::string output_path;
        };

        void push_event(Event event);
        void reader_loop();
        bool send_line(const std::string& line);

        static std::string json_escape(const std::string& s);
        static std::string json_string_field(const std::string& line, const char* key);
        static std::uint64_t json_u64_field(const std::string& line, const char* key);
        static bool json_bool_field(const std::string& line, const char* key, bool fallback);
        static Event parse_worker_line(const std::string& line);

        std::mutex m_event_mutex;
        std::vector<Event> m_events;

        std::mutex m_write_mutex;
        std::thread m_reader;
        std::atomic_bool m_running{ false };
        std::string m_label;

    #if defined(_WIN32)
        bool m_windows_unsupported_started = false;
    #else
        int m_child_pid = -1;
        int m_child_stdin = -1;
        int m_child_stdout = -1;
    #endif
    };
}
