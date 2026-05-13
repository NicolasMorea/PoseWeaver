#include "sketch2rig/Sketch2RigService.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <utility>

#if !defined(_WIN32)
#include <csignal>
#include <cstdio>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace poseweaver
{
    namespace
    {
        constexpr const char* kWorkerPrefix = "POSEWEAVER_JSON ";

        std::string sketch_status_label(Sketch2RigStatus status)
        {
            switch (status)
            {
            case Sketch2RigStatus::NotStarted: return "Model not started";
            case Sketch2RigStatus::Loading: return "Model loading";
            case Sketch2RigStatus::Ready: return "Model loaded";
            case Sketch2RigStatus::Busy: return "Processing";
            case Sketch2RigStatus::Error: return "Model error";
            case Sketch2RigStatus::Stopped: return "Model stopped";
            }
            return "Unknown";
        }
    }

    Sketch2RigService::Sketch2RigService(std::string label)
        : m_label(std::move(label))
    {
    }

    Sketch2RigService::~Sketch2RigService()
    {
        stop();
    }

    void Sketch2RigService::start(
        const std::filesystem::path& model_dir,
        const std::filesystem::path& worker_script)
    {
        if (m_running.load())
            return;

    #if defined(_WIN32)
        (void)model_dir;
        (void)worker_script;
        if (!m_windows_unsupported_started)
        {
            m_windows_unsupported_started = true;
            push_event(Event{ EventKind::Error, 0, false,
                m_label + " background worker is not implemented for Windows builds yet", "" });
        }
    #else
        std::filesystem::path script = worker_script;
        if (script.empty())
        {
    #ifdef POSEWEAVER_SKETCH2RIG_WORKER
            script = std::filesystem::path(POSEWEAVER_SKETCH2RIG_WORKER);
    #else
            script = model_dir / "inference_worker.py";
    #endif
        }
        const std::filesystem::path checkpoint = model_dir / "best.pth";

        if (!std::filesystem::exists(script))
        {
            push_event(Event{ EventKind::Error, 0, false, "Missing " + m_label + " worker: " + script.string(), "" });
            return;
        }
        if (!std::filesystem::exists(checkpoint))
        {
            push_event(Event{ EventKind::Error, 0, false, "Missing " + m_label + " checkpoint: " + checkpoint.string(), "" });
            return;
        }

        int stdin_pipe[2] = { -1, -1 };
        int stdout_pipe[2] = { -1, -1 };
        if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0)
        {
            push_event(Event{ EventKind::Error, 0, false, "Failed to create pipes for " + m_label + " worker", "" });
            if (stdin_pipe[0] >= 0) close(stdin_pipe[0]);
            if (stdin_pipe[1] >= 0) close(stdin_pipe[1]);
            if (stdout_pipe[0] >= 0) close(stdout_pipe[0]);
            if (stdout_pipe[1] >= 0) close(stdout_pipe[1]);
            return;
        }

        const char* env_python = std::getenv("POSEWEAVER_PYTHON");
        const char* env_conda_env = std::getenv("POSEWEAVER_CONDA_ENV");
        const char* env_conda_exe = std::getenv("CONDA_EXE");

        const std::string python = (env_python && env_python[0] != '\0') ? env_python : "";
        const std::string conda_env = (env_conda_env && env_conda_env[0] != '\0') ? env_conda_env : "dev";
        std::string conda_exe = (env_conda_exe && env_conda_exe[0] != '\0') ? env_conda_exe : "conda";
        if (!(env_conda_exe && env_conda_exe[0] != '\0'))
        {
            if (const char* home = std::getenv("HOME"); home && home[0] != '\0')
            {
                const std::filesystem::path home_path(home);
                const std::filesystem::path candidates[] = {
                    home_path / "miniconda3" / "bin" / "conda",
                    home_path / "miniconda3" / "condabin" / "conda",
                    home_path / "anaconda3" / "bin" / "conda",
                    home_path / "anaconda3" / "condabin" / "conda",
                    home_path / "miniforge3" / "bin" / "conda",
                    home_path / "mambaforge" / "bin" / "conda",
                };
                for (const auto& candidate : candidates)
                {
                    if (std::filesystem::exists(candidate))
                    {
                        conda_exe = candidate.string();
                        break;
                    }
                }
            }
        }

        const pid_t pid = fork();
        if (pid < 0)
        {
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            push_event(Event{ EventKind::Error, 0, false, "Failed to fork " + m_label + " worker", "" });
            return;
        }

        if (pid == 0)
        {
            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stdout_pipe[1], STDERR_FILENO);

            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);

            if (!python.empty())
            {
                execlp(
                    python.c_str(),
                    python.c_str(),
                    script.string().c_str(),
                    "--checkpoint",
                    checkpoint.string().c_str(),
                    nullptr);
            }
            else
            {
                execlp(
                    conda_exe.c_str(),
                    conda_exe.c_str(),
                    "run",
                    "--no-capture-output",
                    "-n",
                    conda_env.c_str(),
                    "python",
                    script.string().c_str(),
                    "--checkpoint",
                    checkpoint.string().c_str(),
                    nullptr);
            }
            _exit(127);
        }

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        m_child_pid = (int)pid;
        m_child_stdin = stdin_pipe[1];
        m_child_stdout = stdout_pipe[0];
        m_running.store(true);

        push_event(Event{ EventKind::Loading, 0, false,
            !python.empty()
                ? ("Starting " + m_label + " model worker with " + python + "...")
                : ("Starting " + m_label + " model worker with conda env '" + conda_env + "'..."),
            "" });
        m_reader = std::thread(&Sketch2RigService::reader_loop, this);
    #endif
    }

    bool Sketch2RigService::request_inference(
        std::uint64_t request_id,
        const std::filesystem::path& sketch_path,
        const std::filesystem::path& output_path,
        std::string* out_error)
    {
        if (!m_running.load())
        {
            if (out_error)
                *out_error = m_label + " worker is not running";
            return false;
        }

        std::error_code ec;
        std::filesystem::create_directories(output_path.parent_path(), ec);
        if (ec)
        {
            if (out_error)
                *out_error = "Failed to create output directory: " + ec.message();
            return false;
        }

        std::ostringstream oss;
        oss << "{\"cmd\":\"infer\",\"request_id\":" << request_id
            << ",\"sketch_path\":\"" << json_escape(sketch_path.string())
            << "\",\"output_path\":\"" << json_escape(output_path.string()) << "\"}\n";

        if (!send_line(oss.str()))
        {
            if (out_error)
                *out_error = "Failed to send inference request to " + m_label + " worker";
            return false;
        }

        push_event(Event{ EventKind::Busy, request_id, false, "Processing sketch with " + m_label + "...", "" });
        return true;
    }

    void Sketch2RigService::drain_events(AppState& state)
    {
        drain_events(state, state.sketch2rig);
    }

    void Sketch2RigService::drain_events(AppState& state, Sketch2RigUiState& ui_state)
    {
        std::vector<Event> events;
        {
            std::lock_guard<std::mutex> lock(m_event_mutex);
            events.swap(m_events);
        }

        for (const Event& event : events)
        {
            switch (event.kind)
            {
            case EventKind::Log:
                if (!event.message.empty())
                    console_add_line(state, m_label + ": " + event.message);
                break;
            case EventKind::Loading:
                ui_state.status = Sketch2RigStatus::Loading;
                ui_state.message = event.message.empty() ? sketch_status_label(ui_state.status) : event.message;
                console_add_line(state, m_label + ": " + ui_state.message);
                break;
            case EventKind::Ready:
                ui_state.status = Sketch2RigStatus::Ready;
                ui_state.message = event.message.empty() ? sketch_status_label(ui_state.status) : event.message;
                console_add_line(state, m_label + ": " + ui_state.message);
                break;
            case EventKind::Busy:
                ui_state.status = Sketch2RigStatus::Busy;
                ui_state.active_request_id = event.request_id;
                ui_state.message = event.message.empty() ? sketch_status_label(ui_state.status) : event.message;
                console_add_line(state, m_label + ": " + ui_state.message);
                break;
            case EventKind::Result:
                ui_state.completed_request_id = event.request_id;
                ui_state.active_request_id = 0;
                ui_state.last_output_path = event.output_path;
                ui_state.status = event.ok ? Sketch2RigStatus::Ready : Sketch2RigStatus::Error;
                ui_state.message = event.message.empty()
                    ? (event.ok ? "Inference complete" : "Inference failed")
                    : event.message;
                console_add_line(state, m_label + ": " + ui_state.message);
                if (event.ok && !event.output_path.empty())
                    console_add_line(state, m_label + ": Output JSON: " + event.output_path);
                break;
            case EventKind::Error:
                ui_state.active_request_id = 0;
                ui_state.status = Sketch2RigStatus::Error;
                ui_state.message = event.message.empty() ? sketch_status_label(ui_state.status) : event.message;
                console_add_line(state, m_label + ": " + ui_state.message);
                break;
            case EventKind::Stopped:
                ui_state.active_request_id = 0;
                if (ui_state.status != Sketch2RigStatus::Error)
                {
                    ui_state.status = Sketch2RigStatus::Stopped;
                    ui_state.message = event.message.empty() ? sketch_status_label(ui_state.status) : event.message;
                }
                if (!event.message.empty())
                    console_add_line(state, m_label + ": " + event.message);
                break;
            }
        }
    }

    void Sketch2RigService::stop()
    {
    #if defined(_WIN32)
        m_running.store(false);
    #else
        const bool was_running = m_running.exchange(false);
        if (was_running && m_child_stdin >= 0)
            send_line("{\"cmd\":\"shutdown\"}\n");

        if (m_child_stdin >= 0)
        {
            close(m_child_stdin);
            m_child_stdin = -1;
        }

        if (m_reader.joinable())
            m_reader.join();

        if (m_child_stdout >= 0)
        {
            close(m_child_stdout);
            m_child_stdout = -1;
        }

        if (m_child_pid > 0)
        {
            int status = 0;
            if (waitpid((pid_t)m_child_pid, &status, WNOHANG) == 0)
            {
                kill((pid_t)m_child_pid, SIGTERM);
                waitpid((pid_t)m_child_pid, &status, 0);
            }
            m_child_pid = -1;
        }
    #endif
    }

    void Sketch2RigService::push_event(Event event)
    {
        std::lock_guard<std::mutex> lock(m_event_mutex);
        m_events.emplace_back(std::move(event));
    }

    void Sketch2RigService::reader_loop()
    {
    #if !defined(_WIN32)
        FILE* stream = fdopen(m_child_stdout, "r");
        if (!stream)
        {
            push_event(Event{ EventKind::Error, 0, false, "Failed to read from " + m_label + " worker", "" });
            m_running.store(false);
            return;
        }

        char buffer[8192];
        while (fgets(buffer, sizeof(buffer), stream))
        {
            std::string line(buffer);
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
                line.pop_back();

            if (line.rfind(kWorkerPrefix, 0) == 0)
                push_event(parse_worker_line(line.substr(std::strlen(kWorkerPrefix))));
            else if (!line.empty())
                push_event(Event{ EventKind::Log, 0, false, line, "" });
        }

        fclose(stream);
        m_child_stdout = -1;
        if (m_running.exchange(false))
            push_event(Event{ EventKind::Stopped, 0, false, "Worker stopped", "" });
    #endif
    }

    bool Sketch2RigService::send_line(const std::string& line)
    {
    #if defined(_WIN32)
        (void)line;
        return false;
    #else
        std::lock_guard<std::mutex> lock(m_write_mutex);
        if (m_child_stdin < 0)
            return false;

        const char* data = line.data();
        size_t remaining = line.size();
        while (remaining > 0)
        {
            const ssize_t written = write(m_child_stdin, data, remaining);
            if (written <= 0)
                return false;
            data += written;
            remaining -= (size_t)written;
        }
        return true;
    #endif
    }

    std::string Sketch2RigService::json_escape(const std::string& s)
    {
        std::string out;
        out.reserve(s.size() + 8);
        for (char c : s)
        {
            switch (c)
            {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
            }
        }
        return out;
    }

    std::string Sketch2RigService::json_string_field(const std::string& line, const char* key)
    {
        const std::string needle = std::string("\"") + key + "\":\"";
        const size_t start = line.find(needle);
        if (start == std::string::npos)
            return {};

        std::string out;
        bool escaped = false;
        for (size_t i = start + needle.size(); i < line.size(); ++i)
        {
            const char c = line[i];
            if (escaped)
            {
                switch (c)
                {
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                default: out += c; break;
                }
                escaped = false;
                continue;
            }
            if (c == '\\')
            {
                escaped = true;
                continue;
            }
            if (c == '"')
                break;
            out += c;
        }
        return out;
    }

    std::uint64_t Sketch2RigService::json_u64_field(const std::string& line, const char* key)
    {
        const std::string needle = std::string("\"") + key + "\":";
        const size_t start = line.find(needle);
        if (start == std::string::npos)
            return 0;

        size_t i = start + needle.size();
        while (i < line.size() && line[i] == ' ')
            ++i;

        std::uint64_t value = 0;
        while (i < line.size() && line[i] >= '0' && line[i] <= '9')
        {
            value = value * 10u + (std::uint64_t)(line[i] - '0');
            ++i;
        }
        return value;
    }

    bool Sketch2RigService::json_bool_field(const std::string& line, const char* key, bool fallback)
    {
        const std::string needle = std::string("\"") + key + "\":";
        const size_t start = line.find(needle);
        if (start == std::string::npos)
            return fallback;

        size_t i = start + needle.size();
        while (i < line.size() && line[i] == ' ')
            ++i;
        if (line.compare(i, 4, "true") == 0)
            return true;
        if (line.compare(i, 5, "false") == 0)
            return false;
        return fallback;
    }

    Sketch2RigService::Event Sketch2RigService::parse_worker_line(const std::string& line)
    {
        const std::string event = json_string_field(line, "event");
        Event out;
        out.request_id = json_u64_field(line, "request_id");
        out.ok = json_bool_field(line, "ok", false);
        out.message = json_string_field(line, "message");
        out.output_path = json_string_field(line, "output_path");

        if (event == "loading")
            out.kind = EventKind::Loading;
        else if (event == "ready")
            out.kind = EventKind::Ready;
        else if (event == "busy")
            out.kind = EventKind::Busy;
        else if (event == "result")
            out.kind = EventKind::Result;
        else if (event == "error")
            out.kind = EventKind::Error;
        else if (event == "stopped")
            out.kind = EventKind::Stopped;
        else
            out.kind = EventKind::Log;
        return out;
    }
}
