        struct Sketch2RigJoint2d
        {
            int index = -1;
            int parent = -1;
            float x = 0.0f;
            float y = 0.0f;
            bool valid = false;
        };

        struct VitPoseKeypoint2d
        {
            float x = 0.0f;
            float y = 0.0f;
            float score = 0.0f;
            bool valid = false;
        };

        struct CanvaPreprocessTransform
        {
            float scale = 1.0f;
            ImVec2 src_center = ImVec2(0.0f, 0.0f);
            ImVec2 dst_center = ImVec2(0.0f, 0.0f);
            bool valid = false;
        };

        static const char* model_inference_status_text(ModelInferenceStatus status)
        {
            switch (status)
            {
            case ModelInferenceStatus::NotStarted: return "Not started";
            case ModelInferenceStatus::Loading: return "Loading";
            case ModelInferenceStatus::Ready: return "Loaded";
            case ModelInferenceStatus::Busy: return "Processing";
            case ModelInferenceStatus::Error: return "Error";
            case ModelInferenceStatus::Stopped: return "Stopped";
            }
            return "Unknown";
        }

        static ImVec4 model_inference_status_color(ModelInferenceStatus status)
        {
            switch (status)
            {
            case ModelInferenceStatus::Ready: return ImVec4(0.20f, 0.85f, 0.38f, 1.0f);
            case ModelInferenceStatus::Busy: return ImVec4(0.95f, 0.72f, 0.20f, 1.0f);
            case ModelInferenceStatus::Loading: return ImVec4(0.30f, 0.62f, 1.0f, 1.0f);
            case ModelInferenceStatus::Error: return ImVec4(1.0f, 0.32f, 0.28f, 1.0f);
            case ModelInferenceStatus::Stopped: return ImVec4(0.70f, 0.70f, 0.70f, 1.0f);
            case ModelInferenceStatus::NotStarted: return ImVec4(0.60f, 0.60f, 0.60f, 1.0f);
            }
            return ImVec4(0.60f, 0.60f, 0.60f, 1.0f);
        }

        enum class GenerateClothStepStatus
        {
            NotStarted,
            Working,
            Done,
            Failed
        };

        struct GenerateClothStepView
        {
            const char* label = "";
            GenerateClothStepStatus status = GenerateClothStepStatus::NotStarted;
            float progress = -1.0f;
            std::string detail;
            std::vector<std::string> details;
            double seconds = -1.0;
            double percent = -1.0;
        };

        static int generate_cloth_step_index(GenerateClothStepId step)
        {
            return static_cast<int>(step);
        }

        static std::string canva_format_seconds(double seconds)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(seconds >= 10.0 ? 1 : 2)
                << std::max(0.0, seconds) << "s";
            return oss.str();
        }

        static std::string canva_format_timing_share(double seconds, double total_seconds)
        {
            std::ostringstream oss;
            const double pct = total_seconds > 0.0 ? (100.0 * seconds / total_seconds) : 0.0;
            oss << canva_format_seconds(seconds) << " / " << std::fixed << std::setprecision(1) << pct << "%";
            return oss.str();
        }

        static void canva_generate_cloth_reset_timings(CanvaState& canva, bool include_fitting_steps)
        {
            canva.generate_cloth_include_fitting_steps = include_fitting_steps;
            canva.generate_cloth_timing_started = true;
            canva.generate_cloth_timing_complete = false;
            canva.generate_cloth_total_seconds = 0.0;
            canva.generate_cloth_pipeline_start = std::chrono::steady_clock::now();
            canva.generate_cloth_step_timings = {};
            canva.generate_cloth_last_sim_phase = -1;
        }

        static void canva_generate_cloth_start_step(CanvaState& canva, GenerateClothStepId step)
        {
            if (!canva.generate_cloth_timing_started)
                canva_generate_cloth_reset_timings(canva, canva.generate_cloth_include_fitting_steps);

            GenerateClothStepTiming& timing = canva.generate_cloth_step_timings[(size_t)generate_cloth_step_index(step)];
            if (timing.finished || timing.started)
                return;
            timing.started = true;
            timing.skipped = false;
            timing.start_time = std::chrono::steady_clock::now();
        }

        static void canva_generate_cloth_finish_step(CanvaState& canva, GenerateClothStepId step, bool skipped = false)
        {
            GenerateClothStepTiming& timing = canva.generate_cloth_step_timings[(size_t)generate_cloth_step_index(step)];
            if (timing.finished)
            {
                if (skipped)
                    timing.skipped = true;
                return;
            }
            const auto now = std::chrono::steady_clock::now();
            if (timing.started)
                timing.seconds += std::chrono::duration<double>(now - timing.start_time).count();
            else
                timing.started = true;
            timing.finished = true;
            timing.skipped = skipped;
        }

        static void canva_generate_cloth_append_detail(
            CanvaState& canva,
            GenerateClothStepId step,
            const std::string& detail)
        {
            if (detail.empty())
                return;
            canva.generate_cloth_step_timings[(size_t)generate_cloth_step_index(step)].details.push_back(detail);
        }

        static void canva_generate_cloth_append_timed_detail(
            CanvaState& canva,
            GenerateClothStepId step,
            const char* label,
            std::chrono::steady_clock::time_point start_time)
        {
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();
            std::ostringstream oss;
            oss << label << ": " << canva_format_seconds(seconds);
            canva_generate_cloth_append_detail(canva, step, oss.str());
        }

        static void canva_generate_cloth_finish_pipeline(CanvaState& canva)
        {
            if (!canva.generate_cloth_timing_started || canva.generate_cloth_timing_complete)
                return;
            canva.generate_cloth_total_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - canva.generate_cloth_pipeline_start).count();
            canva.generate_cloth_timing_complete = true;
        }

        static void canva_generate_cloth_update_simulation_phase(const AppState& state, CanvaState& canva)
        {
            if (!canva.generate_cloth_waiting_simulation)
                return;

            static constexpr GenerateClothStepId phase_steps[] = {
                GenerateClothStepId::Sequence,
                GenerateClothStepId::Proxy,
                GenerateClothStepId::Simulation
            };
            const int totals[] = {
                state.controls.sim_sequence_steps_total,
                state.controls.sim_proxy_steps_total,
                state.controls.sim_solver_steps_total
            };
            const int phase = std::clamp(state.controls.sim_pipeline_stage, 0, 2);
            if (canva.generate_cloth_last_sim_phase == phase)
                return;

            if (canva.generate_cloth_last_sim_phase >= 0)
                canva_generate_cloth_finish_step(
                    canva,
                    phase_steps[canva.generate_cloth_last_sim_phase],
                    totals[canva.generate_cloth_last_sim_phase] <= 0);
            for (int i = canva.generate_cloth_last_sim_phase + 1; i < phase; ++i)
                canva_generate_cloth_finish_step(canva, phase_steps[i], totals[i] <= 0);
            canva_generate_cloth_start_step(canva, phase_steps[phase]);
            canva.generate_cloth_last_sim_phase = phase;
        }

        static void canva_generate_cloth_finish_simulation_steps(const AppState& state, CanvaState& canva)
        {
            static constexpr GenerateClothStepId phase_steps[] = {
                GenerateClothStepId::Sequence,
                GenerateClothStepId::Proxy,
                GenerateClothStepId::Simulation
            };
            const int totals[] = {
                state.controls.sim_sequence_steps_total,
                state.controls.sim_proxy_steps_total,
                state.controls.sim_solver_steps_total
            };
            for (int i = 0; i < 3; ++i)
            {
                const bool skipped = totals[i] <= 0;
                canva_generate_cloth_finish_step(canva, phase_steps[i], skipped);
            }
            canva.generate_cloth_last_sim_phase = -1;
        }

        static ImVec4 generate_cloth_step_color(GenerateClothStepStatus status)
        {
            switch (status)
            {
            case GenerateClothStepStatus::Done: return ImVec4(0.20f, 0.85f, 0.38f, 1.0f);
            case GenerateClothStepStatus::Working: return ImVec4(0.95f, 0.72f, 0.20f, 1.0f);
            case GenerateClothStepStatus::Failed: return ImVec4(1.0f, 0.32f, 0.28f, 1.0f);
            case GenerateClothStepStatus::NotStarted: return ImVec4(0.50f, 0.52f, 0.55f, 1.0f);
            }
            return ImVec4(0.50f, 0.52f, 0.55f, 1.0f);
        }

        static const char* generate_cloth_step_status_text(GenerateClothStepStatus status)
        {
            switch (status)
            {
            case GenerateClothStepStatus::Done: return "Done";
            case GenerateClothStepStatus::Working: return "Working";
            case GenerateClothStepStatus::Failed: return "Failed";
            case GenerateClothStepStatus::NotStarted: return "Not started";
            }
            return "Not started";
        }

        static int build_generate_cloth_steps(
            const AppState& state,
            const CanvaState& canva,
            std::array<GenerateClothStepView, kGenerateClothStepCount>& steps)
        {
            const bool failed = canva.generate_cloth_status == "Failed";
            const bool done = canva.generate_cloth_status == "Done";
            const bool infer_working = canva.generate_cloth_active && canva.extract_pose_active;
            const bool pose_working = canva.generate_cloth_waiting_pose;
            const bool sim_working = canva.generate_cloth_waiting_simulation;
            const bool render_working = canva.generate_cloth_waiting_render;
            const bool fitting_infer_working =
                canva.cloth_fitting_active
                && (canva.cloth_fitting_waiting_iso || canva.cloth_fitting_waiting_depth);
            const bool fitting_working =
                canva.cloth_fitting_active
                && !canva.cloth_fitting_waiting_iso
                && !canva.cloth_fitting_waiting_depth;
            const bool show_fitting_steps =
                canva.generate_cloth_include_fitting_steps
                || canva.generate_cloth_status == "Fitting"
                || canva.cloth_fitting_active
                || canva.generate_cloth_step_timings[(size_t)generate_cloth_step_index(GenerateClothStepId::FittingInference)].started
                || canva.generate_cloth_step_timings[(size_t)generate_cloth_step_index(GenerateClothStepId::Fitting)].started;
            const int step_count = show_fitting_steps ? kGenerateClothStepCount : kGenerateClothBaseStepCount;
            const int failed_step = failed
                ? ((canva.generate_cloth_failed_step >= 0 && canva.generate_cloth_failed_step < step_count)
                    ? canva.generate_cloth_failed_step
                    : 0)
                : -1;

            const bool infer_done = done || pose_working || sim_working || render_working || fitting_infer_working || fitting_working;
            const bool pose_done = done || sim_working || render_working || fitting_infer_working || fitting_working;
            const bool sequence_done = done || render_working
                || fitting_infer_working || fitting_working
                || (sim_working && state.controls.sim_pipeline_stage > 0);
            const bool proxy_done = done || render_working
                || fitting_infer_working || fitting_working
                || (sim_working && state.controls.sim_pipeline_stage > 1);
            const bool sim_done = done || render_working || fitting_infer_working || fitting_working;
            const bool render_done = done || fitting_infer_working || fitting_working;
            const bool fitting_infer_done = done || fitting_working
                || canva.generate_cloth_step_timings[(size_t)generate_cloth_step_index(GenerateClothStepId::FittingInference)].finished;
            const bool fitting_done = done
                || canva.generate_cloth_step_timings[(size_t)generate_cloth_step_index(GenerateClothStepId::Fitting)].finished;

            const double pipeline_seconds = canva.generate_cloth_timing_complete
                ? canva.generate_cloth_total_seconds
                : (canva.generate_cloth_timing_started
                    ? std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - canva.generate_cloth_pipeline_start).count()
                    : 0.0);

            const auto set_progress = [](GenerateClothStepView& step, int progress_done, int progress_total)
            {
                if (progress_total <= 0)
                {
                    if (step.status == GenerateClothStepStatus::Done)
                        step.details.push_back("Progress: skipped");
                    return;
                }
                step.progress = (float)std::clamp(progress_done, 0, progress_total) / (float)progress_total;
                step.details.push_back("Progress: "
                    + std::to_string(std::clamp(progress_done, 0, progress_total))
                    + "/" + std::to_string(progress_total));
            };

            steps[0].label = "Inference";
            steps[0].status = infer_done ? GenerateClothStepStatus::Done
                : infer_working ? GenerateClothStepStatus::Working
                                : (failed ? GenerateClothStepStatus::Failed : GenerateClothStepStatus::NotStarted);

            steps[1].label = "Pose";
            steps[1].status = pose_done ? GenerateClothStepStatus::Done
                : pose_working ? GenerateClothStepStatus::Working
                               : GenerateClothStepStatus::NotStarted;

            steps[2].label = "Sequence creation";
            steps[2].status = sequence_done ? GenerateClothStepStatus::Done
                : (sim_working && state.controls.sim_pipeline_stage == 0)
                    ? GenerateClothStepStatus::Working
                    : GenerateClothStepStatus::NotStarted;
            if (sim_working || sequence_done)
                set_progress(steps[2], state.controls.sim_sequence_steps_done, state.controls.sim_sequence_steps_total);

            steps[3].label = "Proxy creation";
            steps[3].status = proxy_done ? GenerateClothStepStatus::Done
                : (sim_working && state.controls.sim_pipeline_stage == 1)
                    ? GenerateClothStepStatus::Working
                    : GenerateClothStepStatus::NotStarted;
            if (sim_working || proxy_done)
                set_progress(steps[3], state.controls.sim_proxy_steps_done, state.controls.sim_proxy_steps_total);

            steps[4].label = "Simulation";
            steps[4].status = sim_done ? GenerateClothStepStatus::Done
                : (sim_working && state.controls.sim_pipeline_stage == 2)
                    ? GenerateClothStepStatus::Working
                    : GenerateClothStepStatus::NotStarted;
            if (sim_working || sim_done)
                set_progress(steps[4], state.controls.sim_solver_steps_done, state.controls.sim_solver_steps_total);

            steps[5].label = "Render";
            steps[5].status = render_done ? GenerateClothStepStatus::Done
                : render_working ? GenerateClothStepStatus::Working
                                 : GenerateClothStepStatus::NotStarted;

            steps[6].label = "Fitting inference";
            steps[6].status = fitting_infer_done ? GenerateClothStepStatus::Done
                : fitting_infer_working ? GenerateClothStepStatus::Working
                                        : GenerateClothStepStatus::NotStarted;

            steps[7].label = "Fitting";
            steps[7].status = fitting_done ? GenerateClothStepStatus::Done
                : fitting_working ? GenerateClothStepStatus::Working
                                  : GenerateClothStepStatus::NotStarted;

            if (failed)
            {
                for (int i = 0; i < step_count; ++i)
                {
                    if (i < failed_step)
                        steps[i].status = GenerateClothStepStatus::Done;
                    else if (i == failed_step)
                        steps[i].status = GenerateClothStepStatus::Failed;
                    else
                        steps[i].status = GenerateClothStepStatus::NotStarted;
                }
            }

            for (int i = 0; i < step_count; ++i)
            {
                const GenerateClothStepTiming& timing = canva.generate_cloth_step_timings[(size_t)i];
                if (timing.finished)
                    steps[i].status = GenerateClothStepStatus::Done;
                else if (timing.started && !failed)
                    steps[i].status = GenerateClothStepStatus::Working;
                if (failed && i == failed_step)
                    steps[i].status = GenerateClothStepStatus::Failed;

                double seconds = timing.seconds;
                if (timing.started && !timing.finished)
                    seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - timing.start_time).count();
                if (timing.started)
                {
                    steps[i].seconds = seconds;
                    steps[i].percent = pipeline_seconds > 0.0 ? (100.0 * seconds / pipeline_seconds) : 0.0;
                    steps[i].detail = canva_format_timing_share(seconds, pipeline_seconds);
                }
                if (timing.skipped)
                    steps[i].details.push_back("Skipped");
                for (const std::string& detail : timing.details)
                    steps[i].details.push_back(detail);
            }

            for (const std::string& detail : state.controls.sim_sequence_timing_details)
                steps[2].details.push_back(detail);
            for (const std::string& detail : state.controls.sim_proxy_timing_details)
                steps[3].details.push_back(detail);
            for (const std::string& detail : state.controls.sim_solver_timing_details)
                steps[4].details.push_back(detail);

            return step_count;
        }

        static void draw_generate_cloth_steps_in_view(
            const AppState& state,
            const CanvaState& canva,
            ImDrawList* dl,
            const ImVec2& anchor)
        {
            std::array<GenerateClothStepView, kGenerateClothStepCount> steps;
            const int step_count = build_generate_cloth_steps(state, canva, steps);

            const ImFont* font = ImGui::GetFont();
            const float font_size = ImGui::GetFontSize();
            const float title_size = font_size + 2.0f;
            const float w = 360.0f;
            const float base_row_h = 58.0f;
            const float detail_line_h = 15.0f;
            const float header_h = 62.0f;
            const float pad = 12.0f;
            std::array<float, kGenerateClothStepCount> row_heights{};
            float rows_h = 0.0f;
            for (int i = 0; i < step_count; ++i)
            {
                const int max_details = (i == generate_cloth_step_index(GenerateClothStepId::Simulation)) ? 11 : 4;
                const int detail_count = std::min<int>((int)steps[(size_t)i].details.size(), max_details);
                row_heights[(size_t)i] = base_row_h + detail_line_h * (float)detail_count;
                rows_h += row_heights[(size_t)i];
            }
            const float h = header_h + rows_h + pad;
            const ImVec2 p0 = anchor;
            const ImVec2 p1(p0.x + w, p0.y + h);
            const double pipeline_seconds = canva.generate_cloth_timing_complete
                ? canva.generate_cloth_total_seconds
                : (canva.generate_cloth_timing_started
                    ? std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - canva.generate_cloth_pipeline_start).count()
                    : 0.0);

            dl->AddRectFilled(p0, p1, IM_COL32(18, 20, 24, 218), 7.0f);
            dl->AddRect(p0, p1, IM_COL32(255, 255, 255, 42), 7.0f, 0, 1.0f);
            dl->AddText(font, title_size, ImVec2(p0.x + pad, p0.y + 10.0f), IM_COL32(238, 240, 244, 255), "Generate Cloth");
            dl->AddText(font, font_size, ImVec2(p0.x + pad, p0.y + 28.0f), IM_COL32(176, 181, 190, 255), canva.generate_cloth_status.c_str());
            const std::string total_text = "Pipeline: " + canva_format_seconds(pipeline_seconds);
            dl->AddText(font, font_size - 1.0f, ImVec2(p0.x + pad, p0.y + 45.0f), IM_COL32(176, 181, 190, 255), total_text.c_str());

            const float rail_x = p0.x + 22.0f;
            const float rows_y = p0.y + header_h;
            dl->AddLine(
                ImVec2(rail_x, rows_y + 17.0f),
                ImVec2(rail_x, rows_y + rows_h - 18.0f),
                IM_COL32(255, 255, 255, 38),
                2.0f);

            float y = rows_y;
            for (int i = 0; i < step_count; ++i)
            {
                const GenerateClothStepView& step = steps[i];
                const ImVec4 col = generate_cloth_step_color(step.status);
                const ImU32 accent = ImGui::ColorConvertFloat4ToU32(col);
                const ImU32 soft = ImGui::ColorConvertFloat4ToU32(ImVec4(col.x, col.y, col.z, 0.16f));
                const float row_h = row_heights[(size_t)i];
                const ImVec2 row0(p0.x + 8.0f, y + 4.0f);
                const ImVec2 row1(p1.x - 8.0f, y + row_h - 4.0f);

                dl->AddRectFilled(row0, row1, soft, 6.0f);
                dl->AddCircleFilled(ImVec2(rail_x, y + row_h * 0.5f), 5.5f, accent, 18);
                dl->AddText(font, font_size, ImVec2(p0.x + 38.0f, y + 11.0f), accent, step.label);
                dl->AddText(
                    font,
                    font_size - 1.0f,
                    ImVec2(p0.x + 38.0f, y + 29.0f),
                    IM_COL32(190, 195, 204, 255),
                    generate_cloth_step_status_text(step.status));

                if (step.progress >= 0.0f)
                {
                    const float bar_x0 = p0.x + 210.0f;
                    const float bar_x1 = p1.x - 16.0f;
                    const float bar_y = y + 34.0f;
                    const float t = std::clamp(step.progress, 0.0f, 1.0f);
                    dl->AddRectFilled(ImVec2(bar_x0, bar_y), ImVec2(bar_x1, bar_y + 5.0f), IM_COL32(62, 66, 74, 255), 2.5f);
                    dl->AddRectFilled(ImVec2(bar_x0, bar_y), ImVec2(bar_x0 + (bar_x1 - bar_x0) * t, bar_y + 5.0f), accent, 2.5f);
                }

                if (!step.detail.empty())
                {
                    dl->AddText(
                        font,
                        font_size - 1.0f,
                        ImVec2(p0.x + 210.0f, y + 12.0f),
                        IM_COL32(190, 195, 204, 255),
                        step.detail.c_str());
                }

                const int max_details = (i == generate_cloth_step_index(GenerateClothStepId::Simulation)) ? 11 : 4;
                const int detail_count = std::min<int>((int)step.details.size(), max_details);
                for (int j = 0; j < detail_count; ++j)
                {
                    dl->AddText(
                        font,
                        font_size - 2.0f,
                        ImVec2(p0.x + 38.0f, y + 47.0f + (float)j * detail_line_h),
                        IM_COL32(168, 174, 184, 255),
                        step.details[(size_t)j].c_str());
                }
                y += row_h;
            }
        }

