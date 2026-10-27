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
    void draw_canva_controls(
        AppState& state, CanvaState& canva, Renderer& renderer, bool parameters_only)
    {
    // Scrollable Canva controls or the fitting-parameters-only view.
    ImGui::BeginChild("##canva_controls", ImVec2(0.0f, 0.0f), true,
        ImGuiWindowFlags_AlwaysVerticalScrollbar);
    {
        const auto show_canva_layers = [&]()
        {
            canva.sequence_preview_active = false;
            canva.sequence_preview_frame_id = 0;
        };

        if (!parameters_only)
        {

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.16f, 0.16f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.70f, 0.20f, 0.20f, 1.0f));
        if (ImGui::Button("Reset Everything"))
            ImGui::OpenPopup("Reset Canva?");
        ImGui::PopStyleColor(2);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Clear canvas layers, rig, UV/maps, generated renders, pose, shape, and simulation.");

        if (ImGui::BeginPopupModal("Reset Canva?", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::TextUnformatted("Reset all computed Canva data?");
            ImGui::TextUnformatted("Generated files on disk will be kept.");
            ImGui::Spacing();
            if (ImGui::Button("Reset", ImVec2(120.0f, 0.0f)))
            {
                canva_reset_all(state, canva, renderer);
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120.0f, 0.0f)))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Brush");
        ImGui::Separator();

        const bool active_visibility_map =
            canva.active_layer >= 0
            && canva.active_layer < (int)canva.layers.size()
            && canva_layer_is_visibility_map(canva.layers[(size_t)canva.active_layer]);
        const bool active_deformation_field =
            canva.active_layer >= 0
            && canva.active_layer < (int)canva.layers.size()
            && canva.layers[(size_t)canva.active_layer].kind
                == CanvaLayerKind::ClothFittingDeformation
            && !canva.layers[(size_t)canva.active_layer].deformation_source_x.empty()
            && !canva.layers[(size_t)canva.active_layer].deformation_source_y.empty();
        ImGui::BeginDisabled(canva.tool != CanvaTool::Brush);
        ImGui::SliderFloat("Size", &canva.brush_radius, 0.5f, 250.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        if (active_visibility_map)
        {
            ImGui::TextUnformatted("Visibility paint");
            if (ImGui::RadioButton("Visible (green)", canva.visibility_brush_visible))
                canva.visibility_brush_visible = true;
            ImGui::SameLine();
            if (ImGui::RadioButton("Hidden (red)", !canva.visibility_brush_visible))
                canva.visibility_brush_visible = false;
            ImGui::TextDisabled("Red/green only; alpha is locked.");
        }
        else if (active_deformation_field)
        {
            ImGui::TextUnformatted("Deformation brush");
            if (ImGui::RadioButton(
                    "Push",
                    canva.deformation_brush_mode == CanvaDeformationBrushMode::Push))
            {
                canva.deformation_brush_mode = CanvaDeformationBrushMode::Push;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton(
                    "Smooth",
                    canva.deformation_brush_mode == CanvaDeformationBrushMode::Smooth))
            {
                canva.deformation_brush_mode = CanvaDeformationBrushMode::Smooth;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton(
                    "Erase",
                    canva.deformation_brush_mode == CanvaDeformationBrushMode::Erase))
            {
                canva.deformation_brush_mode = CanvaDeformationBrushMode::Erase;
            }
            ImGui::SliderFloat(
                "Power",
                &canva.deformation_brush_power,
                0.0f,
                1.0f,
                "%.2f");
            canva.deformation_brush_power =
                std::clamp(canva.deformation_brush_power, 0.0f, 1.0f);
            const char* deformation_hint =
                canva.deformation_brush_mode == CanvaDeformationBrushMode::Push
                ? "Drag to push the refitted cloth."
                : canva.deformation_brush_mode == CanvaDeformationBrushMode::Smooth
                ? "Paint to smooth nearby deformation."
                : "Paint to remove all deformation, including the computed fit.";
            ImGui::TextDisabled("%s", deformation_hint);
            ImGui::TextDisabled("Red = horizontal, green = vertical, blue = magnitude.");
        }
        else
        {
            ImGui::ColorEdit4("Color", (float*)&canva.brush_color, ImGuiColorEditFlags_AlphaBar);
        }
        ImGui::EndDisabled();

        // Persist brush settings via AppState -> imgui.ini handler.
        state.canva_brush_radius = std::clamp(canva.brush_radius, 0.5f, 250.0f);
        state.canva_brush_color = {
            std::clamp(canva.brush_color.x, 0.0f, 1.0f),
            std::clamp(canva.brush_color.y, 0.0f, 1.0f),
            std::clamp(canva.brush_color.z, 0.0f, 1.0f),
            std::clamp(canva.brush_color.w, 0.0f, 1.0f)
        };

        if (ImGui::Button("Reset view"))
        {
            canva.zoom = 1.0f;
            canva.pan = ImVec2(0.0f, 0.0f);
        }
        ImGui::SameLine();
        ImGui::Text("Zoom: %.2fx", (double)canva.zoom);

        ImGui::Separator();
        ImGui::TextUnformatted("Tool");
        ImGui::Separator();
        if (ImGui::RadioButton("Brush", canva.tool == CanvaTool::Brush))
        {
            canva.tool = CanvaTool::Brush;
            canva_cancel_interactions(canva);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Transform", canva.tool == CanvaTool::Transform))
        {
            canva.tool = CanvaTool::Transform;
            canva_cancel_interactions(canva);
        }

        ImGui::TextUnformatted("Layers");
        ImGui::Separator();

        if (ImGui::Button("Load##canva_load_image"))
        {
            const char* patterns[] = {
                "*.png", "*.PNG",
                "*.jpg", "*.JPG",
                "*.jpeg", "*.JPEG",
                "*.gif", "*.GIF",
                "*.mp4", "*.MP4"
            };

            std::error_code ec;
            std::filesystem::path initial_dir = std::filesystem::path("assets") / "sketches";
            if (!std::filesystem::exists(initial_dir, ec))
                initial_dir = std::filesystem::path("assets") / "sketch";
            const std::string initial_dir_string =
                std::filesystem::exists(initial_dir, ec) ? initial_dir.string() : std::string();

            const char* path = tinyfd_openFileDialog(
                "Load image or video",
                initial_dir_string.c_str(),
                (int)(sizeof(patterns) / sizeof(patterns[0])),
                patterns,
                "Image/video files (*.png, *.jpg, *.jpeg, *.gif, *.mp4)",
                0);

            if (path && path[0] != '\0')
            {
                std::string err;
                const std::filesystem::path load_path(path);
                std::string ext = load_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                    [](unsigned char c) { return (char)std::tolower(c); });
                if ((ext == ".gif" || ext == ".mp4")
                    && canva_sequence_import_sketch_file(state, canva, load_path, &err))
                {
                    console_add_line(state, std::string("Canva: Loaded sketch video into sequence: ") + path);
                }
                else if ((ext == ".gif" || ext == ".mp4"))
                {
                    console_add_line(state, std::string("Canva: Video load failed: ") + (err.empty() ? "(unknown error)" : err));
                }
                else if (canva_add_png_layer(canva, path, &err))
                {
                    console_add_line(state, std::string("Canva: Loaded image layer: ") + canva.layers.front().name);
                }
                else
                {
                    console_add_line(state, std::string("Canva: Image load failed: ") + (err.empty() ? "(unknown error)" : err));
                }
            }
            else
            {
                console_add_line(state, "Canva: Load cancelled");
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Save PNG##canva_save_image"))
        {
            std::filesystem::path out_path;
            std::string err;
            if (canva_save_visible_layers_png(canva, &out_path, &err))
            {
                console_add_line(state, std::string("Canva: Saved PNG: ") + out_path.string());
            }
            else
            {
                console_add_line(state, std::string("Canva: Save failed: ") + (err.empty() ? "(unknown error)" : err));
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Save visible layers as a PNG with a transparent background.");

        ImGui::Separator();
        ImGui::TextUnformatted("VitPose");
        ImGui::SameLine();
        ImGui::TextColored(model_inference_status_color(state.vitpose.status), "%s",
            model_inference_status_text(state.vitpose.status));

        if (!state.vitpose.message.empty())
            ImGui::TextWrapped("%s", state.vitpose.message.c_str());

        const bool can_predict_2d_pose =
            state.vitpose.service != nullptr
            && !canva.vitpose_pending_inference
            && state.vitpose.status != ModelInferenceStatus::Loading
            && state.vitpose.status != ModelInferenceStatus::Busy;
        ImGui::BeginDisabled(!can_predict_2d_pose);
        if (ImGui::Button("Predict 2D Pose"))
        {
            std::filesystem::path sketch_path;
            std::string error;
            if (canva_save_model_input_png(canva, &sketch_path, &error))
            {
                const std::filesystem::path output_path =
                    sketch_path.parent_path() / "sketch_vitpose.json";
                const std::uint64_t request_id = state.vitpose.next_request_id++;
                state.vitpose.last_sketch_path = sketch_path.string();

                canva.vitpose_pending_inference = true;
                canva.vitpose_pending_request_id = request_id;
                canva.vitpose_pending_sketch_path = sketch_path;
                canva.vitpose_pending_output_path = output_path;

                if (state.vitpose.status == ModelInferenceStatus::Ready)
                {
                    (void)canva_try_submit_pending_vitpose(state, canva);
                }
                else
                {
                    state.vitpose.message = "Loading VitPose for 2D pose prediction...";
                    state.vitpose.status = ModelInferenceStatus::Loading;
                    state.vitpose.service->start(
                        poseweaver::assets_dir() / "VitPose",
                        std::filesystem::path(POSEWEAVER_VITPOSE_WORKER),
                        "vitpose_base_coco_aic_mpii.pth");
                    console_add_line(state, "Canva: Loading VitPose for 2D pose prediction");
                }
            }
            else
            {
                console_add_line(state, "Canva: VitPose preprocessing failed: "
                    + (error.empty() ? std::string("(unknown error)") : error));
            }
        }
        ImGui::EndDisabled();

        if (!state.vitpose.last_output_path.empty())
            ImGui::TextWrapped("Output: %s", state.vitpose.last_output_path.c_str());

        ImGui::Separator();
        ImGui::TextUnformatted("NLF");
        ImGui::SameLine();
        ImGui::TextColored(model_inference_status_color(state.nlf.status), "%s",
            model_inference_status_text(state.nlf.status));

        if (!state.nlf.message.empty())
            ImGui::TextWrapped("%s", state.nlf.message.c_str());

        const bool can_predict_nlf_bones =
            state.nlf.service != nullptr &&
            state.nlf.status == ModelInferenceStatus::Ready;
        ImGui::BeginDisabled(!can_predict_nlf_bones);
        if (ImGui::Button("Predict Bones with NLF"))
        {
            std::filesystem::path sketch_path;
            std::string err;
            if (canva_save_model_input_png(canva, &sketch_path, &err))
            {
                const std::filesystem::path output_path =
                    sketch_path.parent_path() / "sketch_nlf.json";
                const std::uint64_t request_id = state.nlf.next_request_id++;

                state.nlf.last_sketch_path = sketch_path.string();
                console_add_line(state, std::string("Canva: Saved normalized sketch for NLF: ") + sketch_path.string());

                std::string request_err;
                if (state.nlf.service->request_inference(request_id, sketch_path, output_path, &request_err))
                {
                    state.nlf.status = ModelInferenceStatus::Busy;
                    state.nlf.active_request_id = request_id;
                    state.nlf.message = "Predicting bones with NLF...";
                    canva.bone_prediction_debug_mode = 3;
                    canva.bone_prediction_nlf_confidence = -1.0f;
                    canva.bone_prediction_debug_status = "NLF running";
                }
                else
                {
                    state.nlf.status = ModelInferenceStatus::Error;
                    state.nlf.message = request_err.empty() ? "Failed to queue inference" : request_err;
                    canva.bone_prediction_debug_mode = 4;
                    canva.bone_prediction_debug_status = "NLF queue failed";
                    console_add_line(state, "NLF: " + state.nlf.message);
                }
            }
            else
            {
                canva.bone_prediction_debug_mode = 4;
                canva.bone_prediction_debug_status = "NLF preprocessing failed";
                console_add_line(state, std::string("Canva: NLF preprocessing failed: ")
                    + (err.empty() ? "(unknown error)" : err));
            }
        }
        ImGui::EndDisabled();

        if (!state.nlf.last_output_path.empty())
            ImGui::TextWrapped("Output: %s", state.nlf.last_output_path.c_str());

        ImGui::Separator();
        ImGui::TextUnformatted("Sketch2Rig");
        ImGui::SameLine();
        ImGui::TextColored(model_inference_status_color(state.sketch2rig.status), "%s",
            model_inference_status_text(state.sketch2rig.status));

        if (!state.sketch2rig.message.empty())
            ImGui::TextWrapped("%s", state.sketch2rig.message.c_str());

        const bool can_process =
            state.sketch2rig.service != nullptr &&
            state.sketch2rig.status == ModelInferenceStatus::Ready;
        ImGui::BeginDisabled(!can_process);
        if (ImGui::Button("Extract Rig"))
        {
            std::filesystem::path sketch_path;
            std::string err;
            if (canva_save_model_input_png(canva, &sketch_path, &err))
            {
                const std::filesystem::path output_path =
                    sketch_path.parent_path() / ("sketch_rig.json");
                const std::uint64_t request_id = state.sketch2rig.next_request_id++;

                state.sketch2rig.last_sketch_path = sketch_path.string();
                console_add_line(state, std::string("Canva: Saved normalized sketch for Sketch2Rig: ") + sketch_path.string());

                std::string request_err;
                if (state.sketch2rig.service->request_inference(request_id, sketch_path, output_path, &request_err))
                {
                    state.sketch2rig.status = ModelInferenceStatus::Busy;
                    state.sketch2rig.active_request_id = request_id;
                    state.sketch2rig.message = "Processing sketch with Sketch2Rig...";
                }
                else
                {
                    state.sketch2rig.status = ModelInferenceStatus::Error;
                    state.sketch2rig.message = request_err.empty() ? "Failed to queue inference" : request_err;
                    console_add_line(state, "Sketch2Rig: " + state.sketch2rig.message);
                }
            }
            else
            {
                console_add_line(state, std::string("Canva: Process preprocessing failed: ") + (err.empty() ? "(unknown error)" : err));
            }
        }
        ImGui::EndDisabled();

        if (!state.sketch2rig.last_output_path.empty())
            ImGui::TextWrapped("Output: %s", state.sketch2rig.last_output_path.c_str());

        ImGui::Separator();
        ImGui::TextUnformatted("Sketch2Betas");
        ImGui::SameLine();
        ImGui::TextColored(model_inference_status_color(state.sketch2betas.status), "%s",
            model_inference_status_text(state.sketch2betas.status));

        if (!state.sketch2betas.message.empty())
            ImGui::TextWrapped("%s", state.sketch2betas.message.c_str());

        const bool can_extract_betas =
            state.sketch2betas.service != nullptr &&
            state.sketch2betas.status == ModelInferenceStatus::Ready;
        ImGui::BeginDisabled(!can_extract_betas);
        if (ImGui::Button("Extract betas"))
        {
            std::filesystem::path sketch_path;
            std::string err;
            if (canva_save_model_input_png(canva, &sketch_path, &err))
            {
                const std::filesystem::path output_path =
                    sketch_path.parent_path() / ("sketch_betas.json");
                const std::uint64_t request_id = state.sketch2betas.next_request_id++;

                state.sketch2betas.last_sketch_path = sketch_path.string();
                console_add_line(state, std::string("Canva: Saved normalized sketch for Sketch2Betas: ") + sketch_path.string());

                std::string request_err;
                if (state.sketch2betas.service->request_inference(request_id, sketch_path, output_path, &request_err))
                {
                    state.sketch2betas.status = ModelInferenceStatus::Busy;
                    state.sketch2betas.active_request_id = request_id;
                    state.sketch2betas.message = "Processing sketch with Sketch2Betas...";
                }
                else
                {
                    state.sketch2betas.status = ModelInferenceStatus::Error;
                    state.sketch2betas.message = request_err.empty() ? "Failed to queue inference" : request_err;
                    console_add_line(state, "Sketch2Betas: " + state.sketch2betas.message);
                }
            }
            else
            {
                console_add_line(state, std::string("Canva: Extract betas preprocessing failed: ") + (err.empty() ? "(unknown error)" : err));
            }
        }
        ImGui::EndDisabled();

        if (!state.sketch2betas.last_output_path.empty())
            ImGui::TextWrapped("Output: %s", state.sketch2betas.last_output_path.c_str());

        ImGui::Separator();
        ImGui::TextUnformatted("Sketch2UV");
        ImGui::SameLine();
        ImGui::TextColored(model_inference_status_color(state.sketch2uv.status), "%s",
            model_inference_status_text(state.sketch2uv.status));

        if (!state.sketch2uv.message.empty())
            ImGui::TextWrapped("%s", state.sketch2uv.message.c_str());

        const bool can_predict_uv =
            state.sketch2uv.service != nullptr &&
            (state.sketch2uv.status == ModelInferenceStatus::NotStarted
                || state.sketch2uv.status == ModelInferenceStatus::Ready);
        ImGui::BeginDisabled(!can_predict_uv);
        if (ImGui::Button("Predict UV"))
        {
            std::filesystem::path sketch_path;
            std::string err;
            if (canva_save_model_input_png(canva, &sketch_path, &err))
            {
                const std::filesystem::path output_path =
                    sketch_path.parent_path() / ("sketch_uv.json");
                const std::uint64_t request_id = state.sketch2uv.next_request_id++;

                state.sketch2uv.last_sketch_path = sketch_path.string();
                console_add_line(state, std::string("Canva: Saved normalized sketch for Sketch2UV: ") + sketch_path.string());

                std::string request_err;
                if (state.sketch2uv.service->request_inference(request_id, sketch_path, output_path, &request_err))
                {
                    state.sketch2uv.status = ModelInferenceStatus::Busy;
                    state.sketch2uv.active_request_id = request_id;
                    state.sketch2uv.message = "Processing sketch with Sketch2UV...";
                }
                else
                {
                    state.sketch2uv.status = ModelInferenceStatus::Error;
                    state.sketch2uv.message = request_err.empty() ? "Failed to queue inference" : request_err;
                    console_add_line(state, "Sketch2UV: " + state.sketch2uv.message);
                }
            }
            else
            {
                console_add_line(state, std::string("Canva: Predict UV preprocessing failed: ") + (err.empty() ? "(unknown error)" : err));
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Load UV##canva_load_uv"))
        {
            const char* patterns[] = { "*.png", "*.PNG" };

            std::error_code ec;
            std::filesystem::path initial_dir = std::filesystem::path("renders");
            if (!std::filesystem::exists(initial_dir, ec))
                initial_dir = std::filesystem::path("assets") / "sketches";
            const std::string initial_dir_string =
                std::filesystem::exists(initial_dir, ec) ? initial_dir.string() : std::string();

            const char* path = tinyfd_openFileDialog(
                "Load UV PNG",
                initial_dir_string.c_str(),
                (int)(sizeof(patterns) / sizeof(patterns[0])),
                patterns,
                "UV PNG files (*.png)",
                0);

            if (path && path[0] != '\0')
            {
                std::string err;
                const std::string layer_name = std::string("Loaded UV: ") + canva_basename(path);
                if (canva_add_canvas_filled_png_layer(
                        canva,
                        path,
                        layer_name,
                        CanvaLayerKind::ManualUV,
                        true,
                        &err))
                {
                    state.canva_manual_uv_png_path = path;
                    console_add_line(state, std::string("Canva: Loaded manual UV PNG: ") + path);
                }
                else
                {
                    console_add_line(state, std::string("Canva: UV PNG load failed: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
            else
            {
                console_add_line(state, "Canva: Load UV cancelled");
            }
        }

        if (!state.sketch2uv.last_output_path.empty())
            ImGui::TextWrapped("Output: %s", state.sketch2uv.last_output_path.c_str());
        if (!state.canva_manual_uv_png_path.empty())
        {
            ImGui::TextWrapped("Loaded UV: %s", state.canva_manual_uv_png_path.c_str());
            ImGui::SameLine();
            if (ImGui::Button("Clear##canva_clear_loaded_uv"))
            {
                state.canva_manual_uv_png_path.clear();
                console_add_line(state, "Canva: Cleared loaded UV PNG");
            }
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Sketch2Depth");
        ImGui::SameLine();
        ImGui::TextColored(model_inference_status_color(state.sketch2depth.status), "%s",
            model_inference_status_text(state.sketch2depth.status));

        if (!state.sketch2depth.message.empty())
            ImGui::TextWrapped("%s", state.sketch2depth.message.c_str());

        const bool can_predict_depth =
            state.sketch2depth.service != nullptr
            && state.sketch2depth.status == ModelInferenceStatus::Ready;
        ImGui::BeginDisabled(!can_predict_depth);
        if (ImGui::Button("Predict Depth"))
        {
            std::filesystem::path sketch_path;
            std::string error;
            if (canva_save_model_input_png(canva, &sketch_path, &error))
            {
                const std::filesystem::path output_path =
                    sketch_path.parent_path() / "sketch_depth.json";
                const std::uint64_t request_id = state.sketch2depth.next_request_id++;
                state.sketch2depth.last_sketch_path = sketch_path.string();
                console_add_line(state, "Canva: Saved normalized sketch for Sketch2Depth: "
                    + sketch_path.string());

                std::string request_error;
                if (state.sketch2depth.service->request_inference(
                        request_id, sketch_path, output_path, &request_error))
                {
                    state.sketch2depth.status = ModelInferenceStatus::Busy;
                    state.sketch2depth.active_request_id = request_id;
                    state.sketch2depth.message = "Processing sketch with Sketch2Depth...";
                }
                else
                {
                    state.sketch2depth.status = ModelInferenceStatus::Error;
                    state.sketch2depth.message = request_error.empty()
                        ? "Failed to queue inference"
                        : request_error;
                    console_add_line(state, "Sketch2Depth: " + state.sketch2depth.message);
                }
            }
            else
            {
                console_add_line(state, "Canva: Predict Depth preprocessing failed: "
                    + (error.empty() ? std::string("(unknown error)") : error));
            }
        }
        ImGui::EndDisabled();

        if (!state.sketch2depth.last_output_path.empty())
            ImGui::TextWrapped("Output: %s", state.sketch2depth.last_output_path.c_str());

        ImGui::Separator();
        ImGui::TextUnformatted("Sketch2ISO");
        ImGui::SameLine();
        ImGui::TextColored(model_inference_status_color(state.sketch2iso.status), "%s",
            model_inference_status_text(state.sketch2iso.status));

        if (!state.sketch2iso.message.empty())
            ImGui::TextWrapped("%s", state.sketch2iso.message.c_str());

        const bool can_predict_iso =
            state.sketch2iso.service != nullptr
            && state.sketch2iso.status == ModelInferenceStatus::Ready;
        ImGui::BeginDisabled(!can_predict_iso);
        if (ImGui::Button("Predict ISO"))
        {
            std::filesystem::path sketch_path;
            std::string err;
            if (canva_save_model_input_png(canva, &sketch_path, &err))
            {
                const std::filesystem::path output_path =
                    sketch_path.parent_path() / "sketch_iso.npz";
                const std::uint64_t request_id = state.sketch2iso.next_request_id++;

                state.sketch2iso.last_sketch_path = sketch_path.string();
                console_add_line(state, "Canva: Saved normalized sketch for Sketch2ISO: "
                    + sketch_path.string());

                std::string request_err;
                if (state.sketch2iso.service->request_inference(
                        request_id, sketch_path, output_path, &request_err))
                {
                    state.sketch2iso.status = ModelInferenceStatus::Busy;
                    state.sketch2iso.active_request_id = request_id;
                    state.sketch2iso.message = "Processing sketch with Sketch2ISO...";
                }
                else
                {
                    state.sketch2iso.status = ModelInferenceStatus::Error;
                    state.sketch2iso.message =
                        request_err.empty() ? "Failed to queue inference" : request_err;
                    console_add_line(state, "Sketch2ISO: " + state.sketch2iso.message);
                }
            }
            else
            {
                console_add_line(state, std::string("Canva: Predict ISO preprocessing failed: ")
                    + (err.empty() ? "(unknown error)" : err));
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Load ISO##canva_load_iso"))
        {
            const char* patterns[] = { "*.npz", "*.NPZ" };
            const std::filesystem::path initial_dir =
                assets_dir() / "sketches" / "sketches_iso";
            std::error_code ec;
            const std::string initial_dir_string =
                std::filesystem::exists(initial_dir, ec) ? initial_dir.string() : std::string();

            const char* path = tinyfd_openFileDialog(
                "Load ISO NPZ",
                initial_dir_string.c_str(),
                (int)(sizeof(patterns) / sizeof(patterns[0])),
                patterns,
                "ISO NPZ files (*.npz)",
                0);

            if (path && path[0] != '\0')
            {
                std::string err;
                const std::string layer_name =
                    std::string("Loaded ISO Debug: ") + canva_basename(path);
                if (canva_add_iso_debug_layer(canva, path, {}, layer_name, true, &err))
                {
                    state.canva_manual_iso_npz_path = path;
                    console_add_line(state, std::string("Canva: Loaded ISO NPZ: ") + path);
                }
                else
                {
                    console_add_line(state, std::string("Canva: ISO NPZ load failed: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
            else
            {
                console_add_line(state, "Canva: Load ISO cancelled");
            }
        }

        if (!state.sketch2iso.last_output_path.empty())
            ImGui::TextWrapped("Output: %s", state.sketch2iso.last_output_path.c_str());

        ImGui::Checkbox(
            "Continuity filter##sketch2iso_continuity_filter",
            &state.sketch2iso_continuity_filter_enabled);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Smooths the 64D ISO field within body regions while preserving hard descriptor transitions.");
        ImGui::BeginDisabled(!state.sketch2iso_continuity_filter_enabled);
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat(
            "Filter strength##sketch2iso_continuity_strength",
            &state.sketch2iso_continuity_filter_strength,
            0.0f,
            1.0f,
            "%.2f");
        state.sketch2iso_continuity_filter_strength =
            std::clamp(state.sketch2iso_continuity_filter_strength, 0.0f, 1.0f);
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat(
            "Edge threshold##sketch2iso_continuity_edge",
            &state.sketch2iso_continuity_edge_threshold,
            0.005f,
            0.15f,
            "%.3f");
        state.sketch2iso_continuity_edge_threshold =
            std::clamp(state.sketch2iso_continuity_edge_threshold, 0.001f, 0.25f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Lower values preserve more boundaries; higher values merge more descriptor variation.");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderInt(
            "Filter iterations##sketch2iso_continuity_iterations",
            &state.sketch2iso_continuity_iterations,
            1,
            24);
        state.sketch2iso_continuity_iterations =
            std::clamp(state.sketch2iso_continuity_iterations, 1, 32);
        ImGui::EndDisabled();
        if (!state.canva_manual_iso_npz_path.empty())
        {
            ImGui::TextWrapped("Loaded ISO: %s", state.canva_manual_iso_npz_path.c_str());
            ImGui::SameLine();
            if (ImGui::Button("Clear##canva_clear_loaded_iso"))
            {
                state.canva_manual_iso_npz_path.clear();
                console_add_line(state, "Canva: Cleared loaded ISO NPZ");
            }
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Rig Optimization");
        std::string rig_opt_pose_error;
        std::filesystem::path rig_opt_pose_path;
        const bool rig_opt_pose_available =
            canva_current_predicted_pose_path(state, &rig_opt_pose_path, &rig_opt_pose_error);
        std::error_code rig_opt_ec;
        const bool rig_opt_have_uv =
            !state.sketch2uv.last_output_path.empty()
            && std::filesystem::exists(std::filesystem::path(state.sketch2uv.last_output_path), rig_opt_ec);
        const bool rig_opt_can_queue_uv =
            state.sketch2uv.service != nullptr
            && (state.sketch2uv.status == ModelInferenceStatus::NotStarted
                || state.sketch2uv.status == ModelInferenceStatus::Ready);
        const bool can_optimize_rig =
            rig_opt_pose_available
            && !canva.rig_optimization_active
            && !canva.rig_optimization_job_running
            && (rig_opt_have_uv || rig_opt_can_queue_uv);

        ImGui::PushItemWidth(112.0f);
        ImGui::InputInt("Iterations##rig_opt_iterations", &canva.rig_optimization_iterations, 10, 25);
        canva.rig_optimization_iterations = std::clamp(canva.rig_optimization_iterations, 1, 1000);
        ImGui::SameLine();
        ImGui::InputInt("Render##rig_opt_render_size", &canva.rig_optimization_render_size, 16, 32);
        canva.rig_optimization_render_size = std::clamp(canva.rig_optimization_render_size, 64, 384);
        ImGui::InputFloat("LR##rig_opt_lr", &canva.rig_optimization_lr, 0.005f, 0.02f, "%.4f");
        canva.rig_optimization_lr = std::clamp(canva.rig_optimization_lr, 0.0001f, 1.0f);
        ImGui::SameLine();
        ImGui::InputFloat("Pose reg##rig_opt_pose_reg", &canva.rig_optimization_pose_reg, 0.002f, 0.01f, "%.4f");
        canva.rig_optimization_pose_reg = std::clamp(canva.rig_optimization_pose_reg, 0.0f, 1.0f);
        ImGui::InputFloat("Beta reg##rig_opt_beta_reg", &canva.rig_optimization_beta_reg, 0.002f, 0.01f, "%.4f");
        canva.rig_optimization_beta_reg = std::clamp(canva.rig_optimization_beta_reg, 0.0f, 1.0f);
        ImGui::PopItemWidth();

        ImGui::BeginDisabled(!can_optimize_rig);
        if (ImGui::Button("Rig Optimization"))
        {
            std::string err;
            if (!canva_queue_rig_optimization(state, canva, &err))
            {
                canva.rig_optimization_active = false;
                canva.rig_optimization_job_running = false;
                canva.rig_optimization_waiting_uv = false;
                canva.rig_optimization_status = "Failed";
                canva.rig_optimization_last_error = err.empty() ? "(unknown error)" : err;
                console_add_line(state, "Canva: Rig Optimization failed to queue: " + canva.rig_optimization_last_error);
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Text("Status: %s", canva.rig_optimization_status.c_str());
        if (!rig_opt_pose_available && !canva.rig_optimization_active)
            ImGui::TextWrapped("Run Extract Pose first.");
        else if (!rig_opt_have_uv && !rig_opt_can_queue_uv && !canva.rig_optimization_active)
            ImGui::TextWrapped("Sketch2UV is required for the contour mask.");
        if (canva.rig_optimization_waiting_uv)
            ImGui::TextUnformatted("Rig Optimization: predicting UV mask...");
        if (canva.rig_optimization_job_running)
            ImGui::TextUnformatted("Rig Optimization: optimizing differentiable silhouette...");
        if (canva.rig_optimization_last_final_loss > 0.0f)
            ImGui::Text("Loss: %.5f -> %.5f",
                canva.rig_optimization_last_initial_loss,
                canva.rig_optimization_last_final_loss);
        if (!canva.rig_optimization_last_output.empty())
            ImGui::TextWrapped("Output: %s", canva.rig_optimization_last_output.c_str());
        if (!canva.rig_optimization_last_error.empty())
            ImGui::TextWrapped("Error: %s", canva.rig_optimization_last_error.c_str());

        ImGui::Separator();
        const bool displacement_camera_available =
            state.scene_view_target_valid
            && state.scene_view_snapshot_completed_id > 0
            && !state.scene_view_snapshot_path.empty();
        std::filesystem::path displacement_loaded_uv_path;
        const bool displacement_have_loaded_uv =
            canva_manual_uv_source_path(state, &displacement_loaded_uv_path);
        const bool displacement_can_queue_uv =
            state.sketch2uv.service != nullptr
            && (state.sketch2uv.status == ModelInferenceStatus::NotStarted
                || state.sketch2uv.status == ModelInferenceStatus::Ready);
        const bool can_generate_displacement =
            !canva.generate_displacement_active
            && !canva.generate_iso_displacement_active
            && displacement_camera_available
            && (displacement_have_loaded_uv || displacement_can_queue_uv);
        ImGui::BeginDisabled(!can_generate_displacement);
        if (ImGui::Button("Generate Displacement"))
        {
            std::string err;
            if (!canva_queue_generate_displacement(state, canva, &err))
            {
                canva.generate_displacement_active = false;
                canva.generate_displacement_status = "Failed";
                console_add_line(state, std::string("Canva: Generate Displacement failed to queue: ")
                    + (err.empty() ? "(unknown error)" : err));
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Text("Status: %s",
            !displacement_camera_available ? "Predict camera view first"
            : !displacement_have_loaded_uv && !displacement_can_queue_uv ? "Need UV"
            : canva.generate_displacement_status.c_str());
        if (canva.generate_displacement_waiting_uv)
            ImGui::TextUnformatted("Generate Displacement: generating UV map...");
        if (canva.generate_displacement_waiting_render)
            ImGui::TextUnformatted("Generate Displacement: projecting offsets...");

        ImGui::SetNextItemWidth(140.0f);
        ImGui::InputFloat(
            "ISO distance threshold##canva_iso_distance",
            &state.iso_distance_threshold,
            0.01f,
            0.05f,
            "%.4f");
        state.iso_distance_threshold =
            std::clamp(state.iso_distance_threshold, 0.0f, 10.0f);

        std::filesystem::path displacement_loaded_iso_path;
        const bool displacement_have_loaded_iso =
            canva_manual_iso_source_path(state, &displacement_loaded_iso_path);
        const bool displacement_can_queue_iso =
            state.sketch2iso.service != nullptr
            && state.sketch2iso.status == ModelInferenceStatus::Ready;
        const bool can_generate_iso_displacement =
            !canva.generate_iso_displacement_active
            && !canva.generate_displacement_active
            && displacement_camera_available
            && (displacement_have_loaded_iso || displacement_can_queue_iso);
        ImGui::BeginDisabled(!can_generate_iso_displacement);
        if (ImGui::Button("Create ISO Displacement Map"))
        {
            std::string error;
            if (!canva_queue_generate_iso_displacement(state, canva, &error))
            {
                canva.generate_iso_displacement_active = false;
                canva.generate_iso_displacement_status = "Failed";
                console_add_line(
                    state,
                    "Canva: Create ISO Displacement Map failed to queue: "
                        + (error.empty() ? std::string("(unknown error)") : error));
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Text(
            "Status: %s",
            !displacement_camera_available ? "Predict camera view first"
            : !displacement_have_loaded_iso && !displacement_can_queue_iso ? "Need ISO"
            : canva.generate_iso_displacement_status.c_str());
        if (canva.generate_iso_displacement_waiting_iso)
            ImGui::TextUnformatted("ISO displacement: generating ISO map...");
        if (canva.generate_iso_displacement_waiting_render)
            ImGui::TextUnformatted("ISO displacement: projecting offsets...");

        ImGui::Separator();
        const char* bone_prediction_items[] = { "Custom", "NLF" };
        int preferred_bone_prediction = std::clamp(state.canva_preferred_bone_prediction, 0, 1);
        ImGui::SetNextItemWidth(160.0f);
        if (ImGui::Combo(
                "preferred bone predition",
                &preferred_bone_prediction,
                bone_prediction_items,
                (int)(sizeof(bone_prediction_items) / sizeof(bone_prediction_items[0]))))
        {
            state.canva_preferred_bone_prediction = std::clamp(preferred_bone_prediction, 0, 1);
        }
        ImVec4 bone_prediction_debug_color = ImVec4(0.72f, 0.72f, 0.72f, 1.0f);
        switch (canva.bone_prediction_debug_mode)
        {
        case 1:
            bone_prediction_debug_color = ImVec4(0.20f, 0.78f, 0.38f, 1.0f);
            break;
        case 2:
            bone_prediction_debug_color = ImVec4(0.95f, 0.62f, 0.18f, 1.0f);
            break;
        case 3:
            bone_prediction_debug_color = ImVec4(0.34f, 0.60f, 1.0f, 1.0f);
            break;
        case 4:
            bone_prediction_debug_color = ImVec4(0.95f, 0.25f, 0.22f, 1.0f);
            break;
        default:
            break;
        }
        ImGui::TextColored(
            bone_prediction_debug_color,
            "Bone prediction: %s",
            canva.bone_prediction_debug_status.empty()
                ? "Custom pipeline"
                : canva.bone_prediction_debug_status.c_str());
        if (canva.bone_prediction_nlf_confidence >= 0.0f)
        {
            ImGui::SameLine();
            ImGui::Text("(NLF %.3f)", (double)canva.bone_prediction_nlf_confidence);
        }

        const bool custom_bone_models_available =
            state.sketch2rig.service != nullptr &&
            state.sketch2betas.service != nullptr;
        const bool nlf_bone_model_ready =
            state.nlf.service != nullptr &&
            state.nlf.status == ModelInferenceStatus::Ready;
        const bool can_extract_pose =
            !canva.extract_pose_active &&
            (state.canva_preferred_bone_prediction == 0
                ? custom_bone_models_available
                : nlf_bone_model_ready);
        const bool pose_selected =
            state.selection.pose_index >= 0
            && state.selection.pose_index < (int)state.selection.poses.size()
            && state.pending_pose_selection.empty();
        const bool garment_selected =
            state.selection.garment_index >= 0
            && state.selection.garment_index < (int)state.selection.garments.size();
        ImGui::BeginDisabled(!state.pending_pose_selection.empty());
        if (ImGui::Button("Render SMPL"))
        {
            std::filesystem::path model_input_path;
            std::string err;
            if (canva_save_model_input_png(canva, &model_input_path, &err))
            {
                state.scene_view_snapshot_pose_name = pose_selected
                    ? state.selection.poses[(size_t)state.selection.pose_index]
                    : std::string();
                state.scene_view_snapshot_path.clear();
                state.scene_view_snapshot_include_cloth = false;
                state.scene_view_snapshot_use_rigopt_camera = false;
                canva.pending_scene_view_snapshot_request_id =
                    ++state.scene_view_snapshot_request_id;
                canva.pending_scene_view_snapshot_layer_enabled = true;
                canva.pending_scene_view_snapshot_include_cloth = false;
                canva.pending_scene_view_snapshot_model_input_path = model_input_path;
                state.scene_view_snap_to_target_request_id++;
                console_add_line(state, "Canva: queued SMPL render");
            }
            else
            {
                console_add_line(state, "Canva: SMPL render failed to queue: "
                    + (err.empty() ? std::string("(unknown error)") : err));
            }
        }
        ImGui::EndDisabled();
        if (!state.pending_pose_selection.empty()
            && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Wait for the selected pose to finish loading.");
        else if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Render the current SMPL character, including the Rest / T-pose, as a canvas layer.");

        ImGui::SameLine();
        ImGui::BeginDisabled(!state.pending_pose_selection.empty() || !garment_selected);
        if (ImGui::Button("Render 3D scene"))
        {
            std::filesystem::path model_input_path;
            std::string err;
            if (canva_save_model_input_png(canva, &model_input_path, &err))
            {
                state.scene_view_snapshot_pose_name = pose_selected
                    ? state.selection.poses[(size_t)state.selection.pose_index]
                    : std::string();
                state.scene_view_snapshot_path.clear();
                state.scene_view_snapshot_include_cloth = true;
                state.scene_view_snapshot_use_rigopt_camera = false;
                canva.pending_scene_view_snapshot_request_id =
                    ++state.scene_view_snapshot_request_id;
                canva.pending_scene_view_snapshot_layer_enabled = true;
                canva.pending_scene_view_snapshot_include_cloth = true;
                canva.pending_scene_view_snapshot_model_input_path = model_input_path;
                state.scene_view_snap_to_target_request_id++;
                console_add_line(state, "Canva: queued 3D scene render");
            }
            else
            {
                console_add_line(state, "Canva: 3D scene render failed to queue: "
                    + (err.empty() ? std::string("(unknown error)") : err));
            }
        }
        ImGui::EndDisabled();
        if (!garment_selected
            && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Select a cloth mesh to render the SMPL + cloth scene.");
        else if (!state.pending_pose_selection.empty()
            && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Wait for the selected pose to finish loading.");
        else if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Render the current SMPL character and cloth together as a canvas layer.");

        const bool can_render_cloth =
            garment_selected
            && !canva.render_cloth_active
            && !canva.generate_cloth_active
            && !state.controls.simulating;
        const char* cloth_render_methods[] = {
            "Standard shader",
            "Detailed ink shader",
            "Clean line shader",
            "Soft 2D shader",
            "Grease pencil",
            "Grease pencil + shading",
            "Std shader shaded",
        };
        ImGui::SetNextItemWidth(190.0f);
        ImGui::Combo(
            "Cloth render method",
            &state.cloth_render_method,
            cloth_render_methods,
            IM_ARRAYSIZE(cloth_render_methods));
        state.cloth_render_method = std::clamp(state.cloth_render_method, 0, 6);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Used for generated canvas cloth only; independent of the Scene View shader.");
        ImGui::BeginDisabled(!can_render_cloth);
        if (ImGui::Button("Render"))
        {
            std::string err;
            if (!canva_queue_render_cloth(state, canva, false, false, false, &err))
            {
                canva.render_cloth_active = false;
                canva.render_cloth_status = "Failed";
                console_add_line(state, "Canva: Render failed to queue: "
                    + (err.empty() ? std::string("(unknown error)") : err));
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        const bool can_render_and_refit = can_render_cloth && canva_has_cloth_refit_maps(canva);
        ImGui::BeginDisabled(!can_render_and_refit);
        if (ImGui::Button("Render + Refit"))
        {
            std::string err;
            if (!canva_queue_render_cloth(state, canva, false, false, true, &err))
            {
                canva.render_cloth_active = false;
                canva.render_cloth_status = "Failed";
                console_add_line(state, "Canva: Render + Refit failed to queue: "
                    + (err.empty() ? std::string("(unknown error)") : err));
            }
        }
        ImGui::EndDisabled();
        if (!can_render_and_refit && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Run Fit once to create the visibility map and editable deformation field reused by Refit.");
        ImGui::SameLine();
        ImGui::Text("Status: %s",
            !garment_selected && !canva.render_cloth_active
                ? "Select cloth mesh"
                : canva.render_cloth_status.c_str());

        ImGui::SetNextItemWidth(150.0f);
        ImGui::InputFloat(
            "Target detail size",
            &state.cloth_render_target_detail_size,
            0.001f,
            0.005f,
            "%.4f");
        state.cloth_render_target_detail_size =
            std::clamp(state.cloth_render_target_detail_size, 0.0001f, 0.25f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Maximum fine-render mesh edge length in scene units.");

        const bool cloth_fitting_models_ready =
            state.sketch2iso.service != nullptr
            && state.sketch2iso.status == ModelInferenceStatus::Ready
            && state.sketch2depth.service != nullptr
            && state.sketch2depth.status == ModelInferenceStatus::Ready;
        const bool can_simulate =
            garment_selected &&
            pose_selected &&
            !canva.extract_pose_active &&
            !canva.generate_cloth_active &&
            !canva.render_cloth_active &&
            !state.controls.simulating;
        const bool can_render_and_fit =
            can_render_cloth
            && cloth_fitting_models_ready
            && !canva.cloth_fitting_active
            && !canva.generate_displacement_active
            && !canva.generate_iso_displacement_active;

        ImGui::BeginDisabled(!can_extract_pose);
        if (ImGui::Button("Extract Pose"))
        {
            std::string err;
            if (!canva_queue_extract_pose(state, canva, false, false, &err))
                console_add_line(state, std::string("Canva: Extract Pose failed to queue: ")
                    + (err.empty() ? "(unknown error)" : err));
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!can_simulate);
        if (ImGui::Button("Simulate"))
        {
            state.show_smpl = true;
            state.show_smpl_character = true;
            state.show_garment_mesh = true;
            state.controls.current_frame = 0;
            state.controls.playing = false;
            state.controls.simulating = true;
            state.cloth_simulate_play_on_finish = false;
            state.cloth_simulate_request_id++;
            console_add_line(state, "Canva: queued XPBD simulation from the current pose");
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!can_render_and_fit);
        if (ImGui::Button("Render + Fit"))
        {
            std::string err;
            if (!canva_queue_render_cloth(state, canva, false, true, false, &err))
                console_add_line(state, "Canva: Render + Fit failed to queue: "
                    + (err.empty() ? std::string("(unknown error)") : err));
        }
        ImGui::EndDisabled();
        ImGui::SameLine();

        const bool can_pose_simu_fit = can_extract_pose && can_render_and_fit;
        ImGui::BeginDisabled(!can_pose_simu_fit);
        if (ImGui::Button("Pose+Simu+Fit"))
        {
            std::string err;
            if (canva_queue_extract_pose(state, canva, true, true, &err))
            {
                canva.generate_cloth_active = true;
                canva.generate_cloth_waiting_pose = false;
                canva.generate_cloth_waiting_simulation = false;
                canva.generate_cloth_waiting_render = false;
                canva.generate_cloth_status = "Inference";
                canva.generate_cloth_failed_step = -1;
                canva.generate_cloth_fit_after_render = true;
                canva.generate_cloth_include_fitting_steps = true;
                console_add_line(state, "Canva: Pose+Simu+Fit queued pose extraction");
            }
            else
            {
                canva_generate_cloth_reset_timings(canva, true);
                canva_generate_cloth_start_step(canva, GenerateClothStepId::Inference);
                canva_generate_cloth_finish_step(canva, GenerateClothStepId::Inference);
                canva_generate_cloth_finish_pipeline(canva);
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Failed";
                canva.generate_cloth_failed_step = 0;
                canva.generate_cloth_fit_after_render = false;
                console_add_line(state, std::string("Canva: Pose+Simu+Fit failed to queue: ")
                    + (err.empty() ? "(unknown error)" : err));
            }
        }
        ImGui::EndDisabled();
        ImGui::Text("Status: %s",
            (!garment_selected && !canva.generate_cloth_active) ? "Select cloth mesh" : canva.generate_cloth_status.c_str());
        if (canva.extract_pose_active)
            ImGui::TextUnformatted("Full Generation: waiting for rig and betas...");
        if (canva.generate_cloth_waiting_pose)
            ImGui::TextUnformatted("Full Generation: waiting for pose reload...");
        if (canva.generate_cloth_waiting_simulation)
        {
            const std::string step = state.controls.sim_stage.empty()
                ? std::string("waiting for XPBD simulation")
                : state.controls.sim_stage;
            ImGui::Text("Generate Cloth: %s...", step.c_str());
            const int phase = std::clamp(state.controls.sim_pipeline_stage, 0, 2);
            const char* phase_labels[] = { "Sequence creation", "Proxy creation", "Simulation" };
            const int phase_done[] = {
                state.controls.sim_sequence_steps_done,
                state.controls.sim_proxy_steps_done,
                state.controls.sim_solver_steps_done
            };
            const int phase_total[] = {
                state.controls.sim_sequence_steps_total,
                state.controls.sim_proxy_steps_total,
                state.controls.sim_solver_steps_total
            };
            const int done = std::max(0, phase_done[phase]);
            const int total = std::max(0, phase_total[phase]);
            const float progress = total > 0 ? (float)done / (float)total : 0.0f;
            ImGui::TextUnformatted(phase_labels[phase]);
            ImGui::ProgressBar(std::clamp(progress, 0.0f, 1.0f), ImVec2(-1.0f, 0.0f));
            if (total > 0)
                ImGui::Text("%d / %d", done, total);
        }
        if (canva.generate_cloth_waiting_render)
            ImGui::TextUnformatted("Generate Cloth: rendering cloth layer...");

        }

        if (parameters_only)
        {
        if (canva.cloth_fitting_timing_started)
        {
            ImGui::TextUnformatted("Fitting timing");
            ImGui::Text("Total: %.2fs", canva_cloth_fitting_elapsed_seconds(canva));
            const char* timing_labels[] = {
                "Sketch2ISO inference",
                "Sketch2Depth inference",
                "Depth and visibility",
                "Canvas layer creation"
            };
            for (int i = 0; i < kClothFittingTimingStepCount; ++i)
            {
                const double seconds = canva.cloth_fitting_step_seconds[(size_t)i];
                ImGui::Text("%s: %s%.2fs",
                    timing_labels[i],
                    seconds > 0.0 ? "" : (canva.cloth_fitting_active ? "running / " : "not run / "),
                    seconds);
            }
            for (const auto& timing : canva.cloth_fitting_renderer_timings)
                ImGui::Text("%s: %.2fs", timing.first.c_str(), timing.second);
            ImGui::Separator();
        }
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat("Depth margin (m)", &state.cloth_fitting_depth_margin, 0.0f, 0.02f, "%.5f");
        state.cloth_fitting_depth_margin = std::clamp(state.cloth_fitting_depth_margin, 0.0f, 0.05f);
        ImGui::SliderFloat("Depth margin for pred (m)", &state.cloth_fitting_depth_margin_pred, 0.0f, 1.0f, "%.5f");
        state.cloth_fitting_depth_margin_pred = std::clamp(state.cloth_fitting_depth_margin_pred, 0.0f, 1.0f);
        ImGui::SliderFloat(
            "Close-to-body threshold",
            &state.cloth_fitting_body_distance_threshold,
            0.0f,
            0.1f,
            "%.4f");
        state.cloth_fitting_body_distance_threshold =
            std::clamp(state.cloth_fitting_body_distance_threshold, 0.0f, 1.0f);
        ImGui::SliderFloat(
            "Liquify radius",
            &state.cloth_fitting_warp_radius,
            0.0f,
            256.0f,
            "%.1f px");
        state.cloth_fitting_warp_radius =
            std::clamp(state.cloth_fitting_warp_radius, 0.0f, 512.0f);
        ImGui::SliderFloat(
            "Liquify force",
            &state.cloth_fitting_warp_force,
            0.0f,
            6.0f,
            "%.3f");
        state.cloth_fitting_warp_force =
            std::clamp(state.cloth_fitting_warp_force, 0.0f, 6.0f);
        ImGui::SetNextItemWidth(170.0f);
        ImGui::InputInt(
            "Liquify steps",
            &state.cloth_fitting_warp_steps,
            1,
            4);
        state.cloth_fitting_warp_steps =
            std::clamp(state.cloth_fitting_warp_steps, 1, 32);

        ImGui::Separator();
        ImGui::TextUnformatted("Fitting color-map filter");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderInt("Shape smoothing radius", &state.cloth_fitting_debug_smoothing_radius, 0, 8, "%d px");
        state.cloth_fitting_debug_smoothing_radius =
            std::clamp(state.cloth_fitting_debug_smoothing_radius, 0, 8);
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderInt("Shape smoothing passes", &state.cloth_fitting_debug_smoothing_passes, 0, 8);
        state.cloth_fitting_debug_smoothing_passes =
            std::clamp(state.cloth_fitting_debug_smoothing_passes, 0, 8);
        ImGui::SetNextItemWidth(170.0f);
        ImGui::InputInt("Small color patch area", &state.cloth_fitting_debug_min_patch_area, 1, 16);
        state.cloth_fitting_debug_min_patch_area =
            std::clamp(state.cloth_fitting_debug_min_patch_area, 1, 4096);
        ImGui::SetNextItemWidth(170.0f);
        ImGui::InputInt("Invalid match fill radius", &state.cloth_fitting_debug_error_fill_radius, 8, 32);
        state.cloth_fitting_debug_error_fill_radius =
            std::clamp(state.cloth_fitting_debug_error_fill_radius, 0, 4096);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Maximum distance to a valid nearby color; 0 fills from the nearest color at any distance.");

        ImGui::Separator();
        ImGui::TextUnformatted("True-depth correction");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderInt(
            "Maximum correction shift",
            &state.cloth_fitting_true_depth_correction_max_shift,
            0,
            512,
            "%d px");
        state.cloth_fitting_true_depth_correction_max_shift = std::clamp(
            state.cloth_fitting_true_depth_correction_max_shift,
            0,
            1024);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Maximum screen-space translation searched for each true-depth red component.");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderInt(
            "ISO support radius",
            &state.cloth_fitting_true_depth_correction_support_radius,
            0,
            64,
            "%d px");
        state.cloth_fitting_true_depth_correction_support_radius = std::clamp(
            state.cloth_fitting_true_depth_correction_support_radius,
            0,
            64);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Distance around ISO red pixels considered useful alignment support.");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat(
            "Minimum ISO overlap",
            &state.cloth_fitting_true_depth_correction_min_overlap,
            0.0f,
            0.5f,
            "%.3f");
        state.cloth_fitting_true_depth_correction_min_overlap = std::clamp(
            state.cloth_fitting_true_depth_correction_min_overlap,
            0.0f,
            1.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Minimum fraction of a shifted true-depth component supported by ISO matching.");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat(
            "Minimum alignment gain",
            &state.cloth_fitting_true_depth_correction_min_improvement,
            0.0f,
            0.25f,
            "%.3f");
        state.cloth_fitting_true_depth_correction_min_improvement = std::clamp(
            state.cloth_fitting_true_depth_correction_min_improvement,
            0.0f,
            1.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Minimum ISO alignment-score improvement required to move a component.");

        ImGui::Separator();
        ImGui::TextUnformatted("Final occlusion cleanup");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderInt(
            "Cleanup radius",
            &state.cloth_fitting_occlusion_cleanup_radius,
            1,
            32,
            "%d px");
        state.cloth_fitting_occlusion_cleanup_radius =
            std::clamp(state.cloth_fitting_occlusion_cleanup_radius, 1, 32);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Neighborhood used to recognize and regularize noisy occlusion contours.");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderInt(
            "Cleanup passes",
            &state.cloth_fitting_occlusion_cleanup_passes,
            0,
            8);
        state.cloth_fitting_occlusion_cleanup_passes =
            std::clamp(state.cloth_fitting_occlusion_cleanup_passes, 0, 8);
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat(
            "Chaos threshold",
            &state.cloth_fitting_occlusion_chaos_threshold,
            0.01f,
            0.5f,
            "%.3f");
        state.cloth_fitting_occlusion_chaos_threshold =
            std::clamp(state.cloth_fitting_occlusion_chaos_threshold, 0.01f, 0.5f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Lower values clean more of the contour; higher values preserve more existing corners.");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::InputInt(
            "Minimum patch area",
            &state.cloth_fitting_occlusion_min_patch_area,
            1,
            16);
        state.cloth_fitting_occlusion_min_patch_area =
            std::clamp(state.cloth_fitting_occlusion_min_patch_area, 1, 4096);

        ImGui::Separator();
        ImGui::TextUnformatted("Final cloth stylization");
        const char* stylization_methods[] = { "Anime cel", "Soft toon", "Pure line art", "None" };
        ImGui::SetNextItemWidth(170.0f);
        ImGui::Combo(
            "Method",
            &state.cloth_stylization_method,
            stylization_methods,
            IM_ARRAYSIZE(stylization_methods));
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat(
            "Outline width",
            &state.cloth_stylization_outline_width,
            0.0f,
            12.0f,
            "%.1f px");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat(
            "Wrinkle line width",
            &state.cloth_stylization_wrinkle_width,
            0.0f,
            8.0f,
            "%.1f px");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat(
            "Wrinkle sensitivity",
            &state.cloth_stylization_edge_threshold,
            0.01f,
            0.25f,
            "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Lower values draw more fold and shading boundaries as ink lines.");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderInt(
            "Contour cleanup",
            &state.cloth_stylization_contour_radius,
            1,
            24,
            "%d px");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderInt(
            "Line gap closing",
            &state.cloth_stylization_line_gap_radius,
            0,
            6,
            "%d px");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::ColorEdit3(
            "Lineart color",
            state.cloth_stylization_line_color.data(),
            ImGuiColorEditFlags_NoInputs);
        if (state.cloth_stylization_method != 2)
        {
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderFloat(
                "Shadow threshold",
                &state.cloth_stylization_shadow_threshold,
                0.5f,
                1.0f,
                "%.2f");
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderFloat(
                "Shadow strength",
                &state.cloth_stylization_shadow_strength,
                0.0f,
                0.8f,
                "%.2f");
        }
        if (state.cloth_stylization_method == 1)
        {
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderFloat(
                "Shading softness",
                &state.cloth_stylization_softness,
                0.0f,
                1.0f,
                "%.2f");
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Generated cloth Grease Pencil");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat("GP outline width", &state.cloth_grease_pencil_outline_width, 0.0f, 12.0f, "%.1f px");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat("GP wrinkle width", &state.cloth_grease_pencil_wrinkle_width, 0.0f, 8.0f, "%.1f px");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat("GP crease angle", &state.cloth_grease_pencil_crease_angle, 10.0f, 80.0f, "%.0f deg");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Minimum face-normal angle used by Line Art to identify a wrinkle.");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat("GP shadow start", &state.cloth_grease_pencil_shadow_threshold, 0.35f, 0.95f, "%.2f");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat("GP shadow falloff", &state.cloth_grease_pencil_shadow_range, 0.05f, 0.70f, "%.2f");
        ImGui::SetNextItemWidth(170.0f);
        ImGui::SliderFloat("GP shadow strength", &state.cloth_grease_pencil_shadow_strength, 0.0f, 1.0f, "%.2f");
        }

        if (!parameters_only)
        {

        if (ImGui::Button("Add layer"))
        {
            show_canva_layers();
            const std::string name = std::string("Layer ") + std::to_string(canva.next_layer_number++);
            canva.layers.insert(canva.layers.begin(),
                canva_make_layer(canva.width, canva.height, canva.next_layer_id++, name));
            canva.active_layer = 0;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(canva.layers.empty());
        if (ImGui::Button("Remove layer"))
        {
            const int idx = canva.active_layer;
            if (idx >= 0 && idx < (int)canva.layers.size())
            {
                show_canva_layers();
                const int removed_id = canva.layers[(size_t)idx].id;
                canva_delete_layer_texture(canva.layers[(size_t)idx]);
                canva.layers.erase(canva.layers.begin() + idx);
                canva_prune_undo_for_layer(canva.undo, removed_id);
                canva.active_layer = std::clamp(canva.active_layer, 0, std::max(0, (int)canva.layers.size() - 1));
                canva_cancel_interactions(canva);
            }
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(canva.layers.size() <= 1);
        if (ImGui::Button("Up") && canva.active_layer > 0)
        {
            show_canva_layers();
            std::swap(canva.layers[(size_t)canva.active_layer], canva.layers[(size_t)canva.active_layer - 1]);
            canva.active_layer--;
        }
        ImGui::SameLine();
        if (ImGui::Button("Down") && canva.active_layer + 1 < (int)canva.layers.size())
        {
            show_canva_layers();
            std::swap(canva.layers[(size_t)canva.active_layer], canva.layers[(size_t)canva.active_layer + 1]);
            canva.active_layer++;
        }
        ImGui::EndDisabled();

        ImGui::Separator();

        const auto layer_icon = [](const CanvaLayer& layer)
        {
            if (canva_layer_is_visibility_map(layer))
                return std::pair<const char*, ImVec4>("V", ImVec4(0.30f, 0.85f, 0.38f, 1.0f));
            if (layer.kind == CanvaLayerKind::Scene)
                return std::pair<const char*, ImVec4>("3D", ImVec4(0.42f, 0.78f, 0.95f, 1.0f));
            if (layer.kind == CanvaLayerKind::ClothFittingDeformation
                && !layer.deformation_source_x.empty()
                && !layer.deformation_source_y.empty())
                return std::pair<const char*, ImVec4>("F", ImVec4(0.30f, 0.78f, 0.92f, 1.0f));
            if (layer.kind == CanvaLayerKind::Generic)
                return std::pair<const char*, ImVec4>("S", ImVec4(0.35f, 0.68f, 1.0f, 1.0f));
            if (layer.kind == CanvaLayerKind::GeneratedCloth
                || layer.kind == CanvaLayerKind::ClothFitting
                || layer.kind == CanvaLayerKind::ClothFittingWarped
                || layer.kind == CanvaLayerKind::ClothFittingWarpedSmoothed)
            {
                return std::pair<const char*, ImVec4>("C", ImVec4(0.93f, 0.67f, 0.28f, 1.0f));
            }
            return std::pair<const char*, ImVec4>("D", ImVec4(0.72f, 0.52f, 0.92f, 1.0f));
        };

        // Layer list (top->bottom).
        for (int i = 0; i < (int)canva.layers.size(); ++i)
        {
            CanvaLayer& layer = canva.layers[(size_t)i];

            ImGui::PushID(i);
            if (ImGui::Checkbox("##vis", &layer.enabled))
                show_canva_layers();
            ImGui::SameLine();

            const auto [icon, icon_color] = layer_icon(layer);
            ImGui::TextColored(icon_color, "[%s]", icon);
            if (ImGui::IsItemHovered())
            {
                const char* nature = icon[0] == 'S' ? "Sketch"
                    : icon[0] == '3' ? "3D scene"
                    : icon[0] == 'V' ? "Visibility map"
                    : icon[0] == 'F' ? "Deformation field"
                    : icon[0] == 'C' ? "Cloth render"
                    : "Debug";
                ImGui::SetTooltip("%s layer", nature);
            }
            ImGui::SameLine();

            const bool selected = (canva.active_layer == i);
            if (ImGui::Selectable(layer.name.c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick))
            {
                show_canva_layers();
                canva.active_layer = i;
            }

            // Drag-drop reorder (swap-based).
            if (ImGui::BeginDragDropSource())
            {
                ImGui::SetDragDropPayload("CANVA_LAYER", &i, sizeof(int));
                ImGui::TextUnformatted(layer.name.c_str());
                ImGui::EndDragDropSource();
            }
            if (ImGui::BeginDragDropTarget())
            {
                if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("CANVA_LAYER"))
                {
                    if (payload->DataSize == sizeof(int))
                    {
                        const int from = *(const int*)payload->Data;
                        const int to = i;
                        if (from >= 0 && from < (int)canva.layers.size() && to >= 0 && to < (int)canva.layers.size() && from != to)
                        {
                            show_canva_layers();
                            std::swap(canva.layers[(size_t)from], canva.layers[(size_t)to]);
                            if (canva.active_layer == from)
                                canva.active_layer = to;
                            else if (canva.active_layer == to)
                                canva.active_layer = from;
                        }
                    }
                }
                ImGui::EndDragDropTarget();
            }

            ImGui::PopID();
        }

        if (canva.active_layer >= 0 && canva.active_layer < (int)canva.layers.size())
        {
            CanvaLayer& layer = canva.layers[(size_t)canva.active_layer];
            ImGui::Separator();
            ImGui::SetNextItemWidth(170.0f);
            float opacity_percent = std::clamp(layer.opacity, 0.0f, 1.0f) * 100.0f;
            if (ImGui::SliderFloat("Opacity", &opacity_percent, 0.0f, 100.0f, "%.0f%%",
                    ImGuiSliderFlags_AlwaysClamp))
            {
                layer.opacity = opacity_percent / 100.0f;
                show_canva_layers();
            }

            if (layer.kind == CanvaLayerKind::ClothFittingWarpedSmoothed)
            {
                ImGui::TextUnformatted("Warped cloth stylization");
                bool smoothing_changed = false;
                const char* layer_methods[] = { "Anime cel", "Soft toon", "Pure line art", "None" };
                smoothing_changed |= ImGui::Combo(
                    "Method##stylized_layer",
                    &layer.stylization_method,
                    layer_methods,
                    IM_ARRAYSIZE(layer_methods));
                ImGui::SliderFloat(
                    "Outline width##stylized_layer",
                    &layer.stylization_outline_width,
                    0.0f,
                    12.0f,
                    "%.1f px");
                smoothing_changed |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SliderFloat(
                    "Wrinkle line width##stylized_layer",
                    &layer.stylization_wrinkle_width,
                    0.0f,
                    8.0f,
                    "%.1f px");
                smoothing_changed |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SliderFloat(
                    "Wrinkle sensitivity##stylized_layer",
                    &layer.stylization_edge_threshold,
                    0.01f,
                    0.25f,
                    "%.3f");
                smoothing_changed |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SliderInt(
                    "Contour cleanup##stylized_layer",
                    &layer.stylization_contour_radius,
                    1,
                    24,
                    "%d px");
                smoothing_changed |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SliderInt(
                    "Line gap closing##stylized_layer",
                    &layer.stylization_line_gap_radius,
                    0,
                    6,
                    "%d px");
                smoothing_changed |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SetNextItemWidth(170.0f);
                smoothing_changed |= ImGui::ColorEdit3(
                    "Lineart color##stylized_layer",
                    layer.stylization_line_color.data(),
                    ImGuiColorEditFlags_NoInputs);
                if (layer.stylization_method != 2)
                {
                    ImGui::SliderFloat(
                        "Shadow threshold##stylized_layer",
                        &layer.stylization_shadow_threshold,
                        0.5f,
                        1.0f,
                        "%.2f");
                    smoothing_changed |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::SliderFloat(
                        "Shadow strength##stylized_layer",
                        &layer.stylization_shadow_strength,
                        0.0f,
                        0.8f,
                        "%.2f");
                    smoothing_changed |= ImGui::IsItemDeactivatedAfterEdit();
                }
                if (layer.stylization_method == 1)
                {
                    ImGui::SliderFloat(
                        "Shading softness##stylized_layer",
                        &layer.stylization_softness,
                        0.0f,
                        1.0f,
                        "%.2f");
                    smoothing_changed |= ImGui::IsItemDeactivatedAfterEdit();
                }
                if (smoothing_changed)
                {
                    state.cloth_stylization_method = layer.stylization_method;
                    state.cloth_stylization_outline_width = layer.stylization_outline_width;
                    state.cloth_stylization_wrinkle_width = layer.stylization_wrinkle_width;
                    state.cloth_stylization_edge_threshold = layer.stylization_edge_threshold;
                    state.cloth_stylization_shadow_threshold = layer.stylization_shadow_threshold;
                    state.cloth_stylization_shadow_strength = layer.stylization_shadow_strength;
                    state.cloth_stylization_softness = layer.stylization_softness;
                    state.cloth_stylization_contour_radius = layer.stylization_contour_radius;
                    state.cloth_stylization_line_gap_radius = layer.stylization_line_gap_radius;
                    state.cloth_stylization_line_color = layer.stylization_line_color;
                }
                if (smoothing_changed)
                {
                    const int source_index =
                        canva_find_layer_index_by_id(canva, layer.smoothing_source_layer_id);
                    if (source_index >= 0)
                    {
                        std::string smoothing_error;
                        if (!canva_stylize_warped_cloth_layer(
                                canva.layers[(size_t)source_index],
                                layer,
                                canva.width,
                                canva.height,
                                &smoothing_error))
                        {
                            console_add_line(
                                state,
                                "Canva: cloth smoothing failed: "
                                    + (smoothing_error.empty()
                                        ? std::string("(unknown error)")
                                        : smoothing_error));
                        }
                    }
                }
            }
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Controls:");
        ImGui::BulletText("Brush tool: left mouse");
        ImGui::BulletText("Brush size: Shift + drag (on canvas)");
        ImGui::BulletText("Transform tool: Ctrl+T, drag layer (or corner handles to scale)");
        ImGui::BulletText("Undo: Ctrl+Z");
        ImGui::BulletText("Pan: middle mouse drag");
        ImGui::BulletText("Zoom: mouse wheel");
        }
    }
    ImGui::EndChild();
    }
}
