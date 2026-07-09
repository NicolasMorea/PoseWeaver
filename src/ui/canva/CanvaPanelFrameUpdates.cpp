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
    void update_canva_frame(AppState& state, CanvaState& canva, Renderer& renderer)
    {
    const auto discard_reset_result = [](ModelInferenceUiState& ui, std::uint64_t& ignored_request_id)
    {
        if (ignored_request_id == 0 || ui.completed_request_id != ignored_request_id)
            return;
        ui.last_sketch_path.clear();
        ui.last_output_path.clear();
        ignored_request_id = 0;
    };
    discard_reset_result(state.sketch2rig, canva.ignore_sketch2rig_request_id);
    discard_reset_result(state.sketch2betas, canva.ignore_sketch2betas_request_id);
    discard_reset_result(state.sketch2uv, canva.ignore_sketch2uv_request_id);
    discard_reset_result(state.sketch2iso, canva.ignore_sketch2iso_request_id);
    discard_reset_result(state.sketch2depth, canva.ignore_sketch2depth_request_id);
    discard_reset_result(state.vitpose, canva.ignore_vitpose_request_id);

    // Consume undo requests routed from the GLFW key callback.
    while (canva.last_undo_request_id < state.canva_undo_request_id)
    {
        canva.last_undo_request_id++;
        console_add_line(state, "Canva: Ctrl+Z consumed (GLFW request)");
        const bool ok = canva_apply_undo(canva);
        console_add_line(state, ok ? "Canva: Undo applied" : "Canva: Undo stack empty");
    }

    if (state.scene_view_snapshot_completed_id > canva.last_scene_view_snapshot_completed_id)
    {
        canva.last_scene_view_snapshot_completed_id = state.scene_view_snapshot_completed_id;
        if (canva.pending_scene_view_snapshot_request_id == 0
            || state.scene_view_snapshot_completed_id >= canva.pending_scene_view_snapshot_request_id)
        {
            if (!state.scene_view_snapshot_path.empty())
            {
                std::string err;
                if (canva_add_smpl_render_layer(
                        canva,
                        state.scene_view_snapshot_path,
                        canva.pending_scene_view_snapshot_model_input_path,
                        &err))
                {
                    console_add_line(state, "Canva: Added SMPL layer");
                    canva.pending_scene_view_snapshot_request_id = 0;
                    canva.pending_scene_view_snapshot_model_input_path.clear();
                }
                else
                {
                    console_add_line(state, std::string("Canva: Failed to add SMPL layer: ")
                        + (err.empty() ? "(unknown error)" : err));
                    canva.pending_scene_view_snapshot_model_input_path.clear();
                }
            }
        }
    }

    if (state.vitpose.completed_request_id > canva.last_vitpose_import_request_id)
    {
        canva.last_vitpose_import_request_id = state.vitpose.completed_request_id;
        if (state.vitpose.status == ModelInferenceStatus::Ready
            && !state.vitpose.last_output_path.empty())
        {
            std::string error;
            if (canva_add_vitpose_layer(
                    canva,
                    state.vitpose.last_output_path,
                    state.vitpose.last_sketch_path,
                    &error))
            {
                console_add_line(state, "Canva: Added VitPose 2D pose layer from "
                    + state.vitpose.last_output_path);
            }
            else
            {
                console_add_line(state, "Canva: Failed to load VitPose result: "
                    + (error.empty() ? std::string("(unknown error)") : error));
            }
        }
    }

    if (state.sketch2depth.completed_request_id > canva.last_sketch2depth_import_request_id)
    {
        canva.last_sketch2depth_import_request_id = state.sketch2depth.completed_request_id;
        const bool is_cloth_fitting_depth =
            canva.cloth_fitting_active
            && canva.cloth_fitting_waiting_depth
            && state.sketch2depth.completed_request_id == canva.cloth_fitting_depth_request_id;
        if (state.sketch2depth.status == ModelInferenceStatus::Ready
            && !state.sketch2depth.last_output_path.empty())
        {
            if (is_cloth_fitting_depth)
            {
                if (canva.generate_cloth_timing_started && canva.generate_cloth_include_fitting_steps)
                    canva_generate_cloth_append_timed_detail(
                        canva,
                        GenerateClothStepId::FittingInference,
                        "Sketch2Depth",
                        canva.cloth_fitting_depth_start);
                canva.cloth_fitting_waiting_depth = false;
                canva.cloth_fitting_depth_path = state.sketch2depth.last_output_path;
                console_add_line(state, "Canva: Body depth ready for cloth fitting from "
                    + state.sketch2depth.last_output_path);
                canva_finish_cloth_fitting_if_ready(state, canva, renderer);
            }
            else
            {
                std::string error;
                if (canva_add_sketch2depth_layer(
                        canva,
                        state.sketch2depth.last_output_path,
                        state.sketch2depth.last_sketch_path,
                        &error))
                {
                    console_add_line(state, "Canva: Added Sketch2Depth prediction layer from "
                        + state.sketch2depth.last_output_path);
                }
                else
                {
                    console_add_line(state, "Canva: Failed to load Sketch2Depth result: "
                        + (error.empty() ? std::string("(unknown error)") : error));
                }
            }
        }
        else if (is_cloth_fitting_depth)
        {
            canva.cloth_fitting_active = false;
            canva.cloth_fitting_waiting_depth = false;
            canva.cloth_fitting_waiting_iso = false;
            canva.cloth_fitting_status = "Failed";
            if (canva.generate_cloth_timing_started && canva.generate_cloth_include_fitting_steps)
            {
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Failed";
                canva.generate_cloth_failed_step = generate_cloth_step_index(GenerateClothStepId::FittingInference);
                canva_generate_cloth_finish_step(canva, GenerateClothStepId::FittingInference);
                canva_generate_cloth_finish_pipeline(canva);
            }
            console_add_line(state, "Canva: cloth Fitting cancelled because Sketch2Depth failed");
        }
    }

    if (state.sketch2rig.completed_request_id > canva.last_sketch2rig_import_request_id)
    {
        canva.last_sketch2rig_import_request_id = state.sketch2rig.completed_request_id;
        if (canva.extract_pose_active
            && state.sketch2rig.completed_request_id == canva.extract_pose_rig_request_id)
        {
            if (canva.generate_cloth_timing_started)
                canva_generate_cloth_append_timed_detail(
                    canva,
                    GenerateClothStepId::Inference,
                    "Sketch2Rig",
                    canva.generate_cloth_extract_rig_start);
            canva.extract_pose_rig_done =
                state.sketch2rig.status == ModelInferenceStatus::Ready
                && !state.sketch2rig.last_output_path.empty();
            if (!canva.extract_pose_rig_done)
            {
                console_add_line(state, "Canva: Extract Pose cancelled because rig extraction failed");
                canva.extract_pose_active = false;
                canva.extract_pose_hide_rig_layer = false;
                canva.extract_pose_generate_cloth_after = false;
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Failed";
                canva.generate_cloth_failed_step = 0;
                canva_generate_cloth_finish_step(canva, GenerateClothStepId::Inference);
                canva_generate_cloth_finish_pipeline(canva);
            }
        }
        if (state.sketch2rig.status == ModelInferenceStatus::Ready && !state.sketch2rig.last_output_path.empty())
        {
            std::string err;
            const bool hide_rig_layer =
                canva.extract_pose_active
                && canva.extract_pose_hide_rig_layer
                && state.sketch2rig.completed_request_id == canva.extract_pose_rig_request_id;
            if (canva_add_sketch2rig_layer(
                    canva,
                    state.sketch2rig.last_output_path,
                    state.sketch2rig.last_sketch_path,
                    !hide_rig_layer,
                    &err))
            {
                console_add_line(state, std::string(hide_rig_layer
                        ? "Canva: Added hidden Sketch2Rig rig layer from "
                        : "Canva: Added Sketch2Rig rig layer from ")
                    + state.sketch2rig.last_output_path);
            }
            else
            {
                console_add_line(state, std::string("Canva: Failed to load Sketch2Rig rig: ")
                    + (err.empty() ? "(unknown error)" : err));
            }
        }
    }

    if (state.sketch2betas.completed_request_id > canva.last_sketch2betas_import_request_id)
    {
        canva.last_sketch2betas_import_request_id = state.sketch2betas.completed_request_id;
        if (canva.extract_pose_active
            && state.sketch2betas.completed_request_id == canva.extract_pose_betas_request_id)
        {
            if (canva.generate_cloth_timing_started)
                canva_generate_cloth_append_timed_detail(
                    canva,
                    GenerateClothStepId::Inference,
                    "Sketch2Betas",
                    canva.generate_cloth_extract_betas_start);
            canva.extract_pose_betas_done =
                state.sketch2betas.status == ModelInferenceStatus::Ready
                && !state.sketch2betas.last_output_path.empty();
            if (!canva.extract_pose_betas_done)
            {
                console_add_line(state, "Canva: Extract Pose cancelled because betas extraction failed");
                canva.extract_pose_active = false;
                canva.extract_pose_hide_rig_layer = false;
                canva.extract_pose_generate_cloth_after = false;
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Failed";
                canva.generate_cloth_failed_step = 0;
                canva_generate_cloth_finish_step(canva, GenerateClothStepId::Inference);
                canva_generate_cloth_finish_pipeline(canva);
            }
        }
        if (state.sketch2betas.status == ModelInferenceStatus::Ready && !state.sketch2betas.last_output_path.empty())
        {
            std::string err;
            if (canva_apply_sketch2betas_result(state, state.sketch2betas.last_output_path, &err))
            {
                console_add_line(state, std::string("Canva: Applied Sketch2Betas shape from ")
                    + state.sketch2betas.last_output_path);
            }
            else
            {
                console_add_line(state, std::string("Canva: Failed to apply Sketch2Betas result: ")
                    + (err.empty() ? "(unknown error)" : err));
            }
        }
    }

    if (state.sketch2uv.completed_request_id > canva.last_sketch2uv_import_request_id)
    {
        canva.last_sketch2uv_import_request_id = state.sketch2uv.completed_request_id;
        const bool is_displacement_uv =
            canva.generate_displacement_active
            && canva.generate_displacement_waiting_uv
            && state.sketch2uv.completed_request_id == canva.generate_displacement_uv_request_id;
        const bool is_rig_optimization_uv =
            canva.rig_optimization_active
            && canva.rig_optimization_waiting_uv
            && state.sketch2uv.completed_request_id == canva.rig_optimization_uv_request_id;
        const bool is_internal_uv = is_displacement_uv || is_rig_optimization_uv;

        if (state.sketch2uv.status == ModelInferenceStatus::Ready && !state.sketch2uv.last_output_path.empty())
        {
            std::string err;
            const bool added_uv = canva_add_sketch2uv_layer(
                canva,
                state.sketch2uv.last_output_path,
                state.sketch2uv.last_sketch_path,
                !is_internal_uv,
                is_internal_uv ? "Sketch2UV (hidden)" : "Sketch2UV",
                &err);
            if (added_uv)
            {
                console_add_line(state, std::string(is_internal_uv
                        ? "Canva: Added hidden Sketch2UV layer from "
                        : "Canva: Added Sketch2UV layer from ")
                    + state.sketch2uv.last_output_path);

                if (is_displacement_uv)
                {
                    canva.generate_displacement_waiting_uv = false;
                    canva.generate_displacement_waiting_render = true;
                    canva.generate_displacement_status = "Projecting displacement";
                    canva.generate_displacement_uv_json_path = state.sketch2uv.last_output_path;
                    state.canva_displacement_uv_json_path = state.sketch2uv.last_output_path;
                    state.canva_displacement_model_input_path = canva.generate_displacement_model_input_path.string();
                    state.canva_displacement_path.clear();
                    canva.generate_displacement_render_request_id = ++state.canva_displacement_request_id;
                    console_add_line(state, "Canva: Generate Displacement queued renderer projection");
                }
                if (is_rig_optimization_uv)
                {
                    canva.rig_optimization_waiting_uv = false;
                    canva.rig_optimization_status = "Starting optimizer";
                    std::string opt_err;
                    if (!canva_start_rig_optimization(
                            state,
                            canva,
                            std::filesystem::path(state.sketch2uv.last_output_path),
                            canva.rig_optimization_model_input_path,
                            &opt_err))
                    {
                        canva.rig_optimization_active = false;
                        canva.rig_optimization_job_running = false;
                        canva.rig_optimization_status = "Failed";
                        canva.rig_optimization_last_error = opt_err.empty() ? "failed to start optimizer" : opt_err;
                        console_add_line(state, "Canva: Rig Optimization failed to start: " + canva.rig_optimization_last_error);
                    }
                }
            }
            else
            {
                console_add_line(state, std::string("Canva: Failed to load Sketch2UV result: ")
                    + (err.empty() ? "(unknown error)" : err));
                if (is_displacement_uv)
                {
                    canva.generate_displacement_active = false;
                    canva.generate_displacement_waiting_uv = false;
                    canva.generate_displacement_status = "Failed";
                }
                if (is_rig_optimization_uv)
                {
                    canva.rig_optimization_active = false;
                    canva.rig_optimization_waiting_uv = false;
                    canva.rig_optimization_status = "Failed";
                    canva.rig_optimization_last_error = err.empty() ? "failed to load Sketch2UV result" : err;
                }
            }
        }
        else if (is_displacement_uv)
        {
            canva.generate_displacement_active = false;
            canva.generate_displacement_waiting_uv = false;
            canva.generate_displacement_status = "Failed";
            console_add_line(state, "Canva: Generate Displacement cancelled because Sketch2UV failed");
        }
        else if (is_rig_optimization_uv)
        {
            canva.rig_optimization_active = false;
            canva.rig_optimization_waiting_uv = false;
            canva.rig_optimization_status = "Failed";
            canva.rig_optimization_last_error = "Sketch2UV failed";
            console_add_line(state, "Canva: Rig Optimization cancelled because Sketch2UV failed");
        }
    }

    if (state.sketch2iso.completed_request_id > canva.last_sketch2iso_import_request_id)
    {
        canva.last_sketch2iso_import_request_id = state.sketch2iso.completed_request_id;
        const bool is_iso_displacement =
            canva.generate_iso_displacement_active
            && canva.generate_iso_displacement_waiting_iso
            && state.sketch2iso.completed_request_id
                == canva.generate_iso_displacement_iso_request_id;
        const bool is_cloth_fitting_iso =
            canva.cloth_fitting_active
            && canva.cloth_fitting_waiting_iso
            && state.sketch2iso.completed_request_id
                == canva.cloth_fitting_iso_request_id;
        const bool is_internal_iso = is_iso_displacement || is_cloth_fitting_iso;
        if (state.sketch2iso.status == ModelInferenceStatus::Ready
            && !state.sketch2iso.last_output_path.empty())
        {
            std::string err;
            if (canva_add_iso_debug_layer(
                    canva,
                    state.sketch2iso.last_output_path,
                    state.sketch2iso.last_sketch_path,
                    is_internal_iso ? "Sketch2ISO (hidden)" : "Sketch2ISO Debug",
                    !is_internal_iso,
                    &err))
            {
                console_add_line(
                    state,
                    std::string(is_internal_iso
                            ? "Canva: Added hidden Sketch2ISO layer from "
                            : "Canva: Added Sketch2ISO debug layer from ")
                        + state.sketch2iso.last_output_path);
                if (is_iso_displacement)
                {
                    canva.generate_iso_displacement_waiting_iso = false;
                    canva.generate_iso_displacement_waiting_render = true;
                    canva.generate_iso_displacement_status = "Projecting ISO displacement";
                    canva.generate_iso_displacement_npz_path =
                        state.sketch2iso.last_output_path;
                    state.canva_iso_displacement_npz_path =
                        state.sketch2iso.last_output_path;
                    state.canva_iso_displacement_model_input_path =
                        canva.generate_iso_displacement_model_input_path.string();
                    state.canva_iso_displacement_path.clear();
                    canva.generate_iso_displacement_render_request_id =
                        ++state.canva_iso_displacement_request_id;
                    console_add_line(
                        state,
                        "Canva: Create ISO Displacement Map queued renderer projection");
                }
                if (is_cloth_fitting_iso)
                {
                    if (canva.generate_cloth_timing_started && canva.generate_cloth_include_fitting_steps)
                        canva_generate_cloth_append_timed_detail(
                            canva,
                            GenerateClothStepId::FittingInference,
                            "Sketch2ISO",
                            canva.cloth_fitting_iso_start);
                    canva.cloth_fitting_waiting_iso = false;
                    canva.cloth_fitting_iso_path = state.sketch2iso.last_output_path;
                    canva.cloth_fitting_status = canva.cloth_fitting_waiting_depth
                        ? "Waiting for depth" : "Depth testing";
                    canva_finish_cloth_fitting_if_ready(state, canva, renderer);
                }
            }
            else
            {
                console_add_line(state, std::string("Canva: Failed to load Sketch2ISO result: ")
                    + (err.empty() ? "(unknown error)" : err));
                if (is_iso_displacement)
                {
                    canva.generate_iso_displacement_active = false;
                    canva.generate_iso_displacement_waiting_iso = false;
                    canva.generate_iso_displacement_status = "Failed";
                }
                if (is_cloth_fitting_iso)
                {
                    canva.cloth_fitting_active = false;
                    canva.cloth_fitting_waiting_iso = false;
                    canva.cloth_fitting_waiting_depth = false;
                    canva.cloth_fitting_status = "Failed";
                    if (canva.generate_cloth_timing_started && canva.generate_cloth_include_fitting_steps)
                    {
                        canva.generate_cloth_active = false;
                        canva.generate_cloth_status = "Failed";
                        canva.generate_cloth_failed_step = generate_cloth_step_index(GenerateClothStepId::FittingInference);
                        canva_generate_cloth_finish_step(canva, GenerateClothStepId::FittingInference);
                        canva_generate_cloth_finish_pipeline(canva);
                    }
                }
            }
        }
        else if (is_iso_displacement || is_cloth_fitting_iso)
        {
            if (is_iso_displacement)
            {
                canva.generate_iso_displacement_active = false;
                canva.generate_iso_displacement_waiting_iso = false;
                canva.generate_iso_displacement_status = "Failed";
                console_add_line(
                    state,
                    "Canva: Create ISO Displacement Map cancelled because Sketch2ISO failed");
            }
            if (is_cloth_fitting_iso)
            {
                canva.cloth_fitting_active = false;
                canva.cloth_fitting_waiting_iso = false;
                canva.cloth_fitting_waiting_depth = false;
                canva.cloth_fitting_status = "Failed";
                if (canva.generate_cloth_timing_started && canva.generate_cloth_include_fitting_steps)
                {
                    canva.generate_cloth_active = false;
                    canva.generate_cloth_status = "Failed";
                    canva.generate_cloth_failed_step = generate_cloth_step_index(GenerateClothStepId::FittingInference);
                    canva_generate_cloth_finish_step(canva, GenerateClothStepId::FittingInference);
                    canva_generate_cloth_finish_pipeline(canva);
                }
                console_add_line(state, "Canva: cloth Fitting cancelled because Sketch2ISO failed");
            }
        }
    }

    (void)canva_try_finish_extract_pose(state, canva);

    if (canva.rig_optimization_job_running && canva.rig_optimization_future.valid())
    {
        const auto ready = canva.rig_optimization_future.wait_for(std::chrono::milliseconds(0));
        if (ready == std::future_status::ready)
        {
            RigOptimizationJobResult result = canva.rig_optimization_future.get();
            canva.rig_optimization_job_running = false;
            canva.rig_optimization_active = false;
            if (canva.rig_optimization_discard_result)
            {
                canva.rig_optimization_discard_result = false;
                console_add_line(state, "Canva: Discarded Rig Optimization result after reset");
            }
            else
            {
                if (result.ok)
                {
                    std::string err;
                    if (canva_apply_rig_optimization_result(state, canva, result.output_json_path, &err))
                    {
                        canva.rig_optimization_status = "Done";
                        std::ostringstream oss;
                        oss << "Canva: Rig Optimization complete";
                        if (canva.rig_optimization_last_final_loss > 0.0f)
                            oss << " (loss " << canva.rig_optimization_last_initial_loss
                                << " -> " << canva.rig_optimization_last_final_loss << ")";
                        console_add_line(state, oss.str());
                    }
                    else
                    {
                        canva.rig_optimization_status = "Failed";
                        canva.rig_optimization_last_error = err.empty() ? "failed to apply optimizer result" : err;
                        console_add_line(state, "Canva: Rig Optimization result import failed: " + canva.rig_optimization_last_error);
                    }
                }
                else
                {
                    canva.rig_optimization_status = "Failed";
                    canva.rig_optimization_last_error = result.error.empty() ? "optimizer failed" : result.error;
                    console_add_line(state, "Canva: Rig Optimization failed: " + canva.rig_optimization_last_error);
                }
            }
        }
    }

    if (state.canva_displacement_completed_id > canva.last_displacement_completed_id)
    {
        canva.last_displacement_completed_id = state.canva_displacement_completed_id;
        if (canva.generate_displacement_waiting_render
            && state.canva_displacement_completed_id >= canva.generate_displacement_render_request_id)
        {
            canva.generate_displacement_waiting_render = false;
            canva.generate_displacement_active = false;

            if (!state.canva_displacement_path.empty())
            {
                std::string err;
                const bool added_layer = canva.generate_displacement_model_input_path.empty()
                    ? canva_add_canvas_filled_png_layer(
                        canva,
                        state.canva_displacement_path,
                        "Displacement",
                        CanvaLayerKind::Displacement,
                        true,
                        &err)
                    : canva_add_preprocess_aligned_png_layer(
                        canva,
                        state.canva_displacement_path,
                        canva.generate_displacement_model_input_path,
                        "Displacement",
                        CanvaLayerKind::Displacement,
                        true,
                        &err);
                if (added_layer)
                {
                    canva.generate_displacement_status = "Done";
                    console_add_line(state, "Canva: Added Displacement layer");
                }
                else
                {
                    canva.generate_displacement_status = "Failed";
                    console_add_line(state, std::string("Canva: Failed to add Displacement layer: ")
                        + (err.empty() ? "(unknown error)" : err));
                }
            }
            else
            {
                canva.generate_displacement_status = "Failed";
                console_add_line(state, "Canva: Generate Displacement projection failed");
            }
        }
    }

    if (state.canva_iso_displacement_completed_id
        > canva.last_iso_displacement_completed_id)
    {
        canva.last_iso_displacement_completed_id =
            state.canva_iso_displacement_completed_id;
        if (canva.generate_iso_displacement_waiting_render
            && state.canva_iso_displacement_completed_id
                >= canva.generate_iso_displacement_render_request_id)
        {
            canva.generate_iso_displacement_waiting_render = false;
            canva.generate_iso_displacement_active = false;

            if (!state.canva_iso_displacement_path.empty())
            {
                std::string error;
                const bool added_layer =
                    canva.generate_iso_displacement_model_input_path.empty()
                    ? canva_add_canvas_filled_png_layer(
                        canva,
                        state.canva_iso_displacement_path,
                        "ISO Displacement",
                        CanvaLayerKind::IsoDisplacement,
                        true,
                        &error)
                    : canva_add_preprocess_aligned_png_layer(
                        canva,
                        state.canva_iso_displacement_path,
                        canva.generate_iso_displacement_model_input_path,
                        "ISO Displacement",
                        CanvaLayerKind::IsoDisplacement,
                        true,
                        &error);
                if (added_layer)
                {
                    canva.generate_iso_displacement_status = "Done";
                    console_add_line(state, "Canva: Added ISO Displacement layer");
                }
                else
                {
                    canva.generate_iso_displacement_status = "Failed";
                    console_add_line(
                        state,
                        "Canva: Failed to add ISO Displacement layer: "
                            + (error.empty() ? std::string("(unknown error)") : error));
                }
            }
            else
            {
                canva.generate_iso_displacement_status = "Failed";
                console_add_line(
                    state,
                    "Canva: Create ISO Displacement Map projection failed");
            }
        }
    }

    if (state.cloth_render_completed_id > canva.last_cloth_render_completed_id)
    {
        canva.last_cloth_render_completed_id = state.cloth_render_completed_id;
        if (canva.render_cloth_active
            && state.cloth_render_completed_id >= canva.render_cloth_request_id)
        {
            canva.render_cloth_active = false;
            if (!state.cloth_render_path.empty())
            {
                std::string err;
                if (canva_add_rendered_cloth_layer(
                        canva,
                        state.cloth_render_path,
                        canva.render_cloth_model_input_path,
                        canva.render_cloth_fine,
                        &err))
                {
                    canva.render_cloth_status = "Done";
                    console_add_line(state,
                        canva.render_cloth_fine
                            ? "Canva: Added Fine cloth render layer"
                            : "Canva: Added Rendered cloth layer");
                }
                else
                {
                    canva.render_cloth_status = "Failed";
                    console_add_line(state, "Canva: Failed to add cloth render layer: "
                        + (err.empty() ? std::string("(unknown error)") : err));
                }
            }
            else
            {
                canva.render_cloth_status = "Failed";
                console_add_line(state,
                    canva.render_cloth_fine
                        ? "Canva: Fine cloth render failed"
                        : "Canva: Cloth render failed");
            }
            canva.render_cloth_model_input_path.clear();
        }

        if (canva.generate_cloth_waiting_render
            && state.cloth_render_completed_id >= canva.generate_cloth_render_request_id)
        {
            canva.generate_cloth_waiting_render = false;
            const bool fit_after_render = canva.generate_cloth_fit_after_render;
            canva.generate_cloth_fit_after_render = false;
            canva_generate_cloth_finish_step(canva, GenerateClothStepId::Render);

            if (!state.cloth_render_path.empty())
            {
                std::string err;
                if (canva_add_generated_cloth_layer(
                        canva,
                        state.cloth_render_path,
                        canva.generate_cloth_model_input_path,
                        !fit_after_render,
                        &err))
                {
                    console_add_line(state, "Canva: Added Generated cloth layer");
                    canva.generate_cloth_failed_step = -1;
                    if (fit_after_render)
                    {
                        std::string fit_err;
                        if (canva_queue_cloth_fitting_predictions(
                                state,
                                canva,
                                canva.layers.front(),
                                &fit_err))
                        {
                            canva.generate_cloth_active = true;
                            canva.generate_cloth_status = "Fitting";
                            console_add_line(state, "Canva: Generate + Fit queued cloth fitting predictions");
                        }
                        else
                        {
                            canva.generate_cloth_active = false;
                            canva.generate_cloth_status = "Failed";
                            canva.generate_cloth_failed_step = generate_cloth_step_index(GenerateClothStepId::FittingInference);
                            canva_generate_cloth_finish_step(canva, GenerateClothStepId::FittingInference);
                            canva_generate_cloth_finish_pipeline(canva);
                            console_add_line(state, std::string("Canva: Generate + Fit failed to queue cloth fitting: ")
                                + (fit_err.empty() ? "(unknown error)" : fit_err));
                        }
                    }
                    else
                    {
                        canva.generate_cloth_active = false;
                        canva.generate_cloth_status = "Done";
                        canva_generate_cloth_finish_pipeline(canva);
                    }
                    canva.generate_cloth_model_input_path.clear();
                }
                else
                {
                    console_add_line(state, std::string("Canva: Failed to add Generated cloth layer: ")
                        + (err.empty() ? "(unknown error)" : err));
                    canva.generate_cloth_active = false;
                    canva.generate_cloth_status = "Failed";
                    canva.generate_cloth_failed_step = 5;
                    canva_generate_cloth_finish_pipeline(canva);
                    canva.generate_cloth_model_input_path.clear();
                }
            }
            else
            {
                console_add_line(state, "Canva: Generate Cloth render failed");
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Failed";
                canva.generate_cloth_failed_step = 5;
                canva_generate_cloth_finish_pipeline(canva);
                canva.generate_cloth_model_input_path.clear();
            }
        }
    }

    if (canva.generate_cloth_waiting_pose)
    {
        const bool pose_selected =
            state.pending_pose_selection.empty()
            && state.selection.pose_index >= 0
            && state.selection.pose_index < (int)state.selection.poses.size()
            && state.selection.poses[(size_t)state.selection.pose_index] == canva.generate_cloth_pose_name;

        if (pose_selected)
        {
            if (state.selection.garment_index < 0
                || state.selection.garment_index >= (int)state.selection.garments.size())
            {
                console_add_line(state, "Canva: Generate Cloth cancelled because no garment mesh is selected");
                canva.generate_cloth_waiting_pose = false;
                canva.generate_cloth_active = false;
                canva.generate_cloth_status = "Failed";
                canva.generate_cloth_failed_step = 1;
                canva.generate_cloth_fit_after_render = false;
                canva_generate_cloth_finish_step(canva, GenerateClothStepId::Pose);
                canva_generate_cloth_finish_pipeline(canva);
            }
            else
            {
                state.show_smpl = true;
                state.show_smpl_character = true;
                state.show_garment_mesh = true;
                state.controls.current_frame = 0;
                state.controls.playing = false;
                state.controls.simulating = true;
                state.cloth_simulate_request_id++;
                canva.generate_cloth_waiting_pose = false;
                canva.generate_cloth_waiting_simulation = true;
                canva.generate_cloth_status = "Simulating";
                canva.generate_cloth_failed_step = -1;
                canva_generate_cloth_finish_step(canva, GenerateClothStepId::Pose);
                canva_generate_cloth_start_step(canva, GenerateClothStepId::Sequence);
                canva.generate_cloth_last_sim_phase = 0;
                console_add_line(state, "Canva: Generate Cloth queued XPBD simulation");
            }
        }
    }

    canva_generate_cloth_update_simulation_phase(state, canva);

    if (canva.generate_cloth_waiting_simulation && !state.controls.simulating)
    {
        canva.generate_cloth_waiting_simulation = false;
        canva_generate_cloth_finish_simulation_steps(state, canva);
        if (state.controls.frame_count <= 0)
        {
            canva.generate_cloth_active = false;
            canva.generate_cloth_status = "Failed";
            canva.generate_cloth_failed_step = 2 + std::clamp(state.controls.sim_pipeline_stage, 0, 2);
            canva.generate_cloth_fit_after_render = false;
            canva_generate_cloth_finish_pipeline(canva);
            console_add_line(state, "Canva: Generate Cloth cancelled because XPBD produced no frames");
        }
        else
        {
            state.controls.playing = false;
            state.controls.current_frame = std::max(0, state.controls.frame_count - 1);
            state.cloth_render_path.clear();
            state.cloth_render_fine = false;
            state.cloth_render_use_current_camera = false;
            state.cloth_render_stroke_reference_path = canva.generate_cloth_model_input_path.string();
            canva.generate_cloth_render_request_id = ++state.cloth_render_request_id;
            canva.generate_cloth_waiting_render = true;
            canva.generate_cloth_status = "Rendering";
            canva.generate_cloth_failed_step = -1;
            canva_generate_cloth_start_step(canva, GenerateClothStepId::Render);
            console_add_line(state, "Canva: Generate Cloth queued transparent cloth render");
        }
    }

    // Global shortcuts for this tab (works even when hovering the controls).
    {
        ImGuiIO& io = ImGui::GetIO();
        const bool focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
        if (focused && io.KeyCtrl && !io.WantTextInput)
        {
            // Transform tool toggle (Ctrl+T)
            if (ImGui::IsKeyPressed(GLFW_KEY_T))
            {
                canva.tool = (canva.tool == CanvaTool::Brush) ? CanvaTool::Transform : CanvaTool::Brush;
                canva_cancel_interactions(canva);
            }
        }
    }

    }
}
