#include "automation/AutomationServer.h"

#include "app/AppState.h"
#include "core/Assets.h"
#include "render/Renderer.h"
#include "ui/canva/CanvaPanelInternal.h"

#include <nlohmann/json.hpp>
#include <stb_image_write.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace poseweaver
{
    using json = nlohmann::json;
    using namespace ui_canva;

    namespace
    {
        constexpr std::size_t kMaxRequestBytes = 4u * 1024u * 1024u;

        const char* inference_status_name(ModelInferenceStatus value)
        {
            switch (value)
            {
            case ModelInferenceStatus::NotStarted: return "not_started";
            case ModelInferenceStatus::Loading: return "loading";
            case ModelInferenceStatus::Ready: return "ready";
            case ModelInferenceStatus::Busy: return "busy";
            case ModelInferenceStatus::Error: return "error";
            case ModelInferenceStatus::Stopped: return "stopped";
            }
            return "unknown";
        }

        const char* layer_kind_name(CanvaLayerKind value)
        {
            switch (value)
            {
            case CanvaLayerKind::Generic: return "generic";
            case CanvaLayerKind::Smpl: return "smpl";
            case CanvaLayerKind::VitPose: return "vitpose";
            case CanvaLayerKind::Sketch2UV: return "sketch2uv";
            case CanvaLayerKind::Sketch2ISO: return "sketch2iso";
            case CanvaLayerKind::Sketch2Depth: return "sketch2depth";
            case CanvaLayerKind::ManualUV: return "manual_uv";
            case CanvaLayerKind::GeneratedCloth: return "generated_cloth";
            case CanvaLayerKind::Displacement: return "displacement";
            case CanvaLayerKind::IsoDisplacement: return "iso_displacement";
            case CanvaLayerKind::ClothFitting: return "cloth_fitting";
            case CanvaLayerKind::ClothFittingWarped: return "cloth_fitting_warped";
            case CanvaLayerKind::ClothFittingWarpedSmoothed: return "cloth_fitting_warped_smoothed";
            case CanvaLayerKind::ClothFittingDebug: return "cloth_fitting_debug";
            case CanvaLayerKind::ClothFittingOcclusionDebug: return "cloth_fitting_occlusion_debug";
            case CanvaLayerKind::ClothFittingSceneOcclusionDebug: return "cloth_fitting_scene_occlusion_debug";
            case CanvaLayerKind::ClothFittingAlignedPredictedDepthOcclusionDebug: return "cloth_fitting_aligned_depth_debug";
            case CanvaLayerKind::ClothFittingIsoOcclusionDebug: return "cloth_fitting_iso_debug";
            case CanvaLayerKind::ClothFittingDeformation: return "cloth_fitting_deformation";
            }
            return "unknown";
        }

        json inference_state_json(const ModelInferenceUiState& value)
        {
            return {
                { "status", inference_status_name(value.status) },
                { "message", value.message },
                { "active_request_id", value.active_request_id },
                { "completed_request_id", value.completed_request_id },
                { "last_input", value.last_sketch_path },
                { "last_output", value.last_output_path }
            };
        }

        std::string selected_name(const std::vector<std::string>& values, int index)
        {
            if (index < 0 || index >= static_cast<int>(values.size()))
                return {};
            return values[static_cast<std::size_t>(index)];
        }

        int find_name(const std::vector<std::string>& values, const std::string& name)
        {
            const auto it = std::find(values.begin(), values.end(), name);
            return it == values.end() ? -1 : static_cast<int>(std::distance(values.begin(), it));
        }

        json layer_json(const CanvaState& canva, const CanvaLayer& layer, int index)
        {
            return {
                { "id", layer.id },
                { "index", index },
                { "name", layer.name },
                { "kind", layer_kind_name(layer.kind) },
                { "visible", layer.enabled },
                { "selected", index == canva.active_layer },
                { "offset", { layer.offset.x, layer.offset.y } },
                { "scale", layer.scale },
                { "pixel_width", canva.width },
                { "pixel_height", canva.height },
                { "has_pixels", !layer.pixels.empty() }
            };
        }

        std::filesystem::path checked_output_path(const json& params)
        {
            if (!params.contains("path") || !params["path"].is_string())
                throw std::runtime_error("path must be a non-empty string");
            std::filesystem::path path(params["path"].get<std::string>());
            if (path.empty())
                throw std::runtime_error("path must be a non-empty string");
            if (!path.is_absolute())
                path = std::filesystem::absolute(path);
            std::error_code ec;
            if (!path.parent_path().empty())
                std::filesystem::create_directories(path.parent_path(), ec);
            if (ec)
                throw std::runtime_error("failed to create output directory: " + ec.message());
            return path.lexically_normal();
        }

        bool write_rgba_png(
            const std::filesystem::path& path,
            int width,
            int height,
            const std::vector<unsigned char>& rgba,
            std::string* out_error)
        {
            if (width <= 0 || height <= 0
                || rgba.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u)
            {
                if (out_error)
                    *out_error = "layer has no valid RGBA pixels";
                return false;
            }
            if (!stbi_write_png(path.string().c_str(), width, height, 4, rgba.data(), width * 4))
            {
                if (out_error)
                    *out_error = "failed to write PNG: " + path.string();
                return false;
            }
            return true;
        }

        struct ParameterSpec
        {
            std::string name;
            std::string type;
            std::string description;
            std::optional<double> minimum;
            std::optional<double> maximum;
            std::vector<std::string> choices;
            std::function<json(const AppState&, const CanvaState&)> get;
            std::function<void(AppState&, CanvaState&, const json&)> set;
        };

        std::vector<ParameterSpec> make_parameter_specs()
        {
            std::vector<ParameterSpec> out;
#define PW_BOOL(NAME, DESC, FIELD) out.push_back({ NAME, "boolean", DESC, {}, {}, {}, \
    [](const AppState& state, const CanvaState&) -> json { return state.FIELD; }, \
    [](AppState& state, CanvaState&, const json& value) { state.FIELD = value.get<bool>(); } })
#define PW_INT(NAME, DESC, FIELD, MIN, MAX) out.push_back({ NAME, "integer", DESC, double(MIN), double(MAX), {}, \
    [](const AppState& state, const CanvaState&) -> json { return state.FIELD; }, \
    [](AppState& state, CanvaState&, const json& value) { state.FIELD = value.get<int>(); } })
#define PW_FLOAT(NAME, DESC, FIELD, MIN, MAX) out.push_back({ NAME, "number", DESC, double(MIN), double(MAX), {}, \
    [](const AppState& state, const CanvaState&) -> json { return state.FIELD; }, \
    [](AppState& state, CanvaState&, const json& value) { state.FIELD = value.get<float>(); } })
#define PW_CANVA_INT(NAME, DESC, FIELD, MIN, MAX) out.push_back({ NAME, "integer", DESC, double(MIN), double(MAX), {}, \
    [](const AppState&, const CanvaState& canva) -> json { return canva.FIELD; }, \
    [](AppState&, CanvaState& canva, const json& value) { canva.FIELD = value.get<int>(); } })
#define PW_CANVA_FLOAT(NAME, DESC, FIELD, MIN, MAX) out.push_back({ NAME, "number", DESC, double(MIN), double(MAX), {}, \
    [](const AppState&, const CanvaState& canva) -> json { return canva.FIELD; }, \
    [](AppState&, CanvaState& canva, const json& value) { canva.FIELD = value.get<float>(); } })

            PW_FLOAT("simulation.duration_seconds", "Physical simulation duration.", xpbd_params.duration_seconds, 0.001, 3600.0);
            PW_INT("simulation.steps", "Number of simulation steps.", xpbd_params.simulation_steps, 1, 1000000);
            PW_INT("simulation.solver_iterations", "XPBD solver iterations per step.", xpbd_params.solver_iterations, 1, 10000);
            PW_INT("simulation.substeps", "Solver substeps.", xpbd_params.substeps, 1, 1000);
            PW_FLOAT("simulation.gravity", "Vertical acceleration.", xpbd_params.gravity, -1000.0, 1000.0);
            PW_BOOL("simulation.stretch_enabled", "Enable stretch constraints.", xpbd_params.stretch_enabled);
            PW_FLOAT("simulation.stretch_compliance", "Stretch compliance.", xpbd_params.stretch_compliance, 0.0, 1000.0);
            PW_FLOAT("simulation.base_stretch_multiplier", "Base stretch multiplier.", xpbd_params.base_stretch_mult, 0.0, 1000.0);
            PW_BOOL("simulation.bend_enabled", "Enable bend constraints.", xpbd_params.bend_enabled);
            PW_FLOAT("simulation.bend_compliance", "Bend compliance.", xpbd_params.bend_compliance, 0.0, 1000.0);
            PW_FLOAT("simulation.dihedral_compliance", "Dihedral bend compliance.", xpbd_params.dihedral_compliance, 0.0, 1000.0);
            PW_BOOL("simulation.collision_enabled", "Enable body collision.", xpbd_params.collision_enabled);
            PW_FLOAT("simulation.collision_compliance", "Body collision compliance.", xpbd_params.collision_compliance, 0.0, 1000.0);
            PW_FLOAT("simulation.collision_thickness", "Body collision thickness.", xpbd_params.collision_thickness, 0.0, 10.0);
            PW_BOOL("simulation.self_collision_enabled", "Enable cloth self collision.", xpbd_params.self_collision_enabled);
            PW_FLOAT("simulation.self_collision_compliance", "Self collision compliance.", xpbd_params.self_collision_compliance, 0.0, 1000.0);
            PW_FLOAT("simulation.self_collision_thickness", "Self collision thickness.", xpbd_params.self_collision_thickness, 0.0, 10.0);
            PW_BOOL("simulation.friction_enabled", "Enable friction.", xpbd_params.enable_friction);
            PW_FLOAT("simulation.friction", "Friction coefficient.", xpbd_params.friction_mu, 0.0, 100.0);
            PW_FLOAT("simulation.damping", "Velocity damping.", xpbd_params.damping, 0.0, 1.0);
            PW_FLOAT("simulation.wind_force", "Constant wind magnitude.", xpbd_params.wind_force, -10000.0, 10000.0);
            PW_FLOAT("simulation.wind_yaw_degrees", "Wind yaw in degrees.", xpbd_params.wind_yaw_deg, -36000.0, 36000.0);
            PW_FLOAT("simulation.wind_pitch_degrees", "Wind pitch in degrees.", xpbd_params.wind_pitch_deg, -36000.0, 36000.0);
            PW_FLOAT("simulation.wind_random_magnitude", "Maximum random wind magnitude.", xpbd_params.wind_random_magnitude, 0.0, 10000.0);
            PW_FLOAT("simulation.wind_random_time_scale", "Random wind time scale.", xpbd_params.wind_random_time_scale, 0.0, 10000.0);
            PW_FLOAT("simulation.wind_random_spatial_scale", "Random wind spatial scale.", xpbd_params.wind_random_spatial_scale, 0.0, 10000.0);
            PW_FLOAT("simulation.smpl_geometry_percent", "Percentage of SMPL collision geometry retained.", xpbd_params.smpl_geometry_percent, 1.0, 100.0);
            PW_BOOL("simulation.smooth_sequence", "Smooth pose sequences before simulation.", controls.smooth_sequence);
            PW_FLOAT("simulation.smooth_sequence_length", "Sequence smoothing strength.", controls.smooth_sequence_length, 0.0, 1.0);
            PW_BOOL("simulation.save_smpl_sequence", "Save the generated SMPL sequence.", controls.save_smpl_sequence);
            PW_FLOAT("playback.fps", "Preview playback frames per second.", controls.fps, 0.01, 1000.0);
            PW_INT("playback.current_frame", "Current preview frame.", controls.current_frame, 0, 1000000);
            PW_BOOL("playback.playing", "Whether preview playback is active.", controls.playing);
            PW_BOOL("playback.preview", "Show live simulation preview.", controls.preview);

            PW_BOOL("garment.merge_vertices", "Merge nearby garment vertices on load.", garment_vertex_merge_enabled);
            PW_FLOAT("garment.merge_distance", "Garment vertex merge distance.", garment_vertex_merge_distance, 0.0, 1.0);
            PW_FLOAT("garment.minimum_skin_distance", "Minimum cloth-to-skin distance.", garment_min_cloth_skin_distance, 0.0, 1.0);
            PW_FLOAT("garment.ease_amount", "Garment ease amount.", garment_ease_amount, 0.0, 10.0);
            PW_INT("garment.prefit_steps", "Garment prefit steps.", garment_prefit_steps, 1, 100000);

            PW_BOOL("view.show_smpl", "Show the SMPL body.", show_smpl);
            PW_BOOL("view.show_smpl_character", "Show the character surface.", show_smpl_character);
            PW_BOOL("view.show_garment", "Show garments.", show_garments);
            PW_BOOL("view.show_garment_mesh", "Show the garment mesh.", show_garment_mesh);
            PW_BOOL("view.show_object_mesh", "Show the object mesh.", show_object_mesh);
            PW_BOOL("view.show_collision_colliders", "Show collision geometry.", show_collision_colliders);
            PW_BOOL("view.show_rig", "Show the predicted rig.", show_sketch2rig_rig);
            PW_BOOL("view.garment_color_by_smpl_uv", "Color garment by SMPL UV.", garment_color_by_smpl_uv);
            PW_FLOAT("render.target_detail_size", "Fine cloth render target edge length.", cloth_render_target_detail_size, 0.0001, 0.25);

            PW_FLOAT("fitting.depth_margin", "Cloth fitting depth margin.", cloth_fitting_depth_margin, 0.0, 1.0);
            PW_FLOAT("fitting.predicted_depth_margin", "Predicted depth margin.", cloth_fitting_depth_margin_pred, 0.0, 10.0);
            PW_FLOAT("fitting.body_distance_threshold", "Body distance threshold.", cloth_fitting_body_distance_threshold, 0.0, 10.0);
            PW_FLOAT("fitting.warp_radius", "Warp radius in pixels.", cloth_fitting_warp_radius, 0.0, 100000.0);
            PW_FLOAT("fitting.warp_force", "Warp strength.", cloth_fitting_warp_force, 0.0, 1000.0);
            PW_INT("fitting.warp_steps", "Warp iteration count.", cloth_fitting_warp_steps, 1, 100000);
            PW_FLOAT("fitting.smoothing_strength", "Cloth smoothing filter strength.", cloth_smoothing_filter_strength, 0.0, 1.0);
            PW_FLOAT("fitting.smoothing_inner_line_width", "Protected inner line width.", cloth_smoothing_inner_line_width, 0.0, 10000.0);
            PW_INT("fitting.smoothing_contour_radius", "Contour smoothing radius.", cloth_smoothing_contour_radius, 0, 10000);
            PW_INT("fitting.smoothing_line_gap_radius", "Line gap closing radius.", cloth_smoothing_line_gap_radius, 0, 10000);
            PW_FLOAT("fitting.iso_distance_threshold", "ISO descriptor matching threshold.", iso_distance_threshold, 0.0, 10000.0);

            PW_FLOAT("canvas.brush_radius", "Canvas brush radius.", canva_brush_radius, 0.5, 250.0);
            PW_INT("canvas.preferred_bone_prediction", "0 for custom models, 1 for NLF.", canva_preferred_bone_prediction, 0, 1);
            PW_CANVA_FLOAT("canvas.zoom", "Canvas view zoom.", zoom, 0.01, 100.0);
            PW_CANVA_FLOAT("canvas.pan_x", "Canvas horizontal pan.", pan.x, -1000000.0, 1000000.0);
            PW_CANVA_FLOAT("canvas.pan_y", "Canvas vertical pan.", pan.y, -1000000.0, 1000000.0);
            PW_CANVA_INT("rig_optimization.iterations", "Rig optimizer iteration count.", rig_optimization_iterations, 1, 100000);
            PW_CANVA_INT("rig_optimization.render_size", "Rig optimizer render size.", rig_optimization_render_size, 16, 4096);
            PW_CANVA_FLOAT("rig_optimization.learning_rate", "Rig optimizer learning rate.", rig_optimization_lr, 0.000001, 1000.0);
            PW_CANVA_FLOAT("rig_optimization.pose_regularization", "Pose regularization.", rig_optimization_pose_reg, 0.0, 1000.0);
            PW_CANVA_FLOAT("rig_optimization.beta_regularization", "Shape regularization.", rig_optimization_beta_reg, 0.0, 1000.0);

            out.push_back({ "simulation.device", "string", "Simulation backend.", {}, {}, { "cpu", "gpu" },
                [](const AppState& state, const CanvaState&) -> json {
                    return state.xpbd_params.device == SimulationDevice::Gpu ? "gpu" : "cpu";
                },
                [](AppState& state, CanvaState&, const json& value) {
                    state.xpbd_params.device = value.get<std::string>() == "gpu" ? SimulationDevice::Gpu : SimulationDevice::Cpu;
                } });
            out.push_back({ "simulation.collision_geometry", "string", "Collision geometry source.", {}, {},
                { "raw_smpl", "simplified_smpl", "simple_proxy" },
                [](const AppState& state, const CanvaState&) -> json {
                    switch (state.xpbd_params.collision_geometry)
                    {
                    case CollisionGeometryType::SimplifiedSmpl: return "simplified_smpl";
                    case CollisionGeometryType::SimpleProxy: return "simple_proxy";
                    default: return "raw_smpl";
                    }
                },
                [](AppState& state, CanvaState&, const json& value) {
                    const std::string name = value.get<std::string>();
                    state.xpbd_params.collision_geometry = name == "simple_proxy"
                        ? CollisionGeometryType::SimpleProxy
                        : name == "simplified_smpl" ? CollisionGeometryType::SimplifiedSmpl : CollisionGeometryType::RawSmpl;
                    state.xpbd_params.simplify_smpl = state.xpbd_params.collision_geometry == CollisionGeometryType::SimplifiedSmpl;
                } });
            out.push_back({ "simulation.bending_method", "string", "Bending constraint formulation.", {}, {},
                { "opposite_vertex_distance", "dihedral" },
                [](const AppState& state, const CanvaState&) -> json {
                    return state.xpbd_params.bending_method == BendingMethod::DihedralAngle ? "dihedral" : "opposite_vertex_distance";
                },
                [](AppState& state, CanvaState&, const json& value) {
                    state.xpbd_params.bending_method = value.get<std::string>() == "dihedral"
                        ? BendingMethod::DihedralAngle : BendingMethod::OppositeVertexDistance;
                } });

            for (int i = 0; i < 16; ++i)
            {
                out.push_back({ "smpl.beta." + std::to_string(i), "number", "SMPL shape coefficient.", -3.0, 3.0, {},
                    [i](const AppState& state, const CanvaState&) -> json { return state.smpl_shape.betas[static_cast<std::size_t>(i)]; },
                    [i](AppState& state, CanvaState&, const json& value) { state.smpl_shape.betas[static_cast<std::size_t>(i)] = value.get<float>(); } });
            }

#undef PW_BOOL
#undef PW_INT
#undef PW_FLOAT
#undef PW_CANVA_INT
#undef PW_CANVA_FLOAT
            return out;
        }

        json parameter_schema_json(const ParameterSpec& spec)
        {
            json out = {
                { "name", spec.name },
                { "type", spec.type },
                { "description", spec.description }
            };
            if (spec.minimum)
                out["minimum"] = *spec.minimum;
            if (spec.maximum)
                out["maximum"] = *spec.maximum;
            if (!spec.choices.empty())
                out["choices"] = spec.choices;
            return out;
        }

        void validate_parameter_value(const ParameterSpec& spec, const json& value)
        {
            if (spec.type == "boolean" && !value.is_boolean())
                throw std::runtime_error(spec.name + " must be boolean");
            if (spec.type == "integer" && !value.is_number_integer())
                throw std::runtime_error(spec.name + " must be an integer");
            if (spec.type == "number" && !value.is_number())
                throw std::runtime_error(spec.name + " must be a number");
            if (spec.type == "string" && !value.is_string())
                throw std::runtime_error(spec.name + " must be a string");
            if (value.is_number())
            {
                const double number = value.get<double>();
                if (!std::isfinite(number))
                    throw std::runtime_error(spec.name + " must be finite");
                if (spec.minimum && number < *spec.minimum)
                    throw std::runtime_error(spec.name + " is below its minimum");
                if (spec.maximum && number > *spec.maximum)
                    throw std::runtime_error(spec.name + " is above its maximum");
            }
            if (!spec.choices.empty())
            {
                const std::string choice = value.get<std::string>();
                if (std::find(spec.choices.begin(), spec.choices.end(), choice) == spec.choices.end())
                    throw std::runtime_error(spec.name + " has an unsupported value");
            }
        }
    }

    struct AutomationServer::Impl
    {
        struct PendingRequest
        {
            json request;
            std::promise<json> response;
        };

        struct Job
        {
            std::uint64_t id = 0;
            std::string kind;
            std::string state = "queued";
            std::string message;
            bool observed_active = false;
            std::uint64_t marker = 0;
            std::chrono::steady_clock::time_point created = std::chrono::steady_clock::now();
        };

        std::string socket_path;
        std::thread listener;
        std::atomic_bool running{ false };
        std::atomic_bool stop_requested{ false };
        std::atomic_bool app_shutdown_requested{ false };
        std::mutex queue_mutex;
        std::vector<std::shared_ptr<PendingRequest>> queue;
        std::vector<Job> jobs;
        std::uint64_t next_job_id = 1;
#if !defined(_WIN32)
        int listener_fd = -1;
#endif

        static json success(const json& id, json result)
        {
            return { { "id", id }, { "ok", true }, { "result", std::move(result) } };
        }

        static json failure(const json& id, const std::string& error)
        {
            return { { "id", id }, { "ok", false }, { "error", error } };
        }

        json job_json(const Job& job, const AppState& state, const CanvaState& canva) const
        {
            json out = {
                { "job_id", job.id },
                { "operation", job.kind },
                { "state", job.state },
                { "message", job.message }
            };
            if (job.kind == "simulation")
            {
                out["stage"] = state.controls.sim_stage;
                out["progress"] = {
                    { "done", state.controls.sim_steps_done },
                    { "total", state.controls.sim_steps_total },
                    { "pipeline_stage", state.controls.sim_pipeline_stage },
                    { "sequence_done", state.controls.sim_sequence_steps_done },
                    { "sequence_total", state.controls.sim_sequence_steps_total },
                    { "proxy_done", state.controls.sim_proxy_steps_done },
                    { "proxy_total", state.controls.sim_proxy_steps_total },
                    { "solver_done", state.controls.sim_solver_steps_done },
                    { "solver_total", state.controls.sim_solver_steps_total }
                };
                out["timings_seconds"] = {
                    { "sequence", state.controls.sim_sequence_seconds },
                    { "proxy", state.controls.sim_proxy_seconds },
                    { "solver", state.controls.sim_solver_seconds },
                    { "total", state.controls.sim_pipeline_seconds }
                };
            }
            else if (job.kind == "generate_cloth" || job.kind == "full_generation" || job.kind == "generate_and_fit")
            {
                out["stage"] = canva.generate_cloth_status;
                out["failed_step"] = canva.generate_cloth_failed_step;
                out["elapsed_seconds"] = canva.generate_cloth_total_seconds;
            }
            else if (job.kind == "render_cloth")
                out["stage"] = canva.render_cloth_status;
            else if (job.kind == "cloth_fitting")
                out["stage"] = canva.cloth_fitting_status;
            else if (job.kind == "extract_pose")
                out["stage"] = canva.extract_pose_active ? "extracting" : "done";
            else if (job.kind == "rig_optimization")
                out["stage"] = canva.rig_optimization_status;
            else if (job.kind == "generate_displacement")
                out["stage"] = canva.generate_displacement_status;
            else if (job.kind == "generate_iso_displacement")
                out["stage"] = canva.generate_iso_displacement_status;
            else if (job.kind == "scene_snapshot")
                out["output_path"] = state.scene_view_snapshot_path;
            return out;
        }

        bool job_active(const Job& job, const AppState& state, const CanvaState& canva) const
        {
            if (job.kind == "simulation") return state.controls.simulating;
            if (job.kind == "generate_cloth" || job.kind == "full_generation" || job.kind == "generate_and_fit")
                return canva.generate_cloth_active || canva.extract_pose_active;
            if (job.kind == "render_cloth") return canva.render_cloth_active;
            if (job.kind == "cloth_fitting") return canva.cloth_fitting_active;
            if (job.kind == "extract_pose") return canva.extract_pose_active;
            if (job.kind == "rig_optimization") return canva.rig_optimization_active;
            if (job.kind == "generate_displacement") return canva.generate_displacement_active;
            if (job.kind == "generate_iso_displacement") return canva.generate_iso_displacement_active;
            if (job.kind == "scene_snapshot") return state.scene_view_snapshot_completed_id < job.marker;
            return false;
        }

        bool job_failed(const Job& job, const AppState& state, const CanvaState& canva) const
        {
            if (job.kind == "simulation") return state.controls.sim_stage == "Failed";
            if (job.kind == "generate_cloth" || job.kind == "full_generation" || job.kind == "generate_and_fit")
                return canva.generate_cloth_status == "Failed";
            if (job.kind == "render_cloth") return canva.render_cloth_status == "Failed";
            if (job.kind == "cloth_fitting") return canva.cloth_fitting_status == "Failed";
            if (job.kind == "rig_optimization") return canva.rig_optimization_status == "Failed";
            if (job.kind == "generate_displacement") return canva.generate_displacement_status == "Failed";
            if (job.kind == "generate_iso_displacement") return canva.generate_iso_displacement_status == "Failed";
            return false;
        }

        void update_jobs(const AppState& state, const CanvaState& canva)
        {
            for (Job& job : jobs)
            {
                if (job.state == "completed" || job.state == "failed" || job.state == "cancelled")
                    continue;
                const bool active = job_active(job, state, canva);
                if (active)
                {
                    job.observed_active = true;
                    job.state = "running";
                }
                else if (job_failed(job, state, canva))
                {
                    job.state = "failed";
                }
                else if (job.observed_active || job.kind == "scene_snapshot")
                {
                    job.state = "completed";
                }
            }
        }

        Job& create_job(const std::string& kind, std::uint64_t marker = 0)
        {
            jobs.push_back(Job{});
            Job& job = jobs.back();
            job.id = next_job_id++;
            job.kind = kind;
            job.marker = marker;
            return job;
        }

        json app_status(const AppState& state, const CanvaState& canva) const
        {
            const char* gender = state.selection.gender == SmplType::Female
                ? "female" : state.selection.gender == SmplType::Neutral ? "neutral" : "male";
            return {
                { "running", true },
#if !defined(_WIN32)
                { "pid", static_cast<long long>(getpid()) },
#else
                { "pid", 0 },
#endif
                { "socket_path", socket_path },
                { "selection", {
                    { "gender", gender },
                    { "pose", selected_name(state.selection.poses, state.selection.pose_index) },
                    { "garment", selected_name(state.selection.garments, state.selection.garment_index) },
                    { "mesh", selected_name(state.selection.meshes, state.selection.mesh_index) }
                } },
                { "simulation", {
                    { "active", state.controls.simulating },
                    { "playing", state.controls.playing },
                    { "stage", state.controls.sim_stage },
                    { "frame", state.controls.current_frame },
                    { "frame_count", state.controls.frame_count },
                    { "progress_done", state.controls.sim_steps_done },
                    { "progress_total", state.controls.sim_steps_total }
                } },
                { "canvas", {
                    { "width", canva.width },
                    { "height", canva.height },
                    { "layer_count", canva.layers.size() },
                    { "selected_layer_id", canva.active_layer >= 0 && canva.active_layer < static_cast<int>(canva.layers.size())
                        ? canva.layers[static_cast<std::size_t>(canva.active_layer)].id : 0 },
                    { "generate_cloth", canva.generate_cloth_status },
                    { "render_cloth", canva.render_cloth_status },
                    { "cloth_fitting", canva.cloth_fitting_status },
                    { "rig_optimization", canva.rig_optimization_status }
                } },
                { "models", {
                    { "sketch2rig", inference_state_json(state.sketch2rig) },
                    { "sketch2betas", inference_state_json(state.sketch2betas) },
                    { "sketch2uv", inference_state_json(state.sketch2uv) },
                    { "sketch2iso", inference_state_json(state.sketch2iso) },
                    { "sketch2depth", inference_state_json(state.sketch2depth) },
                    { "nlf", inference_state_json(state.nlf) },
                    { "vitpose", inference_state_json(state.vitpose) }
                } }
            };
        }

        json start_operation(AppState& state, CanvaState& canva, Renderer& renderer, const json& params)
        {
            if (!params.contains("name") || !params["name"].is_string())
                throw std::runtime_error("operation name is required");
            const std::string name = params["name"].get<std::string>();
            std::string error;
            Job* job = nullptr;

            if (name == "simulation")
            {
                if (state.controls.simulating)
                    throw std::runtime_error("simulation is already running");
                if (state.selection.garment_index < 0)
                    throw std::runtime_error("select a garment before starting simulation");
                state.cloth_simulate_play_on_finish = params.value("play_on_finish", true);
                state.cloth_simulate_continue_from_last_frame = params.value("continue_from_last_frame", false);
                ++state.cloth_simulate_request_id;
                job = &create_job(name, state.cloth_simulate_request_id);
            }
            else if (name == "generate_cloth")
            {
                if (!canva_queue_generate_cloth_current_pose(state, canva, &error))
                    throw std::runtime_error(error.empty() ? "failed to queue cloth generation" : error);
                job = &create_job(name);
                job->observed_active = canva.generate_cloth_active;
                job->state = job->observed_active ? "running" : "queued";
            }
            else if (name == "full_generation" || name == "generate_and_fit")
            {
                const bool fit = name == "generate_and_fit";
                if (!canva_queue_extract_pose(state, canva, true, true, &error))
                    throw std::runtime_error(error.empty() ? "failed to queue pose extraction" : error);
                canva.generate_cloth_active = true;
                canva.generate_cloth_waiting_pose = false;
                canva.generate_cloth_waiting_simulation = false;
                canva.generate_cloth_waiting_render = false;
                canva.generate_cloth_status = "Inference";
                canva.generate_cloth_failed_step = -1;
                canva.generate_cloth_fit_after_render = fit;
                canva.generate_cloth_include_fitting_steps = fit;
                job = &create_job(name);
                job->observed_active = true;
                job->state = "running";
            }
            else if (name == "render_cloth")
            {
                if (!canva_queue_render_cloth(state, canva, params.value("fine", false), &error))
                    throw std::runtime_error(error.empty() ? "failed to queue cloth render" : error);
                job = &create_job(name);
                job->observed_active = true;
                job->state = "running";
            }
            else if (name == "cloth_fitting")
            {
                if (canva.active_layer < 0 || canva.active_layer >= static_cast<int>(canva.layers.size()))
                    throw std::runtime_error("select a generated cloth layer first");
                const CanvaLayer& layer = canva.layers[static_cast<std::size_t>(canva.active_layer)];
                if (!canva_layer_is_generated_cloth(layer))
                    throw std::runtime_error("selected layer is not a generated cloth layer");
                if (!canva_queue_cloth_fitting_predictions(state, canva, layer, &error))
                    throw std::runtime_error(error.empty() ? "failed to queue cloth fitting" : error);
                job = &create_job(name);
                job->observed_active = true;
                job->state = "running";
            }
            else if (name == "extract_pose")
            {
                if (!canva_queue_extract_pose(state, canva, false, params.value("hide_rig_layer", false), &error))
                    throw std::runtime_error(error.empty() ? "failed to queue pose extraction" : error);
                job = &create_job(name);
                job->observed_active = true;
                job->state = "running";
            }
            else if (name == "rig_optimization")
            {
                if (!canva_queue_rig_optimization(state, canva, &error))
                    throw std::runtime_error(error.empty() ? "failed to queue rig optimization" : error);
                job = &create_job(name);
                job->observed_active = true;
                job->state = "running";
            }
            else if (name == "generate_displacement")
            {
                if (!canva_queue_generate_displacement(state, canva, &error))
                    throw std::runtime_error(error.empty() ? "failed to queue displacement generation" : error);
                job = &create_job(name);
                job->observed_active = true;
                job->state = "running";
            }
            else if (name == "generate_iso_displacement")
            {
                if (!canva_queue_generate_iso_displacement(state, canva, &error))
                    throw std::runtime_error(error.empty() ? "failed to queue ISO displacement generation" : error);
                job = &create_job(name);
                job->observed_active = true;
                job->state = "running";
            }
            else if (name == "scene_snapshot")
            {
                if (params.contains("path"))
                    state.scene_view_snapshot_path = checked_output_path(params).string();
                const std::uint64_t marker = ++state.scene_view_snapshot_request_id;
                job = &create_job(name, marker);
                job->observed_active = true;
                job->state = "running";
            }
            else if (name == "garment_pushout")
            {
                ++state.garment_pushout_request_id;
                job = &create_job(name);
                job->state = "completed";
            }
            else if (name == "garment_ease")
            {
                ++state.garment_ease_request_id;
                job = &create_job(name);
                job->state = "completed";
            }
            else if (name == "garment_prefit")
            {
                ++state.garment_prefit_request_id;
                job = &create_job(name);
                job->state = "completed";
            }
            else if (name == "garment_relax")
            {
                ++state.garment_relax_request_id;
                job = &create_job(name);
                job->state = "completed";
            }
            else if (name == "save_constraints")
            {
                ++state.cloth_constraints_save_request_id;
                job = &create_job(name);
                job->state = "completed";
            }
            else if (name == "clear_constraints")
            {
                ++state.cloth_constraints_clear_request_id;
                job = &create_job(name);
                job->state = "completed";
            }
            else if (name == "clear_simulation")
            {
                ++state.cloth_clear_simulation_request_id;
                job = &create_job(name);
                job->state = "completed";
            }
            else if (name == "undo_canvas")
            {
                const bool applied = canva_apply_undo(canva);
                job = &create_job(name);
                job->state = "completed";
                job->message = applied ? "undo applied" : "undo stack was empty";
            }
            else if (name == "reset_canvas")
            {
                canva_reset_all(state, canva, renderer);
                job = &create_job(name);
                job->state = "completed";
            }
            else
            {
                throw std::runtime_error("unsupported operation: " + name);
            }

            return job_json(*job, state, canva);
        }

        json execute(const json& request, AppState& state, AssetIndex& assets, Renderer& renderer)
        {
            const json id = request.value("id", json(nullptr));
            try
            {
                if (!request.contains("method") || !request["method"].is_string())
                    throw std::runtime_error("method must be a string");
                const std::string method = request["method"].get<std::string>();
                const json params = request.value("params", json::object());
                if (!params.is_object())
                    throw std::runtime_error("params must be an object");

                CanvaState& canva = canva_state();
                canva_ensure_initialized(state, canva);
                update_jobs(state, canva);

                if (method == "app.status")
                    return success(id, app_status(state, canva));
                if (method == "app.logs")
                {
                    const int requested = params.value("limit", 100);
                    const int limit = std::clamp(requested, 1, 2000);
                    const std::size_t start = state.console.lines.size() > static_cast<std::size_t>(limit)
                        ? state.console.lines.size() - static_cast<std::size_t>(limit) : 0;
                    json lines = json::array();
                    for (std::size_t i = start; i < state.console.lines.size(); ++i)
                        lines.push_back(state.console.lines[i]);
                    return success(id, { { "lines", std::move(lines) }, { "total_retained", state.console.lines.size() } });
                }
                if (method == "app.shutdown")
                {
                    app_shutdown_requested.store(true);
                    return success(id, { { "requested", true } });
                }
                if (method == "assets.list")
                {
                    return success(id, {
                        { "poses", assets.poses() },
                        { "garments", assets.garments() },
                        { "meshes", assets.meshes() }
                    });
                }
                if (method == "assets.refresh")
                {
                    assets.refresh();
                    state.selection.poses = assets.poses();
                    state.selection.garments = assets.garments();
                    state.selection.meshes = assets.meshes();
                    ++state.asset_refresh_request_id;
                    return success(id, {
                        { "pose_count", assets.poses().size() },
                        { "garment_count", assets.garments().size() },
                        { "mesh_count", assets.meshes().size() }
                    });
                }
                if (method == "selection.set")
                {
                    if (params.contains("gender"))
                    {
                        const std::string value = params["gender"].get<std::string>();
                        if (value == "male") state.selection.gender = SmplType::Male;
                        else if (value == "female") state.selection.gender = SmplType::Female;
                        else if (value == "neutral") state.selection.gender = SmplType::Neutral;
                        else throw std::runtime_error("gender must be male, female, or neutral");
                    }
                    const auto set_asset = [&](const char* key, const std::vector<std::string>& values, int& index)
                    {
                        if (!params.contains(key))
                            return;
                        if (params[key].is_null() || params[key].get<std::string>().empty())
                        {
                            index = -1;
                            return;
                        }
                        const std::string name = params[key].get<std::string>();
                        const int found = find_name(values, name);
                        if (found < 0)
                            throw std::runtime_error(std::string("unknown ") + key + ": " + name);
                        index = found;
                    };
                    set_asset("pose", state.selection.poses, state.selection.pose_index);
                    set_asset("garment", state.selection.garments, state.selection.garment_index);
                    set_asset("mesh", state.selection.meshes, state.selection.mesh_index);
                    return success(id, app_status(state, canva)["selection"]);
                }
                if (method == "parameters.describe")
                {
                    json parameters = json::array();
                    for (const ParameterSpec& spec : make_parameter_specs())
                        parameters.push_back(parameter_schema_json(spec));
                    return success(id, { { "parameters", std::move(parameters) } });
                }
                if (method == "parameters.get")
                {
                    const std::vector<ParameterSpec> specs = make_parameter_specs();
                    json values = json::object();
                    if (params.contains("names"))
                    {
                        if (!params["names"].is_array())
                            throw std::runtime_error("names must be an array");
                        for (const json& item : params["names"])
                        {
                            const std::string name = item.get<std::string>();
                            const auto it = std::find_if(specs.begin(), specs.end(), [&](const ParameterSpec& spec) { return spec.name == name; });
                            if (it == specs.end())
                                throw std::runtime_error("unknown parameter: " + name);
                            values[name] = it->get(state, canva);
                        }
                    }
                    else
                    {
                        for (const ParameterSpec& spec : specs)
                            values[spec.name] = spec.get(state, canva);
                    }
                    return success(id, { { "values", std::move(values) } });
                }
                if (method == "parameters.set")
                {
                    if (!params.contains("values") || !params["values"].is_object())
                        throw std::runtime_error("values must be an object");
                    const std::vector<ParameterSpec> specs = make_parameter_specs();
                    std::vector<std::pair<const ParameterSpec*, const json*>> updates;
                    for (auto it = params["values"].begin(); it != params["values"].end(); ++it)
                    {
                        const auto spec = std::find_if(specs.begin(), specs.end(), [&](const ParameterSpec& candidate) { return candidate.name == it.key(); });
                        if (spec == specs.end())
                            throw std::runtime_error("unknown parameter: " + it.key());
                        validate_parameter_value(*spec, it.value());
                        updates.emplace_back(&*spec, &it.value());
                    }
                    for (const auto& update : updates)
                        update.first->set(state, canva, *update.second);
                    state.canva_brush_radius = std::clamp(state.canva_brush_radius, 0.5f, 250.0f);
                    canva.brush_radius = state.canva_brush_radius;
                    json values = json::object();
                    for (const auto& update : updates)
                        values[update.first->name] = update.first->get(state, canva);
                    return success(id, { { "values", std::move(values) } });
                }
                if (method == "layers.list")
                {
                    json layers = json::array();
                    for (int i = 0; i < static_cast<int>(canva.layers.size()); ++i)
                        layers.push_back(layer_json(canva, canva.layers[static_cast<std::size_t>(i)], i));
                    return success(id, { { "layers", std::move(layers) } });
                }
                if (method == "layers.import")
                {
                    if (!params.contains("path") || !params["path"].is_string())
                        throw std::runtime_error("path is required");
                    const std::filesystem::path path = std::filesystem::absolute(params["path"].get<std::string>()).lexically_normal();
                    if (!std::filesystem::is_regular_file(path))
                        throw std::runtime_error("input image does not exist: " + path.string());
                    std::string error;
                    if (!canva_add_png_layer(canva, path.string().c_str(), &error))
                        throw std::runtime_error(error.empty() ? "failed to import image" : error);
                    if (params.contains("name") && canva.active_layer >= 0)
                        canva.layers[static_cast<std::size_t>(canva.active_layer)].name = params["name"].get<std::string>();
                    return success(id, layer_json(canva, canva.layers[static_cast<std::size_t>(canva.active_layer)], canva.active_layer));
                }
                if (method == "layers.set")
                {
                    const int layer_id = params.value("layer_id", 0);
                    const int index = canva_find_layer_index_by_id(canva, layer_id);
                    if (index < 0)
                        throw std::runtime_error("layer not found: " + std::to_string(layer_id));
                    CanvaLayer& layer = canva.layers[static_cast<std::size_t>(index)];
                    if (params.contains("selected") && params["selected"].get<bool>()) canva.active_layer = index;
                    if (params.contains("name")) layer.name = params["name"].get<std::string>();
                    if (params.contains("visible")) layer.enabled = params["visible"].get<bool>();
                    if (params.contains("offset_x")) layer.offset.x = params["offset_x"].get<float>();
                    if (params.contains("offset_y")) layer.offset.y = params["offset_y"].get<float>();
                    if (params.contains("scale"))
                    {
                        const float scale = params["scale"].get<float>();
                        if (!std::isfinite(scale) || scale <= 0.0f)
                            throw std::runtime_error("scale must be finite and greater than zero");
                        layer.scale = scale;
                    }
                    return success(id, layer_json(canva, layer, index));
                }
                if (method == "layers.remove")
                {
                    const int layer_id = params.value("layer_id", 0);
                    const int index = canva_find_layer_index_by_id(canva, layer_id);
                    if (index < 0)
                        throw std::runtime_error("layer not found: " + std::to_string(layer_id));
                    canva_delete_layer_texture(canva.layers[static_cast<std::size_t>(index)]);
                    canva_prune_undo_for_layer(canva.undo, layer_id);
                    canva.layers.erase(canva.layers.begin() + index);
                    if (canva.layers.empty())
                        canva.layers.emplace_back(canva_make_layer(canva.width, canva.height, canva.next_layer_id++, "Layer " + std::to_string(canva.next_layer_number++)));
                    canva.active_layer = std::clamp(canva.active_layer, 0, static_cast<int>(canva.layers.size()) - 1);
                    return success(id, { { "removed_layer_id", layer_id }, { "selected_layer_id", canva.layers[static_cast<std::size_t>(canva.active_layer)].id } });
                }
                if (method == "layers.reorder")
                {
                    const int layer_id = params.value("layer_id", 0);
                    const int old_index = canva_find_layer_index_by_id(canva, layer_id);
                    if (old_index < 0)
                        throw std::runtime_error("layer not found: " + std::to_string(layer_id));
                    const int new_index = params.value("index", -1);
                    if (new_index < 0 || new_index >= static_cast<int>(canva.layers.size()))
                        throw std::runtime_error("index is outside the layer stack");
                    CanvaLayer layer = std::move(canva.layers[static_cast<std::size_t>(old_index)]);
                    canva.layers.erase(canva.layers.begin() + old_index);
                    canva.layers.insert(canva.layers.begin() + new_index, std::move(layer));
                    canva.active_layer = new_index;
                    return success(id, layer_json(canva, canva.layers[static_cast<std::size_t>(new_index)], new_index));
                }
                if (method == "layers.export")
                {
                    const int layer_id = params.value("layer_id", 0);
                    const int index = canva_find_layer_index_by_id(canva, layer_id);
                    if (index < 0)
                        throw std::runtime_error("layer not found: " + std::to_string(layer_id));
                    const std::filesystem::path path = checked_output_path(params);
                    std::string error;
                    if (!write_rgba_png(path, canva.width, canva.height, canva.layers[static_cast<std::size_t>(index)].pixels, &error))
                        throw std::runtime_error(error);
                    return success(id, { { "path", path.string() }, { "layer_id", layer_id }, { "width", canva.width }, { "height", canva.height } });
                }
                if (method == "layers.export_composite")
                {
                    const std::filesystem::path path = checked_output_path(params);
                    std::vector<unsigned char> rgba;
                    std::string error;
                    if (!canva_composite_visible_layers_rgba(canva, &rgba, &error, params.value("sketch_layers_only", false)))
                        throw std::runtime_error(error.empty() ? "failed to composite layers" : error);
                    if (!write_rgba_png(path, canva.width, canva.height, rgba, &error))
                        throw std::runtime_error(error);
                    return success(id, { { "path", path.string() }, { "width", canva.width }, { "height", canva.height } });
                }
                if (method == "operations.list")
                {
                    return success(id, { { "operations", {
                        "simulation", "generate_cloth", "full_generation", "generate_and_fit",
                        "render_cloth", "cloth_fitting", "extract_pose", "rig_optimization",
                        "generate_displacement", "generate_iso_displacement", "scene_snapshot",
                        "garment_pushout", "garment_ease", "garment_prefit", "garment_relax",
                        "save_constraints", "clear_constraints", "clear_simulation", "undo_canvas", "reset_canvas"
                    } } });
                }
                if (method == "operations.start")
                    return success(id, start_operation(state, canva, renderer, params));
                if (method == "operations.cancel")
                {
                    const std::uint64_t job_id = params.value("job_id", std::uint64_t(0));
                    const auto it = std::find_if(jobs.begin(), jobs.end(), [&](const Job& job) { return job.id == job_id; });
                    if (it == jobs.end())
                        throw std::runtime_error("job not found: " + std::to_string(job_id));
                    if (it->kind != "simulation")
                        throw std::runtime_error("only simulation jobs currently support safe cancellation");
                    ++state.cloth_cancel_simulation_request_id;
                    it->state = "cancelled";
                    it->message = "cancellation requested";
                    return success(id, job_json(*it, state, canva));
                }
                if (method == "jobs.get")
                {
                    const std::uint64_t job_id = params.value("job_id", std::uint64_t(0));
                    const auto it = std::find_if(jobs.begin(), jobs.end(), [&](const Job& job) { return job.id == job_id; });
                    if (it == jobs.end())
                        throw std::runtime_error("job not found: " + std::to_string(job_id));
                    return success(id, job_json(*it, state, canva));
                }
                if (method == "jobs.list")
                {
                    const int requested = params.value("limit", 50);
                    const std::size_t limit = static_cast<std::size_t>(std::clamp(requested, 1, 500));
                    const std::size_t start = jobs.size() > limit ? jobs.size() - limit : 0;
                    json result = json::array();
                    for (std::size_t i = start; i < jobs.size(); ++i)
                        result.push_back(job_json(jobs[i], state, canva));
                    return success(id, { { "jobs", std::move(result) } });
                }
                throw std::runtime_error("unknown method: " + method);
            }
            catch (const std::exception& error)
            {
                return failure(id, error.what());
            }
        }

#if !defined(_WIN32)
        bool write_all(int fd, const std::string& text)
        {
            const char* data = text.data();
            std::size_t remaining = text.size();
            while (remaining > 0)
            {
                const ssize_t count = ::send(fd, data, remaining, MSG_NOSIGNAL);
                if (count < 0 && errno == EINTR)
                    continue;
                if (count <= 0)
                    return false;
                data += count;
                remaining -= static_cast<std::size_t>(count);
            }
            return true;
        }

        void handle_client(int fd)
        {
            std::string line;
            char buffer[4096];
            while (line.size() <= kMaxRequestBytes)
            {
                const ssize_t count = ::read(fd, buffer, sizeof(buffer));
                if (count < 0 && errno == EINTR)
                    continue;
                if (count <= 0)
                    break;
                line.append(buffer, static_cast<std::size_t>(count));
                const std::size_t newline = line.find('\n');
                if (newline != std::string::npos)
                {
                    line.resize(newline);
                    break;
                }
            }

            json response;
            try
            {
                if (line.size() > kMaxRequestBytes)
                    throw std::runtime_error("request exceeds 4 MiB");
                auto pending = std::make_shared<PendingRequest>();
                pending->request = json::parse(line);
                std::future<json> future = pending->response.get_future();
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    queue.push_back(pending);
                }
                if (future.wait_for(std::chrono::seconds(15)) != std::future_status::ready)
                    throw std::runtime_error("application main thread did not answer within 15 seconds");
                response = future.get();
            }
            catch (const std::exception& error)
            {
                response = failure(nullptr, error.what());
            }
            write_all(fd, response.dump() + "\n");
        }

        void listener_loop()
        {
            while (!stop_requested.load())
            {
                pollfd descriptor{ listener_fd, POLLIN, 0 };
                const int ready = ::poll(&descriptor, 1, 200);
                if (ready < 0 && errno == EINTR)
                    continue;
                if (ready <= 0)
                    continue;
                const int client = ::accept(listener_fd, nullptr, nullptr);
                if (client < 0)
                    continue;
                handle_client(client);
                ::close(client);
            }
        }
#endif
    };

    AutomationServer::AutomationServer() : m_impl(std::make_unique<Impl>()) {}
    AutomationServer::~AutomationServer() { stop(); }

    bool AutomationServer::start(std::string* out_error)
    {
        if (m_impl->running.load())
            return true;
#if defined(_WIN32)
        if (out_error)
            *out_error = "PoseWeaver automation currently requires the Linux/WSL build";
        return false;
#else
        const char* configured = std::getenv("POSEWEAVER_AUTOMATION_SOCKET");
        m_impl->socket_path = configured && configured[0] != '\0'
            ? configured : "/tmp/poseweaver-" + std::to_string(static_cast<long long>(getuid())) + ".sock";
        if (m_impl->socket_path.size() >= sizeof(sockaddr_un::sun_path))
        {
            if (out_error) *out_error = "automation socket path is too long: " + m_impl->socket_path;
            return false;
        }

        m_impl->listener_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (m_impl->listener_fd < 0)
        {
            if (out_error) *out_error = std::string("failed to create automation socket: ") + std::strerror(errno);
            return false;
        }

        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::strncpy(address.sun_path, m_impl->socket_path.c_str(), sizeof(address.sun_path) - 1);
        if (std::filesystem::exists(m_impl->socket_path))
        {
            const int probe = ::socket(AF_UNIX, SOCK_STREAM, 0);
            const bool occupied = probe >= 0
                && ::connect(probe, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
            if (probe >= 0)
                ::close(probe);
            if (occupied)
            {
                if (out_error) *out_error = "another PoseWeaver instance owns " + m_impl->socket_path;
                ::close(m_impl->listener_fd);
                m_impl->listener_fd = -1;
                return false;
            }
            ::unlink(m_impl->socket_path.c_str());
        }
        if (::bind(m_impl->listener_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
            || ::chmod(m_impl->socket_path.c_str(), S_IRUSR | S_IWUSR) != 0
            || ::listen(m_impl->listener_fd, 8) != 0)
        {
            if (out_error) *out_error = std::string("failed to bind automation socket: ") + std::strerror(errno);
            ::close(m_impl->listener_fd);
            m_impl->listener_fd = -1;
            ::unlink(m_impl->socket_path.c_str());
            return false;
        }

        m_impl->stop_requested.store(false);
        m_impl->running.store(true);
        m_impl->listener = std::thread(&Impl::listener_loop, m_impl.get());
        return true;
#endif
    }

    void AutomationServer::pump(AppState& state, AssetIndex& assets, Renderer& renderer)
    {
        std::vector<std::shared_ptr<Impl::PendingRequest>> pending;
        {
            std::lock_guard<std::mutex> lock(m_impl->queue_mutex);
            pending.swap(m_impl->queue);
        }
        for (const auto& item : pending)
            item->response.set_value(m_impl->execute(item->request, state, assets, renderer));
        CanvaState& canva = canva_state();
        canva_ensure_initialized(state, canva);
        m_impl->update_jobs(state, canva);
    }

    void AutomationServer::stop()
    {
        if (!m_impl || !m_impl->running.exchange(false))
            return;
        m_impl->stop_requested.store(true);
#if !defined(_WIN32)
        if (m_impl->listener_fd >= 0)
            ::shutdown(m_impl->listener_fd, SHUT_RDWR);
        if (m_impl->listener.joinable())
            m_impl->listener.join();
        if (m_impl->listener_fd >= 0)
        {
            ::close(m_impl->listener_fd);
            m_impl->listener_fd = -1;
        }
        ::unlink(m_impl->socket_path.c_str());
#endif
    }

    const std::string& AutomationServer::socket_path() const { return m_impl->socket_path; }
    bool AutomationServer::running() const { return m_impl->running.load(); }
    bool AutomationServer::shutdown_requested() const { return m_impl->app_shutdown_requested.load(); }
}
