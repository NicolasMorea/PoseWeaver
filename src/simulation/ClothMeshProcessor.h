#pragma once

#include <Eigen/Core>
#include <string>

namespace poseweaver
{
    struct ClothPushOutResult
    {
        int moved_vertices = 0;
    };

    struct ClothEaseResult
    {
        int affected_edges = 0;
        int affected_bend_edges = 0;
        double max_vertex_displacement = 0.0;
    };

    struct ClothRelaxResult
    {
        int iterations_done = 0;
        int edges_applied = 0;
        double max_vertex_displacement = 0.0;
    };

    struct ClothPrefitRelaxParams
    {
        int iterations = 10;

        double stretch_weight = 1.0;
        double smooth_weight = 0.05;
        double body_clearance = 0.003;
        double collision_weight = 1.0;
        double tether_weight = 0.1; // stay near kinematic target
        double max_vertex_step = 0.01; // clamp per-iteration motion

        bool enable_stretch = true;
        bool enable_smoothing = true;
        bool enable_collision = true;
        bool enable_tether = true;
    };

    struct ClothPrefitRelaxResult
    {
        int iterations_done = 0;
        int stretch_constraints_applied = 0;
        int collision_constraints_applied = 0;
        double max_vertex_displacement = 0.0;
    };

    struct ClothRenderRefineResult
    {
        int passes = 0;
        int added_vertices = 0;
        int output_faces = 0;
        double max_edge_before = 0.0;
        double max_edge_after = 0.0;
    };

    // Pushes penetrating cloth vertices outside the body surface by projecting them to the closest point on the body mesh and adding a small outward offset
    bool push_cloth_vertices_outside_body(
        Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        double surface_offset,
        ClothPushOutResult* result,
        std::string* error);

    // Applies one relaxation step that pulls the cloth toward its rest pose
    bool ease_cloth_toward_rest_pose(
        Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& cloth_V_rest,
        double amount,
        ClothEaseResult* result,
        std::string* error);

    // Milder relaxation than ease_cloth_toward_rest_pose()
    bool relax_cloth_preserve_stretch(
        Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& cloth_V_rest,
        int iterations,
        double edge_alpha,
        double smooth_lambda,
        ClothRelaxResult* result,
        std::string* error);

    bool relax_cloth_prefit(
        Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& cloth_V_rest,
        const Eigen::MatrixXd& cloth_V_target,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const ClothPrefitRelaxParams& params,
        ClothPrefitRelaxResult* result,
        std::string* error);

    bool refine_cloth_mesh_for_render(
        const Eigen::MatrixXd& cloth_V,
        const Eigen::MatrixXi& cloth_F,
        double target_edge_length,
        Eigen::MatrixXd& out_V,
        Eigen::MatrixXi& out_F,
        ClothRenderRefineResult* result,
        std::string* error,
        const Eigen::MatrixXd* cloth_UV = nullptr,
        Eigen::MatrixXd* out_UV = nullptr,
        Eigen::VectorXi* out_source_faces = nullptr);
}
