#pragma once

#include <Eigen/Core>

#include <atomic>
#include <string>
#include <vector>

namespace poseweaver
{
    struct ClothPdParams
    {
        double dt = 1.0 / 60.0;
        double duration_seconds = 2.0;
        int local_global_iterations = 10;

        double gravity_y = -9.81;
        double damping = 0.01;

        // Stiffness values are expected to be >= 0. Larger values mean stiffer constraints.
        double stretch_stiffness = 1.0;
        double base_stretch_mult = 1.0;

        // Bending models:
        // - bend_stiffness: dihedral-angle hinge constraint (existing)
        // - iso_bend_stiffness: isometric bending (distance between opposite vertices across a hinge)
        double bend_stiffness = 0.5;
        double iso_bend_stiffness = 0.0;

        double collision_stiffness = 1.0;

        double collision_thickness = 0.003;

        bool enable_friction = false;
        double friction_mu = 0.3;

        // Wind (treated as an acceleration-like external force).
        // - wind_dir: world direction (does not need to be normalized)
        // - wind_force: magnitude (0 disables)
        // - wind_random_space/time: smooth variation strengths (0 disables)
        Eigen::Vector3d wind_dir = Eigen::Vector3d(0.0, 0.0, 1.0);
        double wind_force = 0.0;
        double wind_random_space = 0.0;
        double wind_random_time = 0.0;

        int max_frames = 300;

        std::vector<int> fixed_vertex_indices;
        Eigen::MatrixXd fixed_vertex_targets; // (#fixed x 3), in simulation/world space
    };

    struct ClothPdStats
    {
        int steps_simulated = 0;
        int frames_recorded = 0;
        int last_step_contacts = 0;

        // Timing (milliseconds, accumulated over the whole simulation run)
        double ms_total = 0.0;
        double ms_bvh_build = 0.0;
        double ms_predict = 0.0;
        double ms_wind = 0.0;
        double ms_contacts_bvh = 0.0; // total time inside contact finding (broad + BVH query + bookkeeping)
        double ms_contacts_query = 0.0; // time spent inside TriangleBvh::closest_triangle
        double ms_contacts_other = 0.0; // contact finding time excluding BVH query
        double ms_assemble = 0.0;
        double ms_factor = 0.0;
        double ms_local_stretch = 0.0;
        double ms_local_bend = 0.0;
        double ms_local_collision = 0.0;
        double ms_global_solve = 0.0;
        double ms_post_pushout = 0.0;
        double ms_friction = 0.0;
    };

    struct ClothPdResult
    {
        std::vector<Eigen::MatrixXd> frames; // each is (#V x 3)
        ClothPdStats stats;
    };

    using ClothPdOnFrameFn = void (*)(const Eigen::MatrixXd& frame_V, int steps_simulated, void* user);

    // Runs an offline cloth simulation and records frames for playback.
    // - cloth_V0: initial cloth vertices (will be frame 0)
    // - cloth_F:  triangle indices
    // - body_V/body_F: collision mesh (treated static during the run)
    bool run_cloth_projective_dynamics(
        const Eigen::MatrixXd& cloth_V0,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const ClothPdParams& params,
        ClothPdResult* out,
        std::string* error,
        std::atomic_bool* cancel = nullptr,
        std::atomic_int* progress_steps_done = nullptr,
        ClothPdOnFrameFn on_frame = nullptr,
        void* on_frame_user = nullptr);
}
