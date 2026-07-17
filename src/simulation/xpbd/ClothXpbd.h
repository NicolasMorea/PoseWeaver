#pragma once

#include <Eigen/Core>

#include <atomic>
#include <string>
#include <vector>

namespace poseweaver
{
    struct ClothXpbdParams
    {
        double dt = 1.0 / 60.0;
        double duration_seconds = 2.0;
        int solver_iterations = 20;
        int substeps = 2; // XPBD substeps per frame; increases stability and constraint convergence

        double gravity_y = -9.81;
        double damping = 0.005;

        bool stretch_enabled = true;
        double stretch_compliance = 0.0;
        double base_stretch_mult = 1.0;
        bool bend_enabled = true;
        double bend_compliance = 1e-4;
        bool collision_enabled = true;
        double collision_compliance = 0.0;

        double collision_thickness = 0.005;

        bool self_collision_enabled = false;
        double self_collision_compliance = 0.0;
        double self_collision_thickness = 0.005;

        bool enable_friction = false;
        double friction_mu = 0.3;

        Eigen::Vector3d wind_dir = Eigen::Vector3d(0.0, 0.0, 1.0);
        double wind_force = 0.0;
        double wind_random_magnitude = 0.0;
        double wind_random_time_scale = 1.0;
        double wind_random_spatial_scale = 1.0;

        int max_frames = 300;

        // Optional rest/reference cloth used for masses and constraint rest lengths.
        // When empty, cloth_V0 is used as both the initial and reference state.
        Eigen::MatrixXd reference_positions;

        std::vector<Eigen::MatrixXd> body_sequence;
        Eigen::MatrixXd body_capsules; // #C x 15 rows: ax ay az bx by bz ru rv rw ux uy uz vx vy vz
        std::vector<Eigen::MatrixXd> body_capsule_sequence;
        std::vector<int> fixed_vertex_indices;
        std::vector<Eigen::MatrixXd> fixed_vertex_target_sequence; // each (#fixed x 3)
    };

    struct ClothXpbdStats
    {
        int steps_simulated = 0;
        int frames_recorded = 0;
        int last_step_contacts = 0;

        double ms_total = 0.0;
        double ms_bvh_build = 0.0;
        double ms_predict = 0.0;
        double ms_wind = 0.0;
        double ms_candidates = 0.0;
        double ms_stretch = 0.0;
        double ms_bend = 0.0;

        double ms_collision_bvh = 0.0;    // total contact-finding time
        double ms_collision_narrow = 0.0; // TriangleBvh::closest_triangle() time
        double ms_collision_broad = 0.0;  // contact-finding time excluding BVH queries

        double ms_collision_solve = 0.0;
        double ms_self_collision_bvh = 0.0;
        double ms_self_collision_solve = 0.0;
        double ms_friction = 0.0;
        int last_step_self_contacts = 0;
    };

    struct ClothXpbdResult
    {
        std::vector<Eigen::MatrixXd> frames;
        ClothXpbdStats stats;
    };

    using ClothXpbdOnFrameFn = void (*)(const Eigen::MatrixXd& frame_V, const Eigen::MatrixXd* body_V, int steps_simulated, void* user);

    bool run_cloth_xpbd(
        const Eigen::MatrixXd& cloth_V0,
        const Eigen::MatrixXi& cloth_F,
        const Eigen::MatrixXd& body_V,
        const Eigen::MatrixXi& body_F,
        const ClothXpbdParams& params,
        ClothXpbdResult* out,
        std::string* error,
        std::atomic_bool* cancel = nullptr,
        std::atomic_int* progress_steps_done = nullptr,
        ClothXpbdOnFrameFn on_frame = nullptr,
        void* on_frame_user = nullptr);
}
