#pragma once

#include "smpl/Smplh.h"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace poseweaver
{
    struct SmplCollisionCapsule
    {
        int joint_a = -1;
        int joint_b = -1;
        double t0 = 0.0;
        double t1 = 1.0;
        double radius = 0.03;
        double radius_u = 0.03;
        double radius_v = 0.03;
        double radius_w = 0.03;
        double offset_a_u = 0.0;
        double offset_a_v = 0.0;
        double offset_b_u = 0.0;
        double offset_b_v = 0.0;
        int frame_joint_a = -1;
        int frame_joint_b = -1;
    };

    struct SmplCollisionProxy
    {
        std::vector<SmplCollisionCapsule> capsules;

        bool valid() const { return !capsules.empty(); }
    };

    bool build_smpl_collision_proxy(
        const Eigen::MatrixXd& reference_body_V,
        const SmplhModel& model,
        SmplCollisionProxy& out,
        std::string* error = nullptr);

    bool apply_smpl_collision_proxy(
        const SmplCollisionProxy& proxy,
        const Eigen::MatrixXd& body_V,
        const SmplhModel& model,
        Eigen::MatrixXd& out_V,
        Eigen::MatrixXi& out_F,
        std::string* error = nullptr);

    bool apply_smpl_collision_proxy_capsules(
        const SmplCollisionProxy& proxy,
        const Eigen::MatrixXd& body_V,
        const SmplhModel& model,
        Eigen::MatrixXd& out_capsules,
        std::string* error = nullptr);
}
