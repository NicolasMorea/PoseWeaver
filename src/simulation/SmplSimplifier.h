#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

namespace poseweaver
{
    struct SmplSimplification
    {
        Eigen::MatrixXi F;
        std::vector<int> source_to_simplified;
        std::vector<int> cluster_counts;

        int source_vertex_count = 0;
        int requested_vertex_count = 0;

        bool valid() const
        {
            return source_vertex_count > 0
                && (int)source_to_simplified.size() == source_vertex_count
                && (int)cluster_counts.size() > 0
                && F.rows() > 0;
        }
    };

    bool build_smpl_simplification(
        const Eigen::MatrixXd& reference_V,
        const Eigen::MatrixXi& source_F,
        const Eigen::MatrixXd& skinning_weights,
        double keep_percent,
        SmplSimplification& out,
        std::string* error = nullptr);

    bool apply_smpl_simplification(
        const Eigen::MatrixXd& source_V,
        const SmplSimplification& simplification,
        Eigen::MatrixXd& out_V,
        std::string* error = nullptr);
}
