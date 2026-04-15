#pragma once

#include "core/Geometry.h"

#include <Eigen/Core>
#include <limits>
#include <vector>

namespace poseweaver::collision
{
    using core::Aabb;

    class TriangleBvh
    {
    public:
        void build(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
        void refit(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

        const Aabb& bounds() const { return m_bounds; }

        // Finds the closest triangle on the mesh to point p
        // false if BVH is empty.
        bool closest_triangle( 
            const Eigen::Vector3d& p,
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            int& out_face,
            Eigen::Vector3d& out_closest,
            double& out_dist2,
            double max_dist2 = std::numeric_limits<double>::infinity()) const;

    private:
        struct Node
        {
            Aabb box;
            int left = -1;
            int right = -1;
            int start = 0;
            int count = 0;
            bool leaf = false;
        };

        int build_node(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int start, int count);
        Aabb triangle_bounds(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int face) const;
        Eigen::Vector3d triangle_centroid(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int face) const;

        std::vector<Node> m_nodes;
        std::vector<int> m_faces; // indices into F
        Aabb m_bounds;
    };
}
